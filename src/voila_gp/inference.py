"""Outer variational-inference loop — port of `do_inference` in
voila/src/sde_variational_inferencer.cpp.

Alternates:
    1. Closed-form drift posterior update.
    2. Laplace diffusion posterior update.
    3. Hyperparameter optimization via scipy L-BFGS-B on −L w.r.t. zipped vector
       [drift_kernel_hp, diff_kernel_hp, inducing_points (flat), v]
       with PyTorch autograd providing the gradient.

Convergence is on relative L change between outer iterations (matches reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

from ._compat import _default_dtype_for
from .elbo import SparseSDEModel, lower_bound
from .kernels import Kernel
from .posterior import update_diffusion_laplace, update_drift_closed_form
from .sparse_gp import sparse_gp_intermediates


@dataclass
class SDEVIResult:
    """Output of `SDEVI.fit`."""

    f_mean: Tensor
    f_cov: Tensor
    s_mean: Tensor
    s_cov: Tensor
    inducing_points: Tensor
    v: float
    lower_bound_history: list[float]
    converged: bool
    iterations: int

    # The kernels (with optimized hyperparameters) are kept on the result so the
    # caller can invoke prediction without re-bookkeeping them.
    drift_kernel: Kernel | None = None
    diff_kernel: Kernel | None = None
    head_x: Tensor | None = None  # (n-1, d) base points; needed by predict
    sampling_period: float = 0.0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64


@dataclass
class _State:
    """Mutable model state during the alternation."""

    head_x: Tensor
    target: Tensor
    sampling_period: float
    inducing_points: Tensor
    v: float
    f_mean: Tensor
    f_cov: Tensor
    s_mean: Tensor
    s_cov: Tensor
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64


class SDEVI:
    """Variational inference driver.

    Usage::

        fit = SDEVI(drift_kernel, diff_kernel)
        res = fit.fit(time_series, sampling_period, inducing_points, v_init, ...)
    """

    def __init__(
        self,
        drift_kernel: Kernel,
        diff_kernel: Kernel,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.drift_kernel = drift_kernel
        self.diff_kernel = diff_kernel
        resolved = torch.device(device) if device is not None else drift_kernel.device
        if drift_kernel.device != resolved or diff_kernel.device != resolved:
            raise ValueError(
                "drift_kernel.device, diff_kernel.device, and SDEVI(device=...) must agree; got "
                f"drift={drift_kernel.device}, diff={diff_kernel.device}, sdevi={resolved}"
            )
        self.device = resolved
        self.dtype = dtype if dtype is not None else _default_dtype_for(resolved)

    # ------------------------------------------------------------------ utils
    def _model_matrices(self, head_x: Tensor, inducing_points: Tensor) -> tuple[dict, dict]:
        drift = sparse_gp_intermediates(self.drift_kernel, head_x, inducing_points)
        diff = sparse_gp_intermediates(self.diff_kernel, head_x, inducing_points)
        return drift, diff

    def _build_model(self, st: _State, drift: dict, diff: dict) -> SparseSDEModel:
        return SparseSDEModel(
            dx=st.target,
            sampling_period=st.sampling_period,
            f_mean=st.f_mean,
            f_cov=st.f_cov,
            A=drift["A"],
            Q_ii=drift["Q_ii"],
            K_mm_inv=drift["K_mm_inv"],
            v=st.v,
            s_mean=st.s_mean,
            s_cov=st.s_cov,
            B=diff["A"],
            H_ii=diff["Q_ii"],
            J_mm_inv=diff["K_mm_inv"],
        )

    # --------------------------------------------------------------- public
    def initialize(
        self,
        time_series: np.ndarray | Tensor,
        sampling_period: float,
        inducing_points: np.ndarray | Tensor,
        v_init: float,
        target_index: int = 0,
    ) -> _State:
        """Build the initial state matching voila's constructor."""
        ts = torch.as_tensor(time_series, dtype=self.dtype, device=self.device)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        head_x = ts[:-1].clone()
        target = (ts[1:] - ts[:-1])[:, target_index].clone()
        ip = torch.as_tensor(inducing_points, dtype=self.dtype, device=self.device).clone()
        if ip.ndim == 1:
            ip = ip.reshape(-1, 1)
        m = ip.shape[0]
        # initial posteriors: f_mean=0, f_cov=K_mm; s_mean=v, s_cov=J_mm
        with torch.no_grad():
            drift0, diff0 = self._model_matrices(head_x, ip)
        return _State(
            head_x=head_x,
            target=target,
            sampling_period=float(sampling_period),
            inducing_points=ip,
            v=float(v_init),
            f_mean=torch.zeros(m, dtype=self.dtype, device=self.device),
            f_cov=drift0["K_mm"].detach().clone(),
            s_mean=torch.full((m,), float(v_init), dtype=self.dtype, device=self.device),
            s_cov=diff0["K_mm"].detach().clone(),
            device=self.device,
            dtype=self.dtype,
        )

    # -------------------------------------------------------- hp pack / bounds
    def _zip_hp(self, st: _State) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Pack hyperparameters into a flat numpy vector + bounds.

        Order: drift_kernel_hp | diff_kernel_hp | inducing_points (flat) | v
        """
        d_hp = self.drift_kernel.get_hyperparams().detach().cpu().numpy()
        f_hp = self.diff_kernel.get_hyperparams().detach().cpu().numpy()
        ip_flat = st.inducing_points.detach().cpu().numpy().reshape(-1)
        x = np.concatenate([d_hp, f_hp, ip_flat, np.array([st.v])])

        d_lb, d_ub = (b.cpu().numpy() for b in self.drift_kernel.hp_bounds())
        f_lb, f_ub = (b.cpu().numpy() for b in self.diff_kernel.hp_bounds())
        head = st.head_x.detach().cpu().numpy()
        n_pseudo, d = st.inducing_points.shape
        ip_lb = np.full(n_pseudo * d, head.min())
        ip_ub = np.full(n_pseudo * d, head.max())
        v_lb = np.array([-1e6])
        v_ub = np.array([1e6])
        lb = np.concatenate([d_lb, f_lb, ip_lb, v_lb])
        ub = np.concatenate([d_ub, f_ub, ip_ub, v_ub])

        layout = {
            "n_drift": d_hp.size,
            "n_diff": f_hp.size,
            "n_ip": ip_flat.size,
            "ip_shape": (n_pseudo, d),
        }
        return x, lb, ub, layout

    def _unzip_hp(self, x: np.ndarray, st: _State, layout: dict) -> None:
        """Inverse of `_zip_hp`. Mutates kernels and `st` in place."""
        i = 0
        d = torch.as_tensor(x[i : i + layout["n_drift"]], dtype=st.dtype, device=st.device)
        i += layout["n_drift"]
        f = torch.as_tensor(x[i : i + layout["n_diff"]], dtype=st.dtype, device=st.device)
        i += layout["n_diff"]
        ip = torch.as_tensor(x[i : i + layout["n_ip"]], dtype=st.dtype, device=st.device).reshape(
            layout["ip_shape"]
        )
        i += layout["n_ip"]
        v = float(x[i])
        self.drift_kernel.set_hyperparams(d)
        self.diff_kernel.set_hyperparams(f)
        with torch.no_grad():
            st.inducing_points.copy_(ip)
        st.v = v

    # ----------------------------------------------- hyperparameter step (LBFGSB)
    def _hp_objective(self, st: _State):
        """Build closure for scipy.optimize.minimize: returns (-L, -dL/dx) at x."""
        # Capture posterior tensors as detached constants — the hp step holds them fixed.
        f_mean = st.f_mean.detach().clone()
        f_cov = st.f_cov.detach().clone()
        s_mean = st.s_mean.detach().clone()
        s_cov = st.s_cov.detach().clone()
        target = st.target.detach()
        h = st.sampling_period
        head_x = st.head_x

        def fun(x_np: np.ndarray, layout):
            # Set kernel hyperparameters from x_np (no autograd graph yet).
            i = 0
            d_hp = torch.as_tensor(
                x_np[i : i + layout["n_drift"]], dtype=st.dtype, device=st.device
            )
            i += layout["n_drift"]
            f_hp = torch.as_tensor(
                x_np[i : i + layout["n_diff"]], dtype=st.dtype, device=st.device
            )
            i += layout["n_diff"]
            ip = torch.as_tensor(
                x_np[i : i + layout["n_ip"]], dtype=st.dtype, device=st.device
            ).reshape(layout["ip_shape"])
            i += layout["n_ip"]
            v_scalar = torch.as_tensor(x_np[i], dtype=st.dtype, device=st.device)

            # Re-assign kernel parameters in-place (preserves nn.Parameter identity)
            self.drift_kernel.set_hyperparams(d_hp)
            self.diff_kernel.set_hyperparams(f_hp)

            # Make ip and v leaf tensors with grad to capture their contribution
            ip_leaf = ip.clone().requires_grad_(True)
            v_leaf = v_scalar.clone().requires_grad_(True)
            # Re-enable grads on kernel parameters (they live as nn.Parameter)
            for kernel in (self.drift_kernel, self.diff_kernel):
                for p in kernel.parameters():
                    p.requires_grad_(True)
                    if p.grad is not None:
                        p.grad.zero_()

            drift = sparse_gp_intermediates(self.drift_kernel, head_x, ip_leaf)
            diff = sparse_gp_intermediates(self.diff_kernel, head_x, ip_leaf)
            model = SparseSDEModel(
                dx=target, sampling_period=h,
                f_mean=f_mean, f_cov=f_cov,
                A=drift["A"], Q_ii=drift["Q_ii"], K_mm_inv=drift["K_mm_inv"],
                v=v_leaf,
                s_mean=s_mean, s_cov=s_cov,
                B=diff["A"], H_ii=diff["Q_ii"], J_mm_inv=diff["K_mm_inv"],
            )
            L = lower_bound(model)
            neg = -L
            neg.backward()

            # Gather gradients in the same order as packing
            grads = []
            for n in self.drift_kernel._HP_NAMES:
                grads.append(getattr(self.drift_kernel, n).grad.detach().reshape(-1).cpu().numpy())
            for n in self.diff_kernel._HP_NAMES:
                grads.append(getattr(self.diff_kernel, n).grad.detach().reshape(-1).cpu().numpy())
            assert ip_leaf.grad is not None and v_leaf.grad is not None
            grads.append(ip_leaf.grad.detach().reshape(-1).cpu().numpy())
            grads.append(np.array([float(v_leaf.grad)]))
            grad_np = np.concatenate(grads)
            return float(neg.item()), grad_np

        return fun

    def _step_hyperparameters(self, st: _State, max_iter: int) -> None:
        x0, lb, ub, layout = self._zip_hp(st)
        bounds = list(zip(lb.tolist(), ub.tolist(), strict=False))
        objective = self._hp_objective(st)

        def fun_jac(x):
            return objective(x, layout)

        res = minimize(
            fun_jac, x0,
            method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9},
        )
        # Apply the optimizer's iterate back to the state (even if not converged).
        self._unzip_hp(res.x, st, layout)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        time_series: np.ndarray | Tensor,
        sampling_period: float,
        inducing_points: np.ndarray | Tensor,
        v_init: float,
        target_index: int = 0,
        max_iter: int = 10,
        rel_tol: float = 1e-6,
        hp_max_iter: int = 50,
        verbose: bool = False,
    ) -> SDEVIResult:
        st = self.initialize(
            time_series=time_series,
            sampling_period=sampling_period,
            inducing_points=inducing_points,
            v_init=v_init,
            target_index=target_index,
        )
        # Initial bound
        with torch.no_grad():
            drift0, diff0 = self._model_matrices(st.head_x, st.inducing_points)
            L0 = float(lower_bound(self._build_model(st, drift0, diff0)))
        history = [L0]
        if verbose:
            print(f"Initial L = {L0:.6g}")

        prev_L = L0
        converged = False
        n_done = 0
        for it in range(1, max_iter + 1):
            # Posterior updates (closed form drift, then Laplace diffusion)
            with torch.no_grad():
                drift, diff = self._model_matrices(st.head_x, st.inducing_points)
            model = self._build_model(st, drift, diff)
            d_new = update_drift_closed_form(model)
            st.f_mean, st.f_cov = d_new["f_mean"].detach(), d_new["f_cov"].detach()
            model = self._build_model(st, drift, diff)
            sd_new = update_diffusion_laplace(model)
            st.s_mean, st.s_cov = sd_new["s_mean"].detach(), sd_new["s_cov"].detach()

            with torch.no_grad():
                drift, diff = self._model_matrices(st.head_x, st.inducing_points)
                L_after_dist = float(lower_bound(self._build_model(st, drift, diff)))
            history.append(L_after_dist)
            if verbose:
                print(f"Iter {it}: distributions update L = {L_after_dist:.6g}")

            # Hyperparameter optimization (-L w.r.t. zipped vector, scipy L-BFGS-B)
            self._step_hyperparameters(st, max_iter=hp_max_iter)

            with torch.no_grad():
                drift, diff = self._model_matrices(st.head_x, st.inducing_points)
                L_after_hp = float(lower_bound(self._build_model(st, drift, diff)))
            history.append(L_after_hp)
            if verbose:
                print(f"Iter {it}: hyperparam optimization L = {L_after_hp:.6g}")

            n_done = it
            denom = max(1.0, abs(prev_L))
            if abs(L_after_hp - prev_L) / denom < rel_tol:
                converged = True
                break
            prev_L = L_after_hp

        return SDEVIResult(
            f_mean=st.f_mean,
            f_cov=st.f_cov,
            s_mean=st.s_mean,
            s_cov=st.s_cov,
            inducing_points=st.inducing_points.detach().clone(),
            v=st.v,
            lower_bound_history=history,
            converged=converged,
            iterations=n_done,
            drift_kernel=self.drift_kernel,
            diff_kernel=self.diff_kernel,
            head_x=st.head_x.detach().clone(),
            sampling_period=st.sampling_period,
            device=self.device,
            dtype=self.dtype,
        )
