"""Multivariate convenience API: fit each component independently and bundle results.

The reference voila algorithm operates on one component at a time (selected via
`target_index`), so a multivariate Langevin SDE in d dimensions requires d
independent VI runs. This module wraps that pattern in a single call and
returns a `MultiSDEVIResult` whose `simulate_forward` and prediction utilities
make it easy to reason about the joint dynamics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .inference import SDEVI, SDEVIResult
from .init_heuristics import select_diffusion_parameters
from .kernels import Kernel
from .prediction import predict_diffusion, predict_drift


@dataclass
class MultiSDEVIResult:
    """Bundle of per-component SDEVI results for a d-dimensional system.

    `components[d]` is the SDEVIResult for the dth dimension's drift/diffusion.
    Drift and diffusion at new points are batched across components via the
    `predict_drift` / `predict_diffusion` methods.
    """

    components: list[SDEVIResult]
    sampling_period: float
    target_indices: list[int]

    @property
    def state_dim(self) -> int:
        return len(self.components)

    def predict_drift(self, new_x: np.ndarray | Tensor) -> Tensor:
        """Posterior mean drift vector at `new_x` — shape (n_new, state_dim)."""
        x_t = torch.as_tensor(new_x, dtype=torch.float64)
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        out = []
        for comp in self.components:
            assert comp.drift_kernel is not None
            pred = predict_drift(
                kernel=comp.drift_kernel,
                inducing_points=comp.inducing_points,
                posterior_mean=comp.f_mean,
                posterior_cov=comp.f_cov,
                new_x=x_t,
            )
            out.append(pred["mean"])
        return torch.stack(out, dim=-1)

    def predict_diffusion(self, new_x: np.ndarray | Tensor) -> Tensor:
        """Posterior mean diagonal diffusion vector g²(x) — shape (n_new, state_dim).

        Each component is modelled with its own log-normal GP; voila assumes
        diagonal diffusion.
        """
        x_t = torch.as_tensor(new_x, dtype=torch.float64)
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        out = []
        for comp in self.components:
            assert comp.diff_kernel is not None
            pred = predict_diffusion(
                kernel=comp.diff_kernel,
                inducing_points=comp.inducing_points,
                posterior_mean=comp.s_mean,
                posterior_cov=comp.s_cov,
                new_x=x_t, v=comp.v,
            )
            out.append(pred["mean"])
        return torch.stack(out, dim=-1)


KernelFactory = Callable[[np.ndarray, float, int], Kernel]


class MultiSDEVI:
    """Run voila VI independently per component, sharing config across them.

    Usage::

        fit = MultiSDEVI(
            drift_kernel_factory=lambda x, h, i: ExpKernel(amplitude=5.0, ...),
            diff_kernel_factory=lambda x, h, i: ExpConstKernel(...),
            prior_on_sd=5.0,
        )
        res = fit.fit(time_series, sampling_period, inducing_points, components=[0, 1])
    """

    def __init__(
        self,
        drift_kernel_factory: KernelFactory,
        diff_kernel_factory: KernelFactory,
        prior_on_sd: float = 5.0,
    ) -> None:
        self.drift_kernel_factory = drift_kernel_factory
        self.diff_kernel_factory = diff_kernel_factory
        self.prior_on_sd = float(prior_on_sd)

    def fit(
        self,
        time_series: np.ndarray | Tensor,
        sampling_period: float,
        inducing_points: np.ndarray | Tensor,
        *,
        components: list[int] | None = None,
        max_iter: int = 10,
        rel_tol: float = 1e-6,
        hp_max_iter: int = 100,
        verbose: bool = False,
    ) -> MultiSDEVIResult:
        ts = np.asarray(time_series, dtype=np.float64)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        d = ts.shape[1]
        if components is None:
            components = list(range(d))

        results: list[SDEVIResult] = []
        for target in components:
            if verbose:
                print(f"\n--- fitting component {target} of {d} ---")
            drift_kernel = self.drift_kernel_factory(ts, sampling_period, target)
            diff_params = select_diffusion_parameters(
                ts, sampling_period=sampling_period,
                prior_on_sd=self.prior_on_sd, target_index=target,
            )
            diff_kernel = self.diff_kernel_factory(ts, sampling_period, target)
            fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)
            res = fit.fit(
                time_series=ts, sampling_period=sampling_period,
                inducing_points=inducing_points, v_init=diff_params["v"],
                target_index=target,
                max_iter=max_iter, rel_tol=rel_tol,
                hp_max_iter=hp_max_iter, verbose=verbose,
            )
            results.append(res)
        return MultiSDEVIResult(
            components=results,
            sampling_period=float(sampling_period),
            target_indices=list(components),
        )


def sample_drift_functions(
    fit: SDEVIResult,
    new_x: np.ndarray | Tensor,
    n_samples: int,
    seed: int | None = None,
) -> Tensor:
    """Draw `n_samples` independent realizations of the drift function from the
    GP posterior at `new_x`.

    Returns a tensor of shape (n_samples, n_new). The samples capture the
    posterior uncertainty as functional draws, not just pointwise variance.
    """
    from .sparse_gp import sparse_gp_intermediates

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    xnew = torch.as_tensor(new_x, dtype=torch.float64)
    if xnew.ndim == 1:
        xnew = xnew.reshape(-1, 1)
    assert fit.drift_kernel is not None, "fit must come from SDEVI.fit() with drift_kernel set"
    drift_kernel = fit.drift_kernel
    Z = fit.inducing_points
    inter = sparse_gp_intermediates(drift_kernel, xnew, Z)
    K_mm_inv = inter["K_mm_inv"]
    kmx = drift_kernel.cov(Z, xnew)  # (m, n_new)
    A_new = kmx.T @ K_mm_inv  # (n_new, m)
    mean = A_new @ fit.f_mean  # (n_new,)
    # full predictive covariance (not just diagonal)
    base_cov = drift_kernel.cov(xnew)  # (n_new, n_new)
    cov = base_cov - A_new @ kmx + A_new @ fit.f_cov @ A_new.T
    cov = 0.5 * (cov + cov.T)
    # Add tiny jitter for Cholesky
    n = cov.shape[0]
    cov = cov + 1e-9 * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(cov)
    eps = torch.randn(n_samples, n, generator=g, dtype=torch.float64)
    samples = mean.detach() + eps @ L.T.detach()
    return samples
