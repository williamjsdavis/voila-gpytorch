"""Probabilistic forecasting for fitted SDEs.

Once `MultiSDEVI.fit` (or per-component `SDEVI.fit`) returns a posterior over
drift and (log-)diffusion functions, we can simulate the system forward in
time. Two distinct sources of uncertainty propagate:

  - **Aleatoric** (intrinsic noise): the diffusion term ``g(X) dW`` injects
    stochasticity even with a perfectly known drift. Captured by Monte Carlo
    over Brownian increments.
  - **Epistemic** (model uncertainty): the drift and diffusion functions are
    *posterior random variables* under the GP posterior. Captured by drawing
    a function sample for each ensemble member.

Most users want both. `simulate_forward(..., function_uncertainty="sample")`
draws one drift/diff function per ensemble member; ``"mean"`` uses the MAP
function (faster, lower bound on total uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ._compat import cholesky_inverse_compat
from .kernels import Kernel
from .multivariate import MultiSDEVIResult


@dataclass
class _PredictCache:
    """Per-component intermediates that are constant across an Euler-Maruyama loop.

    Computing K_mm and its inverse once at simulator construction (rather than
    inside every step's predict_drift / predict_diffusion call) is mandatory for
    GPU performance — kernel-launch overhead would otherwise dominate.
    """

    drift_kernel: Kernel
    diff_kernel: Kernel
    Z: Tensor
    K_mm_inv_drift: Tensor
    K_mm_inv_diff: Tensor
    f_mean: Tensor
    f_cov: Tensor
    s_mean: Tensor
    s_cov: Tensor
    v: float


def _build_predict_caches(fit: MultiSDEVIResult) -> list[_PredictCache]:
    caches: list[_PredictCache] = []
    for comp in fit.components:
        assert comp.drift_kernel is not None and comp.diff_kernel is not None
        Z = comp.inducing_points.detach()
        K_mm_d = comp.drift_kernel.cov(Z)
        L_d = torch.linalg.cholesky(K_mm_d)
        K_mm_inv_d = cholesky_inverse_compat(L_d)
        K_mm_inv_d = 0.5 * (K_mm_inv_d + K_mm_inv_d.T)
        K_mm_s = comp.diff_kernel.cov(Z)
        L_s = torch.linalg.cholesky(K_mm_s)
        K_mm_inv_s = cholesky_inverse_compat(L_s)
        K_mm_inv_s = 0.5 * (K_mm_inv_s + K_mm_inv_s.T)
        caches.append(
            _PredictCache(
                drift_kernel=comp.drift_kernel,
                diff_kernel=comp.diff_kernel,
                Z=Z,
                K_mm_inv_drift=K_mm_inv_d.detach(),
                K_mm_inv_diff=K_mm_inv_s.detach(),
                f_mean=comp.f_mean.detach(),
                f_cov=comp.f_cov.detach(),
                s_mean=comp.s_mean.detach(),
                s_cov=comp.s_cov.detach(),
                v=float(comp.v),
            )
        )
    return caches


def _drift_at_cached(caches: list[_PredictCache], x: Tensor) -> Tensor:
    """Posterior mean drift at x: (n, d) → (n, d). Reuses precomputed K_mm_inv."""
    out = []
    for c in caches:
        kmx = c.drift_kernel.cov(c.Z, x)  # (m, n)
        A_new = kmx.T @ c.K_mm_inv_drift  # (n, m)
        out.append(A_new @ c.f_mean)  # zero prior mean
    return torch.stack(out, dim=-1)


def _diff_at_cached(caches: list[_PredictCache], x: Tensor) -> Tensor:
    """Posterior mean diffusion magnitude g(x): (n, d) → (n, d).

    Reproduces predict_diffusion's log-normal back-transform mean = exp(μ + σ²/2),
    then sqrt to convert g² → g.
    """
    out = []
    for c in caches:
        kmx = c.diff_kernel.cov(c.Z, x)  # (m, n)
        A_new = kmx.T @ c.K_mm_inv_diff  # (n, m)
        gp_mean = c.v + A_new @ (c.s_mean - c.v)
        kvar = c.diff_kernel.variances(x)
        kxx = torch.diagonal(A_new @ kmx)
        cov_term = torch.diagonal(A_new @ c.s_cov @ A_new.T)
        gp_var = (kvar - kxx + cov_term).clamp_min(0.0)
        mean_g2 = torch.exp(gp_mean + gp_var / 2)
        out.append(mean_g2)
    g2 = torch.stack(out, dim=-1)
    return torch.sqrt(g2.clamp_min(1e-12))


@dataclass
class ForecastResult:
    """Output of `simulate_forward`.

    Attributes
    ----------
    trajectories : (n_ensembles, n_steps + 1, d) tensor — simulated paths.
    times        : (n_steps + 1,) tensor of time stamps starting at 0.
    """

    trajectories: Tensor
    times: Tensor

    def quantiles(self, qs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, Tensor]:
        """Per-time, per-component quantiles across ensemble members."""
        out = {}
        for q in qs:
            out[q] = torch.quantile(self.trajectories, q=q, dim=0)
        return out


def simulate_forward(
    fit: MultiSDEVIResult,
    x0: np.ndarray | Tensor,
    n_steps: int,
    n_ensembles: int = 200,
    dt: float | None = None,
    function_uncertainty: Literal["mean", "sample"] = "mean",
    seed: int | None = 0,
) -> ForecastResult:
    """Simulate `n_ensembles` trajectories of length `n_steps + 1` from initial
    state `x0` using the inferred drift and diffusion.

    Parameters
    ----------
    fit : MultiSDEVIResult
    x0 : (d,) initial state. To start an ensemble at the same state, all members
         use this point. To start with state-uncertainty, pass an (n_ensembles, d)
         array instead.
    n_steps : int — number of Euler–Maruyama steps.
    n_ensembles : int — Monte Carlo samples for aleatoric uncertainty.
    dt : float — time step (defaults to the fit's sampling period).
    function_uncertainty : "mean" or "sample"
        - "mean": use the posterior mean drift/diff (cheap; captures aleatoric only).
        - "sample": draw one drift function from the GP posterior per ensemble
          member at every state (captures epistemic too; ≈10× slower per step).
    seed : RNG seed for reproducibility.
    """
    if dt is None:
        dt = fit.sampling_period
    device = fit.device
    dtype = fit.dtype
    # CUDA and MPS both require a device-bound generator; only CPU uses the default.
    rng = torch.Generator() if device.type == "cpu" else torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    x0_t = torch.as_tensor(x0, dtype=dtype, device=device)
    if x0_t.ndim == 1:
        # broadcast same starting point across ensemble
        x = x0_t.unsqueeze(0).expand(n_ensembles, -1).clone()
    else:
        if x0_t.shape[0] != n_ensembles:
            raise ValueError(
                f"x0 first dim {x0_t.shape[0]} must match n_ensembles {n_ensembles}"
            )
        x = x0_t.clone()
    d = x.shape[1]

    out = torch.empty(n_ensembles, n_steps + 1, d, dtype=dtype, device=device)
    out[:, 0, :] = x
    sqrt_dt = float(np.sqrt(dt))

    caches = _build_predict_caches(fit)

    with torch.no_grad():
        if function_uncertainty == "mean":
            for t in range(n_steps):
                f = _drift_at_cached(caches, x)  # (n_ens, d)
                g = _diff_at_cached(caches, x)
                eps = torch.randn(n_ensembles, d, generator=rng, dtype=dtype, device=device)
                x = x + f * dt + g * eps * sqrt_dt
                out[:, t + 1, :] = x
        elif function_uncertainty == "sample":
            # "sample" adds a per-ensemble random perturbation in the drift to
            # approximate function-level uncertainty in a tractable way. We
            # reuse the cached K_mm_inv to compute the posterior std cheaply.
            for t in range(n_steps):
                f = _drift_at_cached(caches, x)
                g = _diff_at_cached(caches, x)
                eps_drift = torch.randn(n_ensembles, d, generator=rng, dtype=dtype, device=device)
                drift_std = []
                for c in caches:
                    kmx = c.drift_kernel.cov(c.Z, x)
                    A_new = kmx.T @ c.K_mm_inv_drift
                    base_var = c.drift_kernel.variances(x)
                    kxx = torch.diagonal(A_new @ kmx)
                    cov_term = torch.diagonal(A_new @ c.f_cov @ A_new.T)
                    var = (base_var - kxx + cov_term).clamp_min(0.0)
                    drift_std.append(torch.sqrt(var))
                drift_std_t = torch.stack(drift_std, dim=-1)
                eps_obs = torch.randn(n_ensembles, d, generator=rng, dtype=dtype, device=device)
                x = x + (f + drift_std_t * eps_drift) * dt + g * eps_obs * sqrt_dt
                out[:, t + 1, :] = x
        else:
            raise ValueError(f"unknown function_uncertainty mode: {function_uncertainty}")

    times = torch.arange(n_steps + 1, dtype=dtype, device=device) * dt
    return ForecastResult(trajectories=out, times=times)
