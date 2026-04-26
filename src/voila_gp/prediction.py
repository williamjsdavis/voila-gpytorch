"""Posterior prediction for drift and diffusion — port of voila/R/sde_prediction.R.

Drift uses a zero-mean GP. Diffusion uses a log-normal GP with prior mean v;
the back-transform is:
    μ_lognorm = exp(μ_gp + σ²_gp / 2)
    σ²_lognorm = (exp(σ²_gp) − 1) · exp(2 μ_gp + σ²_gp)
    quantiles_lognorm = exp(qnorm(p, μ_gp, σ_gp))
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor

from ._compat import ndtri_compat
from .kernels import Kernel
from .sparse_gp import sparse_gp_intermediates


def _qnorm(p: float, mean: Tensor, sd: Tensor) -> Tensor:
    """Inverse CDF of a normal distribution. Uses erfinv (works on every backend)."""
    p_t = torch.tensor(p, dtype=mean.dtype, device=mean.device)
    z = ndtri_compat(p_t)
    return mean + z * sd


def _gp_posterior_at_new(
    kernel: Kernel,
    inducing_points: Tensor,
    posterior_mean: Tensor,
    posterior_cov: Tensor,
    new_x: Tensor,
    prior_mean_at_Z: Tensor,
    prior_mean_at_new: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute predictive mean and variance of the underlying GP at `new_x`.

    Reference R `predict.sgp_sde`:
        kmx = k(xm, newX);  A = kmx.T @ kmmInv
        mu  = prior_mean(newX) + A @ (posteriorMean - prior_mean(xm))
        var = vars(newX) - diag(A @ kmx) + diag(A @ posteriorCov @ A.T)
    """
    inter = sparse_gp_intermediates(kernel, new_x, inducing_points)
    K_mm_inv = inter["K_mm_inv"]
    kmx = kernel.cov(inducing_points, new_x)  # (m, n_new)
    A_new = kmx.T @ K_mm_inv  # (n_new, m)
    mean = prior_mean_at_new + A_new @ (posterior_mean - prior_mean_at_Z)
    var = (
        kernel.variances(new_x)
        - torch.diagonal(A_new @ kmx)
        + torch.diagonal(A_new @ posterior_cov @ A_new.T)
    )
    return mean, var.clamp_min(0.0)


def _to_tensor(
    x,
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if isinstance(x, Tensor):
        if device is not None and x.device != device:
            x = x.to(device=device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x
    target_dtype = dtype if dtype is not None else torch.float64
    target_device = device if device is not None else torch.device("cpu")
    return torch.as_tensor(np.asarray(x), dtype=target_dtype, device=target_device)


def predict_drift(
    *,
    kernel: Kernel,
    inducing_points,
    posterior_mean,
    posterior_cov,
    new_x,
    quantiles: Iterable[float] = (0.05, 0.95),
) -> dict:
    """Posterior of drift f(·) at `new_x`. Prior mean is identically zero."""
    device = kernel.device
    dtype = kernel.dtype
    Z = _to_tensor(inducing_points, "inducing_points", device=device, dtype=dtype)
    mu_post = _to_tensor(posterior_mean, "posterior_mean", device=device, dtype=dtype)
    cov_post = _to_tensor(posterior_cov, "posterior_cov", device=device, dtype=dtype)
    Xnew = _to_tensor(new_x, "new_x", device=device, dtype=dtype)
    if Xnew.ndim == 1:
        Xnew = Xnew.reshape(-1, 1)

    prior_at_Z = torch.zeros_like(mu_post)
    prior_at_new = torch.zeros(Xnew.shape[0], dtype=Xnew.dtype, device=Xnew.device)
    mean, var = _gp_posterior_at_new(
        kernel, Z, mu_post, cov_post, Xnew, prior_at_Z, prior_at_new
    )
    sd = torch.sqrt(var)
    quantile_tensor = torch.stack(
        [_qnorm(float(q), mean, sd) for q in quantiles], dim=-1
    )
    return {
        "mean": mean,
        "var": var,
        "quantiles": quantile_tensor,
        "x": Xnew,
        "lognormal": False,
    }


def predict_diffusion(
    *,
    kernel: Kernel,
    inducing_points,
    posterior_mean,
    posterior_cov,
    new_x,
    v: float,
    quantiles: Iterable[float] = (0.05, 0.95),
) -> dict:
    """Posterior of g²(·) (i.e. squared diffusion) at `new_x`, log-normal prior."""
    device = kernel.device
    dtype = kernel.dtype
    Z = _to_tensor(inducing_points, "inducing_points", device=device, dtype=dtype)
    mu_post = _to_tensor(posterior_mean, "posterior_mean", device=device, dtype=dtype)
    cov_post = _to_tensor(posterior_cov, "posterior_cov", device=device, dtype=dtype)
    Xnew = _to_tensor(new_x, "new_x", device=device, dtype=dtype)
    if Xnew.ndim == 1:
        Xnew = Xnew.reshape(-1, 1)

    prior_at_Z = torch.full_like(mu_post, v)
    prior_at_new = torch.full((Xnew.shape[0],), v, dtype=Xnew.dtype, device=Xnew.device)
    gp_mean, gp_var = _gp_posterior_at_new(
        kernel, Z, mu_post, cov_post, Xnew, prior_at_Z, prior_at_new
    )
    gp_sd = torch.sqrt(gp_var)
    # Quantiles in the underlying-GP space, then back-transformed
    gp_quantiles = torch.stack(
        [_qnorm(float(q), gp_mean, gp_sd) for q in quantiles], dim=-1
    )

    mean = torch.exp(gp_mean + gp_var / 2)
    var = (torch.exp(gp_var) - 1) * torch.exp(2 * gp_mean + gp_var)
    quantile_tensor = torch.exp(gp_quantiles)
    return {
        "mean": mean,
        "var": var,
        "quantiles": quantile_tensor,
        "x": Xnew,
        "lognormal": True,
    }
