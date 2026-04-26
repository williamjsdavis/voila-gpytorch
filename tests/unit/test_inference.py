"""TDD tests for voila_gp.inference — outer alternating loop + L-BFGS-B hyperparams."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voila_gp.inference import SDEVI, SDEVIResult
from voila_gp.kernels import ExpKernel


def _toy_ou(n=2000, dt=0.001, seed=0):
    g = torch.Generator().manual_seed(seed)
    # OU: X_{t+1} = X_t - X_t · dt + sqrt(1.5 · dt) · ε
    x = torch.zeros(n + 1, 1, dtype=torch.float64)
    sqrt_g = float(np.sqrt(1.5 * dt))
    eps = torch.randn(n, generator=g, dtype=torch.float64)
    for t in range(n):
        x[t + 1, 0] = x[t, 0] - x[t, 0] * dt + sqrt_g * eps[t]
    return x.numpy()


def test_inference_returns_result_object():
    x = _toy_ou()
    drift = ExpKernel(amplitude=1.5, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    diff = ExpKernel(amplitude=0.5, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    inducing = np.linspace(x.min(), x.max(), 6).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff)
    res = fit.fit(
        time_series=x, sampling_period=0.001,
        inducing_points=inducing, v_init=0.4, max_iter=2,
        rel_tol=1e-3, hp_max_iter=20, verbose=False,
    )
    assert isinstance(res, SDEVIResult)
    assert res.lower_bound_history[-1] > res.lower_bound_history[0]
    assert res.f_mean.shape == (6,)
    assert res.f_cov.shape == (6, 6)
    assert res.s_mean.shape == (6,)
    assert res.s_cov.shape == (6, 6)
    assert res.inducing_points.shape == (6, 1)
    assert isinstance(res.v, float)


def test_inference_lower_bound_monotone_per_outer_iteration():
    """Voila's reference implementation has L non-decreasing across outer iterations
    after the first warmup step (each outer iter does posterior + hyperparam updates).
    """
    x = _toy_ou()
    drift = ExpKernel(amplitude=1.5, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    diff = ExpKernel(amplitude=0.5, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    inducing = np.linspace(x.min(), x.max(), 6).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff)
    res = fit.fit(
        time_series=x, sampling_period=0.001,
        inducing_points=inducing, v_init=0.4, max_iter=4,
        rel_tol=1e-4, hp_max_iter=30, verbose=False,
    )
    # After-hyperparam-update lower bounds are at indices 0, 2, 4, ... in the history
    # (position 0 = initial; position 2k+1 = after distributions update at iter k+1;
    #  position 2k+2 = after HP optimization at iter k+1). We expect that the
    # AFTER-HP values are non-decreasing iteration-to-iteration.
    after_hp = res.lower_bound_history[2::2]
    assert after_hp == sorted(after_hp), f"L decreased after-HP across iterations: {after_hp}"


@pytest.mark.slow
def test_inference_recovers_OU_drift_qualitatively():
    """End-to-end qualitative validation against the known OU process.

    True drift f(x) = -x. Posterior drift mean evaluated on the prediction support
    should have negative slope close to -1.
    """
    x = _toy_ou(n=8000, dt=0.001, seed=1)
    drift = ExpKernel(amplitude=2.0, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    diff = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.5]), epsilon=1e-6)
    inducing = np.linspace(np.quantile(x, 0.05), np.quantile(x, 0.95), 8).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff)
    res = fit.fit(
        time_series=x, sampling_period=0.001,
        inducing_points=inducing, v_init=0.4, max_iter=8,
        rel_tol=1e-5, hp_max_iter=50, verbose=False,
    )
    # f_mean is the posterior mean of drift at the (optimized) inducing points.
    z = res.inducing_points.squeeze()
    f = res.f_mean
    # Linear regression slope of f on z (1D)
    z_np = z.detach().numpy() if torch.is_tensor(z) else z
    f_np = f.detach().numpy() if torch.is_tensor(f) else f
    slope, intercept = np.polyfit(z_np, f_np, 1)
    # True slope is -1; require posterior slope within ±0.4 (loose: 8000 samples is short)
    assert -1.5 < slope < -0.5
    # Drift at zero should be near zero
    assert abs(intercept) < 0.5


def test_inference_initial_state_zero_mean_kmm_cov():
    """Initial drift posterior is f_mean=0, f_cov=K_mm (matching voila constructor)."""
    x = _toy_ou(n=1000)
    drift = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    diff = ExpKernel(amplitude=0.5, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    inducing = np.linspace(x.min(), x.max(), 4).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff)
    init = fit.initialize(
        time_series=x, sampling_period=0.001, inducing_points=inducing, v_init=0.5
    )
    assert torch.allclose(init.f_mean, torch.zeros(4, dtype=torch.float64))
    # f_cov starts equal to K_mm; check positive-definite and symmetric
    assert torch.allclose(init.f_cov, init.f_cov.T)
    assert float(torch.linalg.eigvalsh(init.f_cov).min()) > 0
