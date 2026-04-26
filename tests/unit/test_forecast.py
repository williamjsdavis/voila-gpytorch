"""Tests for the forecasting module."""

from __future__ import annotations

import numpy as np
import torch

from voila_gp.forecast import simulate_forward
from voila_gp.kernels import ExpKernel
from voila_gp.multivariate import MultiSDEVI
from voila_gp.simulate import euler_maruyama


def _ou_2d(n_steps=2000, dt=0.001, seed=0):
    """Decoupled 2D OU: f(x) = -x; g(x) = sqrt(1.0)·I."""

    def drift(x):
        return -x

    def diffusion(x):
        return np.array([1.0, 1.0])

    rng = np.random.default_rng(seed)
    return euler_maruyama(drift, diffusion, np.zeros(2), dt, n_steps, rng), dt


def _fit_2d(x, h):
    inducing = np.random.default_rng(99).multivariate_normal(
        x.mean(0), np.cov(x, rowvar=False), size=6
    )
    fit = MultiSDEVI(
        drift_kernel_factory=lambda *_: ExpKernel(
            amplitude=2.0, length_scales=torch.tensor([1.0, 1.0]), epsilon=1e-5
        ),
        diff_kernel_factory=lambda *_: ExpKernel(
            amplitude=1.0, length_scales=torch.tensor([1.0, 1.0]), epsilon=1e-5
        ),
        prior_on_sd=5.0,
    )
    return fit.fit(time_series=x, sampling_period=h, inducing_points=inducing,
                   max_iter=3, hp_max_iter=30)


def test_simulate_forward_returns_ensemble_with_correct_shape():
    x, h = _ou_2d(n_steps=500)
    res = _fit_2d(x, h)
    fc = simulate_forward(res, x0=np.zeros(2), n_steps=200, n_ensembles=50, dt=h)
    assert fc.trajectories.shape == (50, 201, 2)
    assert fc.times.shape == (201,)
    assert torch.all(fc.times >= 0)


def test_simulate_forward_ensemble_diverges_through_diffusion():
    """OU process started from origin: ensemble spread should grow with time."""
    x, h = _ou_2d(n_steps=1000)
    res = _fit_2d(x, h)
    fc = simulate_forward(res, x0=np.zeros(2), n_steps=500, n_ensembles=100, dt=h)
    early_std = fc.trajectories[:, 10, :].std(dim=0)
    late_std = fc.trajectories[:, 500, :].std(dim=0)
    assert torch.all(late_std > early_std)


def test_simulate_forward_quantiles_widening():
    x, h = _ou_2d(n_steps=1000)
    res = _fit_2d(x, h)
    fc = simulate_forward(res, x0=np.zeros(2), n_steps=300, n_ensembles=200, dt=h)
    qs = fc.quantiles((0.05, 0.95))
    width_early = float((qs[0.95][10] - qs[0.05][10]).abs().sum())
    width_late = float((qs[0.95][300] - qs[0.05][300]).abs().sum())
    assert width_late > width_early


def test_simulate_forward_function_uncertainty_sample_mode_runs():
    """The slower 'sample' mode should run and return finite trajectories."""
    x, h = _ou_2d(n_steps=500)
    res = _fit_2d(x, h)
    fc = simulate_forward(
        res, x0=np.zeros(2), n_steps=50, n_ensembles=20, dt=h,
        function_uncertainty="sample",
    )
    assert torch.all(torch.isfinite(fc.trajectories))
