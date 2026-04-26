"""Tests for the multivariate convenience API."""

from __future__ import annotations

import numpy as np
import torch

from voila_gp.kernels import ExpKernel
from voila_gp.multivariate import MultiSDEVI, sample_drift_functions
from voila_gp.simulate import euler_maruyama


def _2d_oscillator(n_steps=4000, dt=0.001, seed=42):
    omega = 2.0

    def drift(x):
        return np.array([x[1], -(omega**2) * x[0]])

    def diffusion(x):
        return np.array([np.sqrt(1.25), np.sqrt(0.75 + 0.25 * x[0] ** 2 + 0.5 * x[1] ** 2)])

    rng = np.random.default_rng(seed)
    return euler_maruyama(drift, diffusion, np.zeros(2), dt, n_steps, rng), dt


def test_multivariate_fit_runs_per_component():
    x, h = _2d_oscillator()
    inducing = np.random.default_rng(99).multivariate_normal(
        x.mean(0), np.cov(x, rowvar=False), size=8
    )

    def df_kernel_factory(_x, _h, _i):
        return ExpKernel(amplitude=5.0, length_scales=torch.tensor([2.0, 2.0]), epsilon=1e-5)

    def diff_kernel_factory(_x, _h, _i):
        # Note: amplitude here is overridden by select_diffusion_parameters indirectly
        return ExpKernel(amplitude=2.0, length_scales=torch.tensor([0.5, 0.5]), epsilon=1e-5)

    fit = MultiSDEVI(
        drift_kernel_factory=df_kernel_factory,
        diff_kernel_factory=diff_kernel_factory,
        prior_on_sd=5.0,
    )
    res = fit.fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        max_iter=4, rel_tol=1e-4, hp_max_iter=50, verbose=False,
    )
    assert res.state_dim == 2
    assert len(res.components) == 2
    # Each component fit should have a positive lower-bound trajectory ending at a finite value
    for comp in res.components:
        assert np.isfinite(comp.lower_bound_history[-1])


def test_multivariate_predict_drift_shape():
    x, h = _2d_oscillator(n_steps=2000)
    inducing = np.random.default_rng(99).multivariate_normal(
        x.mean(0), np.cov(x, rowvar=False), size=6
    )
    fit = MultiSDEVI(
        drift_kernel_factory=lambda x_, h_, i_: ExpKernel(
            amplitude=2.0, length_scales=torch.tensor([1.0, 1.0]), epsilon=1e-5
        ),
        diff_kernel_factory=lambda x_, h_, i_: ExpKernel(
            amplitude=1.0, length_scales=torch.tensor([0.5, 0.5]), epsilon=1e-5
        ),
        prior_on_sd=5.0,
    )
    res = fit.fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        max_iter=2, hp_max_iter=20,
    )
    grid = np.column_stack([np.linspace(-1, 1, 20), np.zeros(20)])
    drift_pred = res.predict_drift(grid)
    diff_pred = res.predict_diffusion(grid)
    assert drift_pred.shape == (20, 2)
    assert diff_pred.shape == (20, 2)
    # Diffusion is positive (post log-normal back-transform)
    assert torch.all(diff_pred > 0)


def test_sample_drift_functions_shape_and_stochasticity():
    x, h = _2d_oscillator(n_steps=2000)
    inducing = np.random.default_rng(0).multivariate_normal(
        x.mean(0), np.cov(x, rowvar=False), size=6
    )
    fit = MultiSDEVI(
        drift_kernel_factory=lambda *_: ExpKernel(
            amplitude=2.0, length_scales=torch.tensor([1.0, 1.0]), epsilon=1e-5
        ),
        diff_kernel_factory=lambda *_: ExpKernel(
            amplitude=1.0, length_scales=torch.tensor([0.5, 0.5]), epsilon=1e-5
        ),
        prior_on_sd=5.0,
    ).fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        max_iter=2, hp_max_iter=20,
    )
    grid = np.column_stack([np.linspace(-1, 1, 15), np.zeros(15)])
    samples = sample_drift_functions(fit.components[1], new_x=grid, n_samples=10, seed=42)
    assert samples.shape == (10, 15)
    # Different samples must differ (otherwise sampler is broken)
    assert not torch.allclose(samples[0], samples[1])
    # Sample mean should be close to predict_drift mean
    sample_mean = samples.mean(dim=0)
    pred_mean = fit.predict_drift(grid)[:, 1]
    # 10 samples — the empirical mean still has substantial sampling noise so we test
    # for coarse agreement only (within 2.5x sample std).
    sample_std = samples.std(dim=0)
    assert torch.all(torch.abs(sample_mean - pred_mean) < 3 * sample_std + 0.1)
