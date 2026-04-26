"""Tests for the model-comparison diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from voila_gp.diagnostics import compare_models, predictive_log_likelihood
from voila_gp.inference import SDEVI
from voila_gp.kernels import ExpKernel
from voila_gp.simulate import euler_maruyama


def _simulate_ou(n_steps=2000, dt=0.001, seed=0):
    rng = np.random.default_rng(seed)
    return euler_maruyama(lambda x: -x, lambda x: np.array([1.0]), np.zeros(1), dt, n_steps, rng), dt


def _quick_fit(x, h, length_scale=1.0):
    drift = ExpKernel(amplitude=2.0, length_scales=torch.tensor([length_scale]), epsilon=1e-5)
    diff = ExpKernel(amplitude=1.0, length_scales=torch.tensor([length_scale]), epsilon=1e-5)
    inducing = np.linspace(x.min(), x.max(), 6).reshape(-1, 1)
    return SDEVI(drift_kernel=drift, diff_kernel=diff).fit(
        time_series=x, sampling_period=h,
        inducing_points=inducing, v_init=0.0, max_iter=3, hp_max_iter=30,
    )


def test_predictive_log_likelihood_finite_and_dx_consistent():
    train, h = _simulate_ou(seed=0)
    held_out, _ = _simulate_ou(seed=1)
    fit = _quick_fit(train, h)
    ll = predictive_log_likelihood(fit, held_out, sampling_period=h)
    assert np.isfinite(ll)
    # log-likelihood scales linearly with the number of increments
    half = predictive_log_likelihood(fit, held_out[: len(held_out) // 2 + 1], sampling_period=h)
    # Same per-increment likelihood density on average → ratio ≈ 0.5
    assert abs(half / ll - 0.5) < 0.1


def test_compare_models_returns_sorted_descending_log_likelihood():
    """compare_models returns results sorted from best (highest log-lik) to worst."""
    train, h = _simulate_ou(seed=0)
    held_out, _ = _simulate_ou(seed=2)
    fit_a = _quick_fit(train, h, length_scale=1.0)
    fit_b = _quick_fit(train, h, length_scale=0.5)
    fit_c = _quick_fit(train, h, length_scale=2.0)
    cmp = compare_models(
        {"a": fit_a, "b": fit_b, "c": fit_c},
        held_out_time_series=held_out, sampling_period=h,
    )
    # Sorted descending — first entry has the highest log-likelihood
    assert all(
        cmp.held_out_log_likelihoods[i] >= cmp.held_out_log_likelihoods[i + 1]
        for i in range(len(cmp.names) - 1)
    )


def test_compare_models_falls_back_to_elbo_when_no_holdout():
    train, h = _simulate_ou()
    fit_a = _quick_fit(train, h, length_scale=1.0)
    fit_b = _quick_fit(train, h, length_scale=0.5)
    cmp = compare_models({"a": fit_a, "b": fit_b})
    # Sorted by ELBO descending
    assert cmp.elbos[0] >= cmp.elbos[1]
