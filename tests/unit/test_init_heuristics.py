"""TDD tests for voila_gp.init_heuristics — port of select_diffusion_parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from voila_gp.init_heuristics import mad, select_diffusion_parameters


def test_mad_matches_R_default_constant():
    # R's mad(x) defaults to constant=1.4826 * median(|x - median(x)|)
    x = np.array([1.0, 2.0, 2.5, 3.0, 100.0])  # 100 is an outlier
    med = np.median(x)
    expected = 1.4826 * np.median(np.abs(x - med))
    assert mad(x) == pytest.approx(expected, rel=1e-12)


def test_select_diffusion_parameters_formula_1d():
    # Reference R formula:
    #   varX = mad(diff(x))²
    #   kernelAmplitude = log(1 + (priorOnSd · h / varX)²)
    #   v = log(varX / h) - kernelAmplitude / 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    h = 0.01
    prior_on_sd = 3.0
    out = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=prior_on_sd)
    var_x = mad(np.diff(x)) ** 2
    expected_amp = math.log(1 + (prior_on_sd * h / var_x) ** 2)
    expected_v = math.log(var_x / h) - expected_amp / 2
    assert out["kernel_amplitude"] == pytest.approx(expected_amp, rel=1e-12)
    assert out["v"] == pytest.approx(expected_v, rel=1e-12)


def test_select_diffusion_parameters_accepts_torch_tensor():
    x = torch.tensor([0.0, 0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.0])
    out = select_diffusion_parameters(x, sampling_period=0.1, prior_on_sd=1.0)
    assert math.isfinite(out["v"])
    assert out["kernel_amplitude"] > 0


def test_select_diffusion_parameters_target_index_selects_column():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 2))
    # target_index=0 should ignore column 1
    out0 = select_diffusion_parameters(X, sampling_period=0.01, prior_on_sd=1.0, target_index=0)
    out1 = select_diffusion_parameters(X, sampling_period=0.01, prior_on_sd=1.0, target_index=1)
    # different columns, different stats
    assert out0["v"] != out1["v"]
    # Sanity: each matches the 1D call on the corresponding column
    out0_1d = select_diffusion_parameters(X[:, 0], sampling_period=0.01, prior_on_sd=1.0)
    assert out0["v"] == pytest.approx(out0_1d["v"], rel=1e-12)
