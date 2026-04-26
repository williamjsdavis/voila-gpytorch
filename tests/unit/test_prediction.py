"""TDD tests for voila_gp.prediction — posterior drift/diffusion at new points.

Reference: voila/R/sde_prediction.R lines 86-116 (predict.sgp_sde).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voila_gp.kernels import ExpKernel
from voila_gp.prediction import predict_diffusion, predict_drift
from voila_gp.sparse_gp import sparse_gp_intermediates


@pytest.fixture
def trained_state():
    """Toy state mimicking what `SDEVI.fit` returns."""
    torch.Generator().manual_seed(3)
    n_pseudo = 5
    Z = torch.linspace(-2.0, 2.0, n_pseudo, dtype=torch.float64).reshape(-1, 1)
    kernel = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    f_mean = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)  # ≈ identity
    # diagonal covariance ≈ small uncertainty
    f_cov = 0.1 * torch.eye(n_pseudo, dtype=torch.float64)
    return {"kernel": kernel, "Z": Z, "f_mean": f_mean, "f_cov": f_cov}


def test_predict_drift_zero_prior_mean(trained_state):
    s = trained_state
    new_x = torch.linspace(-1.5, 1.5, 7, dtype=torch.float64).reshape(-1, 1)
    pred = predict_drift(
        kernel=s["kernel"], inducing_points=s["Z"],
        posterior_mean=s["f_mean"], posterior_cov=s["f_cov"],
        new_x=new_x,
    )
    # mean shape (n_new,)
    assert pred["mean"].shape == (7,)
    assert pred["var"].shape == (7,)
    # Prior mean is zero: at the inducing points, prediction should ≈ posterior_mean
    pred_at_Z = predict_drift(
        kernel=s["kernel"], inducing_points=s["Z"],
        posterior_mean=s["f_mean"], posterior_cov=s["f_cov"],
        new_x=s["Z"],
    )
    assert torch.allclose(pred_at_Z["mean"], s["f_mean"], atol=1e-6)


def test_predict_drift_against_explicit_R_formula(trained_state):
    s = trained_state
    new_x = torch.tensor([[0.5], [-0.7]])
    pred = predict_drift(
        kernel=s["kernel"], inducing_points=s["Z"],
        posterior_mean=s["f_mean"], posterior_cov=s["f_cov"],
        new_x=new_x,
    )
    # Reference R:
    #   kmx  = k(xm, newX)  -> (m, n_new)
    #   A    = kmx.T @ kmmInv  -> (n_new, m)
    #   mu   = 0 + A @ (posteriorMean - 0)
    #   var  = vars(newX) - diag(A @ kmx) + diag(A @ posteriorCov @ A.T)
    inter = sparse_gp_intermediates(s["kernel"], new_x, s["Z"])
    K_mm_inv = inter["K_mm_inv"]
    kmx = s["kernel"].cov(s["Z"], new_x)  # (m, n_new)
    A_new = kmx.T @ K_mm_inv
    expected_mean = A_new @ s["f_mean"]
    expected_var = (
        s["kernel"].variances(new_x)
        - torch.diagonal(A_new @ kmx)
        + torch.diagonal(A_new @ s["f_cov"] @ A_new.T)
    )
    assert torch.allclose(pred["mean"], expected_mean, atol=1e-10)
    assert torch.allclose(pred["var"], expected_var, atol=1e-10)


def test_predict_drift_quantiles_are_normal(trained_state):
    s = trained_state
    new_x = torch.tensor([[0.5]])
    pred = predict_drift(
        kernel=s["kernel"], inducing_points=s["Z"],
        posterior_mean=s["f_mean"], posterior_cov=s["f_cov"],
        new_x=new_x, quantiles=(0.025, 0.975),
    )
    # 95% normal: mean ± 1.959963985 * sqrt(var)
    z = 1.959963985  # qnorm(0.975)
    sd = float(torch.sqrt(pred["var"]).squeeze())
    mu = float(pred["mean"].squeeze())
    assert pred["quantiles"].shape == (1, 2)
    assert float(pred["quantiles"][0, 0]) == pytest.approx(mu - z * sd, rel=1e-6)
    assert float(pred["quantiles"][0, 1]) == pytest.approx(mu + z * sd, rel=1e-6)


def test_predict_diffusion_lognormal_back_transform(trained_state):
    """Diffusion uses log-normal back-transform on the underlying GP."""
    s = trained_state
    v = -0.3  # prior mean for log-diffusion
    new_x = torch.tensor([[0.5]])
    pred = predict_diffusion(
        kernel=s["kernel"], inducing_points=s["Z"],
        posterior_mean=s["f_mean"], posterior_cov=s["f_cov"],
        new_x=new_x, v=v,
    )
    # Underlying-GP mean and var (before back-transform)
    inter = sparse_gp_intermediates(s["kernel"], new_x, s["Z"])
    K_mm_inv = inter["K_mm_inv"]
    kmx = s["kernel"].cov(s["Z"], new_x)
    A_new = kmx.T @ K_mm_inv
    prior_at_new = v
    prior_at_Z = torch.full_like(s["f_mean"], v)
    gp_mean = prior_at_new + A_new @ (s["f_mean"] - prior_at_Z)
    gp_var = (
        s["kernel"].variances(new_x)
        - torch.diagonal(A_new @ kmx)
        + torch.diagonal(A_new @ s["f_cov"] @ A_new.T)
    )
    # Log-normal back-transform: mu_lognormal = exp(gp_mean + gp_var/2)
    expected_mean = torch.exp(gp_mean + gp_var / 2)
    expected_var = (torch.exp(gp_var) - 1) * torch.exp(2 * gp_mean + gp_var)
    assert torch.allclose(pred["mean"], expected_mean, atol=1e-10)
    assert torch.allclose(pred["var"], expected_var, atol=1e-10)
    # All quantiles should be positive (exp of normal quantiles)
    assert torch.all(pred["quantiles"] > 0)


def test_predict_drift_with_numpy_inputs(trained_state):
    s = trained_state
    new_x = np.linspace(-1, 1, 5).reshape(-1, 1)
    pred = predict_drift(
        kernel=s["kernel"],
        inducing_points=s["Z"].numpy(),
        posterior_mean=s["f_mean"].numpy(),
        posterior_cov=s["f_cov"].numpy(),
        new_x=new_x,
    )
    assert pred["mean"].shape == (5,)
