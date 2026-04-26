"""TDD tests for voila_gp.sparse_gp — sparse-GP intermediates K_mm, K_nm, A, Q_ii."""

from __future__ import annotations

import pytest
import torch

from voila_gp.kernels import ExpKernel, RQKernel
from voila_gp.sparse_gp import sparse_gp_intermediates


def _grid(n: int, d: int, low: float = -2.0, high: float = 2.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(n, d, generator=g, dtype=torch.float64)


@pytest.fixture
def small_problem():
    n, m, d = 30, 6, 1
    X = _grid(n, d, seed=42)
    Z = _grid(m, d, seed=43)
    kernel = ExpKernel(amplitude=1.5, length_scales=torch.tensor([0.8]), epsilon=1e-6)
    return X, Z, kernel, n, m


def test_sparse_gp_shapes(small_problem):
    X, Z, kernel, n, m = small_problem
    out = sparse_gp_intermediates(kernel, X, Z)
    assert out["K_mm"].shape == (m, m)
    assert out["K_mm_inv"].shape == (m, m)
    assert out["K_nm"].shape == (n, m)
    assert out["A"].shape == (n, m)
    assert out["Q_ii"].shape == (n,)


def test_K_mm_includes_jitter(small_problem):
    X, Z, kernel, _n, m = small_problem
    out = sparse_gp_intermediates(kernel, X, Z)
    K_mm_no_jit = ExpKernel(
        amplitude=1.5, length_scales=torch.tensor([0.8]), epsilon=0.0
    ).cov(Z)
    diff = out["K_mm"] - K_mm_no_jit
    assert torch.allclose(diff, 1e-6 * torch.eye(m, dtype=torch.float64), atol=1e-14)


def test_K_mm_inv_is_correct_and_symmetric(small_problem):
    X, Z, kernel, _n, m = small_problem
    out = sparse_gp_intermediates(kernel, X, Z)
    I = torch.eye(m, dtype=torch.float64)
    assert torch.allclose(out["K_mm"] @ out["K_mm_inv"], I, atol=1e-9)
    # voila symmetrizes: kmmInv = (kmmInv + kmmInv.T)/2
    assert torch.allclose(out["K_mm_inv"], out["K_mm_inv"].T, atol=1e-14)


def test_A_satisfies_A_K_mm_equals_K_nm(small_problem):
    X, Z, kernel, _n, _m = small_problem
    out = sparse_gp_intermediates(kernel, X, Z)
    # A = K_nm K_mm^{-1}, so A K_mm = K_nm
    assert torch.allclose(out["A"] @ out["K_mm"], out["K_nm"], atol=1e-8)


def test_Q_ii_matches_reference_definition(small_problem):
    X, Z, kernel, _n, _m = small_problem
    out = sparse_gp_intermediates(kernel, X, Z)
    # Reference: Q_ii(i) = k(x_i, x_i) - Σ_{j,k} K_nm(i,j) K_mm_inv(j,k) K_nm(i,k)
    diag_ref = kernel.variances(X)
    aux = torch.einsum("ij,jk,ik->i", out["K_nm"], out["K_mm_inv"], out["K_nm"])
    expected = (diag_ref - aux).clamp_min(0.0)
    assert torch.allclose(out["Q_ii"], expected, atol=1e-12)


def test_Q_ii_non_negative(small_problem):
    X, Z, kernel, _n, _m = small_problem
    out = sparse_gp_intermediates(kernel, X, Z)
    assert torch.all(out["Q_ii"] >= 0)


def test_Q_ii_zero_when_inducing_equals_inputs():
    # If Z == X (so K_nm = K_nn), the sparse approximation is exact and Q_ii = 0.
    X = _grid(8, 1, seed=7)
    kernel = ExpKernel(amplitude=1.0, length_scales=torch.tensor([0.7]), epsilon=1e-8)
    out = sparse_gp_intermediates(kernel, X, X)
    assert torch.all(out["Q_ii"] < 1e-6)


def test_intermediates_with_rq_kernel():
    n, m = 20, 5
    X = _grid(n, 2, seed=11)
    Z = _grid(m, 2, seed=12)
    kernel = RQKernel(amplitude=1.0, alpha=1.5, length_scale=1.2, epsilon=1e-6)
    out = sparse_gp_intermediates(kernel, X, Z)
    assert out["K_mm"].shape == (m, m)
    assert torch.allclose(out["K_mm"] @ out["K_mm_inv"], torch.eye(m, dtype=torch.float64), atol=1e-9)


def test_gradient_flows_through_intermediates():
    n, m = 10, 4
    X = _grid(n, 1, seed=21)
    Z = _grid(m, 1, seed=22)
    kernel = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.0]), epsilon=1e-6)
    out = sparse_gp_intermediates(kernel, X, Z)
    out["A"].sum().backward()
    # amplitude is a fixed float per voila's convention; only length_scales is trainable
    assert kernel.length_scales.grad is not None
    assert torch.all(torch.isfinite(kernel.length_scales.grad))
