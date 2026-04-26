"""TDD tests for voila_gp.elbo — port of get_lower_bound + sub-quantities.

Reference: voila/src/sde_variational_inferencer.cpp lines 225-267 (get_lower_bound),
341-368 (calculate_E_vector), 376-393 (calculate_ksi_vector).
"""

from __future__ import annotations

import math

import pytest
import torch

from voila_gp.elbo import (
    SparseSDEModel,
    calculate_E_vector,
    calculate_ksi_vector,
    lower_bound,
)
from voila_gp.kernels import ExpKernel
from voila_gp.sparse_gp import sparse_gp_intermediates


def _toy_setup(n=20, m=4, d=1, seed=7):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=torch.float64) * 2 - 1
    Z = torch.linspace(-1.0, 1.0, m, dtype=torch.float64).reshape(-1, d)
    dx = 0.01 * torch.randn(n, generator=g, dtype=torch.float64)
    drift_kernel = ExpKernel(amplitude=1.0, length_scales=torch.tensor([0.5]), epsilon=1e-6)
    diff_kernel = ExpKernel(amplitude=0.5, length_scales=torch.tensor([0.7]), epsilon=1e-6)
    drift = sparse_gp_intermediates(drift_kernel, X, Z)
    diff = sparse_gp_intermediates(diff_kernel, X, Z)
    return {
        "X": X, "Z": Z, "dx": dx, "h": 0.001,
        "drift_kernel": drift_kernel, "diff_kernel": diff_kernel,
        "K_mm": drift["K_mm"], "K_mm_inv": drift["K_mm_inv"], "A": drift["A"], "Q_ii": drift["Q_ii"],
        "J_mm": diff["K_mm"], "J_mm_inv": diff["K_mm_inv"], "B": diff["A"], "H_ii": diff["Q_ii"],
        # initial posterior state matching voila's init: f_mean=0, f_cov=K_mm; s_mean=v, s_cov=J_mm
        "f_mean": torch.zeros(m, dtype=torch.float64),
        "f_cov": drift["K_mm"].detach().clone(),
        "v": -0.3,
        "s_cov": diff["K_mm"].detach().clone(),
    }


# ---------------------------------------------------------------------------
# E vector: E_i = exp(-v - (B(s-v))_i + 0.5 * (B Σ_s B^T + H)_ii)
# ---------------------------------------------------------------------------


def test_E_vector_matches_explicit_numpy_formula():
    s = _toy_setup()
    s_mean = torch.full((s["B"].shape[1],), s["v"], dtype=torch.float64)  # init: s_mean = v
    E = calculate_E_vector(s["v"], s_mean, s["s_cov"], s["B"], s["H_ii"])
    # When s_mean == v, B(s-v) == 0, and E_i = exp(-v + 0.5*(B Σ_s B^T + H)_ii)
    BSBT_diag = torch.einsum("ij,jk,ik->i", s["B"], s["s_cov"], s["B"])
    expected = torch.exp(-s["v"] + 0.5 * (BSBT_diag + s["H_ii"]))
    assert torch.allclose(E, expected, atol=1e-12)


def test_E_vector_is_positive():
    s = _toy_setup()
    s_mean = torch.full((s["B"].shape[1],), s["v"], dtype=torch.float64)
    E = calculate_E_vector(s["v"], s_mean, s["s_cov"], s["B"], s["H_ii"])
    assert torch.all(E > 0)


def test_E_vector_grows_with_diffusion_uncertainty():
    s = _toy_setup()
    s_mean = torch.full((s["B"].shape[1],), s["v"], dtype=torch.float64)
    E_low = calculate_E_vector(s["v"], s_mean, 0.1 * s["s_cov"], s["B"], s["H_ii"])
    E_high = calculate_E_vector(s["v"], s_mean, s["s_cov"], s["B"], s["H_ii"])
    assert torch.all(E_high >= E_low - 1e-12)


# ---------------------------------------------------------------------------
# xi vector: ξ_i = ΔX² - 2h ΔX (Af)_i + h² ((a_i^T Σ_f a_i + (a_i^T f)² ) + Q_ii)
# Expanded: a_i^T (Σ_f + f f^T) a_i + Q_ii
# ---------------------------------------------------------------------------


def test_ksi_vector_matches_reference_definition():
    s = _toy_setup()
    f_mean = s["f_mean"]
    f_cov = s["f_cov"]
    ksi = calculate_ksi_vector(s["dx"], s["h"], f_mean, f_cov, s["A"], s["Q_ii"])
    # Direct port of C++ inner loop
    Af = s["A"] @ f_mean
    Fmat = f_cov + torch.outer(f_mean, f_mean)
    h2 = s["h"] ** 2
    expected = torch.zeros_like(ksi)
    for i in range(s["dx"].numel()):
        ai = s["A"][i]
        expected[i] = (
            s["dx"][i] ** 2
            - 2 * s["h"] * s["dx"][i] * Af[i]
            + h2 * (ai @ Fmat @ ai + s["Q_ii"][i])
        )
    assert torch.allclose(ksi, expected, atol=1e-14)


def test_ksi_vector_zero_when_dx_equals_drift_step_and_no_uncertainty():
    # If f_mean exactly explains dx at each x (i.e., dx = h * Af), f_cov=0, Q_ii=0,
    # then ξ should reduce to 0 (drift increment matches; no residual variance).
    n, m = 4, 2
    A = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    f_mean = torch.tensor([0.3, -0.7], dtype=torch.float64)
    h = 0.1
    dx = h * (A @ f_mean)
    f_cov = torch.zeros(m, m, dtype=torch.float64)
    Q_ii = torch.zeros(n, dtype=torch.float64)
    ksi = calculate_ksi_vector(dx, h, f_mean, f_cov, A, Q_ii)
    assert torch.allclose(ksi, torch.zeros(n, dtype=torch.float64), atol=1e-14)


# ---------------------------------------------------------------------------
# Full lower bound: assemble all terms, gradient flow, finite-difference check.
# ---------------------------------------------------------------------------


def _build_model(s) -> SparseSDEModel:
    return SparseSDEModel(
        dx=s["dx"],
        sampling_period=s["h"],
        f_mean=s["f_mean"],
        f_cov=s["f_cov"],
        A=s["A"],
        Q_ii=s["Q_ii"],
        K_mm_inv=s["K_mm_inv"],
        v=s["v"],
        s_mean=torch.full((s["B"].shape[1],), s["v"], dtype=torch.float64),
        s_cov=s["s_cov"],
        B=s["B"],
        H_ii=s["H_ii"],
        J_mm_inv=s["J_mm_inv"],
    )


def test_lower_bound_returns_finite_scalar():
    L = lower_bound(_build_model(_toy_setup()))
    assert L.dim() == 0
    assert math.isfinite(float(L.detach()))


def test_lower_bound_matches_term_by_term_assembly():
    s = _toy_setup()
    m = _build_model(s)
    L_module = float(lower_bound(m).detach())

    # Direct re-assembly from the C++ formula (lines 257-267)
    n = s["dx"].numel()
    n_pseudo = s["B"].shape[1]
    log2pi = math.log(2 * math.pi)
    log_ent_constant = n_pseudo * math.log(2 * math.pi * math.e)

    with torch.no_grad():
        s_mean = torch.full((n_pseudo,), s["v"], dtype=torch.float64)
        E = calculate_E_vector(s["v"], s_mean, s["s_cov"], s["B"], s["H_ii"])
        ksi = calculate_ksi_vector(s["dx"], s["h"], s["f_mean"], s["f_cov"], s["A"], s["Q_ii"])
        s_minus_v = s_mean - s["v"]

        _, logabsdet_jmm = torch.linalg.slogdet(s["J_mm_inv"])
        _, logabsdet_kmm = torch.linalg.slogdet(s["K_mm_inv"])
        _, logabsdet_fcov = torch.linalg.slogdet(s["f_cov"])
        _, logabsdet_scov = torch.linalg.slogdet(s["s_cov"])

        expected = 0.5 * (
            -float((E * ksi).sum()) / s["h"]
            - n * s["v"]
            - float((s["B"] @ s_minus_v).sum())
            - n * math.log(s["h"])
            - n * log2pi
            - float(torch.trace(s["J_mm_inv"] @ s["s_cov"]))
            - float(s_minus_v @ s["J_mm_inv"] @ s_minus_v)
            - n_pseudo * log2pi
            + float(logabsdet_jmm)
            - float(torch.trace(s["K_mm_inv"] @ s["f_cov"]))
            - float(s["f_mean"] @ s["K_mm_inv"] @ s["f_mean"])
            - n_pseudo * log2pi
            + float(logabsdet_kmm)
            + float(logabsdet_fcov)
            + float(logabsdet_scov)
            + 2 * log_ent_constant
        )
    assert L_module == pytest.approx(expected, rel=1e-12)


def test_lower_bound_gradients_flow_to_kernel_hyperparameters():
    """The bound must backprop through K_mm_inv, A, Q_ii, B, H_ii, J_mm_inv."""
    s = _toy_setup()
    drift = sparse_gp_intermediates(s["drift_kernel"], s["X"], s["Z"])
    diff = sparse_gp_intermediates(s["diff_kernel"], s["X"], s["Z"])
    n_pseudo = drift["K_mm"].shape[0]
    model = SparseSDEModel(
        dx=s["dx"],
        sampling_period=s["h"],
        f_mean=torch.zeros(n_pseudo, dtype=torch.float64),
        f_cov=drift["K_mm"],
        A=drift["A"],
        Q_ii=drift["Q_ii"],
        K_mm_inv=drift["K_mm_inv"],
        v=s["v"],
        s_mean=torch.full((n_pseudo,), s["v"], dtype=torch.float64),
        s_cov=diff["K_mm"],
        B=diff["A"],
        H_ii=diff["Q_ii"],
        J_mm_inv=diff["K_mm_inv"],
    )
    L = lower_bound(model)
    L.backward()
    for kernel in (s["drift_kernel"], s["diff_kernel"]):
        for name, p in kernel.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.all(torch.isfinite(p.grad)), f"non-finite grad for {name}"


def test_lower_bound_increases_after_one_drift_update():
    """Sanity: after updating only the drift posterior in closed form, L should not decrease.
    This is a *necessary* property of any correct VI step and a strong correctness signal
    even without an R reference at converged hyperparams."""
    s = _toy_setup(n=50, m=5)
    m_init = _build_model(s)
    L_init = float(lower_bound(m_init).detach())

    # Closed-form drift update at the initial state:
    #   Σ_f^{-1} = K_mm_inv + h · A^T diag(E) A
    #   μ_f      = Σ_f · A^T (E ⊙ dx)
    s_mean = torch.full((s["B"].shape[1],), s["v"], dtype=torch.float64)
    E = calculate_E_vector(s["v"], s_mean, s["s_cov"], s["B"], s["H_ii"])
    AtdiagEA = s["A"].T @ torch.diag(E) @ s["A"]
    f_cov_inv = s["K_mm_inv"] + s["h"] * AtdiagEA
    f_cov_inv = 0.5 * (f_cov_inv + f_cov_inv.T)
    f_cov_new = torch.linalg.inv(f_cov_inv)
    f_cov_new = 0.5 * (f_cov_new + f_cov_new.T)
    f_mean_new = f_cov_new @ s["A"].T @ (E * s["dx"])

    m_after = SparseSDEModel(
        dx=s["dx"], sampling_period=s["h"],
        f_mean=f_mean_new, f_cov=f_cov_new,
        A=s["A"], Q_ii=s["Q_ii"], K_mm_inv=s["K_mm_inv"],
        v=s["v"], s_mean=s_mean, s_cov=s["s_cov"],
        B=s["B"], H_ii=s["H_ii"], J_mm_inv=s["J_mm_inv"],
    )
    L_after = float(lower_bound(m_after).detach())
    assert L_after >= L_init - 1e-9
