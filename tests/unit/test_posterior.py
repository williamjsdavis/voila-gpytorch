"""TDD tests for voila_gp.posterior — closed-form drift + Laplace diffusion updates.

Reference: voila/src/sde_variational_inferencer.cpp::update_distributions (lines 275-310).
"""

from __future__ import annotations

import torch

from voila_gp.elbo import (
    SparseSDEModel,
    calculate_E_vector,
    calculate_ksi_vector,
    lower_bound,
)
from voila_gp.kernels import ExpKernel
from voila_gp.posterior import (
    laplace_diffusion_objective,
    update_diffusion_laplace,
    update_drift_closed_form,
)
from voila_gp.sparse_gp import sparse_gp_intermediates


def _toy_model(n=80, m=6, dt=0.01, v=-0.3, seed=11):
    g = torch.Generator().manual_seed(seed)
    X = torch.linspace(-2, 2, n, dtype=torch.float64).reshape(-1, 1)
    Z = torch.linspace(-2, 2, m, dtype=torch.float64).reshape(-1, 1)
    # synthetic increments dx ≈ -X·dt + noise (drift = -x)
    dx = (-X.squeeze() * dt + 0.05 * torch.randn(n, generator=g, dtype=torch.float64)).to(
        torch.float64
    )

    drift_kernel = ExpKernel(
        amplitude=1.0, length_scales=torch.tensor([0.8]), epsilon=1e-6
    )
    diff_kernel = ExpKernel(
        amplitude=0.5, length_scales=torch.tensor([1.0]), epsilon=1e-6
    )
    drift = sparse_gp_intermediates(drift_kernel, X, Z)
    diff = sparse_gp_intermediates(diff_kernel, X, Z)

    return SparseSDEModel(
        dx=dx,
        sampling_period=dt,
        f_mean=torch.zeros(m, dtype=torch.float64),
        f_cov=drift["K_mm"].detach().clone(),
        A=drift["A"].detach(),
        Q_ii=drift["Q_ii"].detach(),
        K_mm_inv=drift["K_mm_inv"].detach(),
        v=v,
        s_mean=torch.full((m,), v, dtype=torch.float64),
        s_cov=diff["K_mm"].detach().clone(),
        B=diff["A"].detach(),
        H_ii=diff["Q_ii"].detach(),
        J_mm_inv=diff["K_mm_inv"].detach(),
    )


# ---------------------------------------------------------------------------
# Drift: closed-form update
# ---------------------------------------------------------------------------


def test_update_drift_matches_reference_formula():
    m = _toy_model()
    new = update_drift_closed_form(m)
    # Reference:
    #   Σ_f^{-1} = K_mm_inv + h · A^T diag(E) A
    #   μ_f      = Σ_f · A^T (E ⊙ dx)
    E = calculate_E_vector(m.v, m.s_mean, m.s_cov, m.B, m.H_ii)
    cov_inv = m.K_mm_inv + m.sampling_period * m.A.T @ torch.diag(E) @ m.A
    cov_inv = 0.5 * (cov_inv + cov_inv.T)
    cov_expected = torch.linalg.inv(cov_inv)
    cov_expected = 0.5 * (cov_expected + cov_expected.T)
    mean_expected = cov_expected @ m.A.T @ (E * m.dx)
    assert torch.allclose(new["f_cov"], cov_expected, atol=1e-10)
    assert torch.allclose(new["f_mean"], mean_expected, atol=1e-10)


def test_update_drift_increases_lower_bound():
    m = _toy_model()
    L_before = float(lower_bound(m))
    new = update_drift_closed_form(m)
    m2 = SparseSDEModel(
        **{**m.__dict__, "f_mean": new["f_mean"], "f_cov": new["f_cov"]}
    )
    L_after = float(lower_bound(m2))
    assert L_after >= L_before - 1e-9


def test_update_drift_returns_symmetric_covariance():
    m = _toy_model()
    new = update_drift_closed_form(m)
    assert torch.allclose(new["f_cov"], new["f_cov"].T, atol=1e-12)


# ---------------------------------------------------------------------------
# Diffusion: Laplace update — objective and one-step
# ---------------------------------------------------------------------------


def test_laplace_objective_matches_reference_formula():
    m = _toy_model()
    # cvector at the initial state where s_mean == v: c_i = exp(log(ksi_i) - v + H_ii/2)
    ksi = calculate_ksi_vector(m.dx, m.sampling_period, m.f_mean, m.f_cov, m.A, m.Q_ii)
    cvec = torch.exp(torch.log(ksi) - m.v + m.H_ii / 2.0)
    s_test = torch.tensor([m.v + 0.1, m.v, m.v - 0.05, m.v + 0.02, m.v, m.v - 0.1], dtype=torch.float64)
    obj = laplace_diffusion_objective(s_test, cvec, m.B, m.J_mm_inv, m.v, m.sampling_period)
    # C++: -0.5 * (-1/h · dot(c, exp(-B(s-v))) - sum(B(s-v)) - (s-v)^T J_inv (s-v))
    smv = s_test - m.v
    bsmv = m.B @ smv
    expected = -0.5 * (
        -1.0 / m.sampling_period * float((cvec * torch.exp(-bsmv)).sum())
        - float(bsmv.sum())
        - float(smv @ m.J_mm_inv @ smv)
    )
    assert abs(float(obj) - expected) < 1e-12


def test_update_diffusion_increases_lower_bound():
    m = _toy_model()
    # Before updating diffusion we typically apply the drift update so ξ is informative
    drift_new = update_drift_closed_form(m)
    m_after_drift = SparseSDEModel(
        **{**m.__dict__, "f_mean": drift_new["f_mean"], "f_cov": drift_new["f_cov"]}
    )
    L_before = float(lower_bound(m_after_drift))

    s_new = update_diffusion_laplace(m_after_drift)
    m_after_diff = SparseSDEModel(
        **{
            **m_after_drift.__dict__,
            "s_mean": s_new["s_mean"],
            "s_cov": s_new["s_cov"],
        }
    )
    L_after = float(lower_bound(m_after_diff))
    assert L_after >= L_before - 1e-6


def test_update_diffusion_covariance_matches_analytical_hessian():
    m = _toy_model()
    drift_new = update_drift_closed_form(m)
    m_after_drift = SparseSDEModel(
        **{**m.__dict__, "f_mean": drift_new["f_mean"], "f_cov": drift_new["f_cov"]}
    )
    out = update_diffusion_laplace(m_after_drift)

    # Reference:
    #   c_i  = exp(log(ksi_i) - v + H_ii/2)
    #   aux_i = exp(-B(s* - v))_i
    #   Σ_s = inv( (1/(2h)) · Bᵀ · diag(c · aux) · B + J_mm_inv )
    ksi = calculate_ksi_vector(
        m_after_drift.dx, m_after_drift.sampling_period,
        m_after_drift.f_mean, m_after_drift.f_cov,
        m_after_drift.A, m_after_drift.Q_ii,
    )
    cvec = torch.exp(torch.log(ksi) - m.v + m.H_ii / 2.0)
    s_star = out["s_mean"]
    aux = torch.exp(-(m.B @ (s_star - m.v)))
    auxMat = 1.0 / (2.0 * m.sampling_period) * m.B.T @ torch.diag(cvec * aux) @ m.B + m.J_mm_inv
    auxMat = 0.5 * (auxMat + auxMat.T)
    cov_expected = torch.linalg.inv(auxMat)
    cov_expected = 0.5 * (cov_expected + cov_expected.T)
    assert torch.allclose(out["s_cov"], cov_expected, atol=1e-8)


def test_repeated_alternation_converges_lower_bound():
    """Iterating update_drift / update_diffusion alternately should converge L.

    Note: the diffusion update is a Laplace approximation (mode + Hessian) to the
    optimal Gaussian q(s) — this is NOT the KL-minimizing Gaussian, so individual
    diffusion updates are not guaranteed to monotonically increase L. The
    reference voila implementation has the same property. What we *do* require:
    (a) the bound makes large progress in the first few iterations,
    (b) it converges to a stable value (drift among consecutive iterates → 0),
    (c) the final value is well above the initial.
    """
    m = _toy_model(n=200, m=8)
    L_history = [float(lower_bound(m))]
    for _ in range(15):
        d = update_drift_closed_form(m)
        m = SparseSDEModel(**{**m.__dict__, "f_mean": d["f_mean"], "f_cov": d["f_cov"]})
        sd = update_diffusion_laplace(m)
        m = SparseSDEModel(**{**m.__dict__, "s_mean": sd["s_mean"], "s_cov": sd["s_cov"]})
        L_history.append(float(lower_bound(m)))
    # Final value well above initial
    assert L_history[-1] - L_history[0] > 1.0
    # Convergence: last-three step changes are tiny
    diffs = [L_history[i + 1] - L_history[i] for i in range(len(L_history) - 1)]
    assert max(abs(d) for d in diffs[-3:]) < 1e-6
