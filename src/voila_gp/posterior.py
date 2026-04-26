"""Closed-form drift posterior + Laplace diffusion posterior updates.

Direct port of `update_distributions()` in voila/src/sde_variational_inferencer.cpp
(lines 275-310). Drift is closed-form; diffusion uses a Laplace approximation
around the mode of a non-quadratic objective (we find the mode via Newton's
method on the analytical gradient/Hessian).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .elbo import SparseSDEModel, calculate_E_vector, calculate_ksi_vector

__all__ = [
    "laplace_diffusion_objective",
    "update_diffusion_laplace",
    "update_drift_closed_form",
]


def _symmetric_inverse(A: Tensor) -> Tensor:
    L = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(L)
    return 0.5 * (Ainv + Ainv.T)


def update_drift_closed_form(model: SparseSDEModel) -> dict[str, Tensor]:
    """Closed-form Gaussian update for the drift inducing-value posterior.

    Σ_f^{-1} = K_mm^{-1} + h · Aᵀ diag(E) A
    μ_f      = Σ_f · Aᵀ (E ⊙ ΔX)
    """
    E = calculate_E_vector(model.v, model.s_mean, model.s_cov, model.B, model.H_ii)
    cov_inv = model.K_mm_inv + model.sampling_period * model.A.T @ torch.diag(E) @ model.A
    cov_inv = 0.5 * (cov_inv + cov_inv.T)
    cov = _symmetric_inverse(cov_inv)
    mean = cov @ (model.A.T @ (E * model.dx))
    return {"f_mean": mean, "f_cov": cov}


# ---------------------------------------------------------------------------
# Laplace diffusion update
# ---------------------------------------------------------------------------


def laplace_diffusion_objective(
    s: Tensor,
    cvec: Tensor,
    B: Tensor,
    J_mm_inv: Tensor,
    v: float,
    sampling_period: float,
) -> Tensor:
    """The minimizable Laplace objective from voila/src/sde_variational_inferencer.cpp:439-448.

    f(s) = -0.5 · (
        -1/h · ⟨c, exp(-B(s − v))⟩
        - 1ᵀ B(s − v)
        - (s − v)ᵀ J⁻¹ (s − v)
    )
    """
    smv = s - v
    Bsmv = B @ smv
    inner = (
        -1.0 / sampling_period * (cvec * torch.exp(-Bsmv)).sum()
        - Bsmv.sum()
        - smv @ J_mm_inv @ smv
    )
    return -0.5 * inner


def _laplace_grad_hess(
    s: Tensor,
    cvec: Tensor,
    B: Tensor,
    J_mm_inv: Tensor,
    v: float,
    sampling_period: float,
) -> tuple[Tensor, Tensor]:
    """Analytical gradient and Hessian of `laplace_diffusion_objective` w.r.t. s.

    f(s) = -0.5 · [ -1/h · cᵀ exp(-B(s-v)) - 1ᵀ B(s-v) - (s-v)ᵀ J⁻¹ (s-v) ]

    ∂f/∂s = -0.5 · [  1/h · Bᵀ (c ⊙ exp(-B(s-v))) - Bᵀ 1 - 2 J⁻¹ (s-v) ]
    ∂²f/∂s² = -0.5 · [ -1/h · Bᵀ diag(c ⊙ exp(-B(s-v))) B - 2 J⁻¹ ]
            = (1/(2h)) · Bᵀ diag(c · aux) B + J⁻¹                  [matches voila's auxMat]
    """
    smv = s - v
    aux = torch.exp(-(B @ smv))  # (n,)
    weight = cvec * aux  # (n,)
    grad = -0.5 * (
        (1.0 / sampling_period) * (B.T @ weight)
        - B.sum(dim=0)
        - 2.0 * (J_mm_inv @ smv)
    )
    hess = (1.0 / (2.0 * sampling_period)) * (B.T @ torch.diag(weight) @ B) + J_mm_inv
    hess = 0.5 * (hess + hess.T)
    return grad, hess


def update_diffusion_laplace(
    model: SparseSDEModel,
    *,
    max_newton_iters: int = 50,
    grad_tol: float = 1e-9,
    step_tol: float = 1e-12,
) -> dict[str, Tensor]:
    """Laplace approximation to the diffusion-inducing posterior.

    1. Build c_i = exp(log ξ_i - v + H_ii/2).
    2. Find the mode s* of the Laplace objective via damped Newton iterations
       (Hessian is PSD so undamped Newton converges; we add a backtracking
       safeguard for robustness).
    3. The Laplace covariance is the inverse of the Hessian at s*:
       Σ_s = ((1/(2h)) Bᵀ diag(c · exp(-B(s* - v))) B + J⁻¹)⁻¹
    """
    ksi = calculate_ksi_vector(
        model.dx, model.sampling_period, model.f_mean, model.f_cov, model.A, model.Q_ii
    )
    # ksi must be positive (it's a sum of squares). Clamp tiny negatives from float roundoff.
    ksi = ksi.clamp_min(torch.finfo(ksi.dtype).tiny)
    cvec = torch.exp(torch.log(ksi) - model.v + model.H_ii / 2.0)

    s = model.s_mean.detach().clone()
    v_scalar = float(model.v) if isinstance(model.v, torch.Tensor) else model.v

    f_prev = float(
        laplace_diffusion_objective(s, cvec, model.B, model.J_mm_inv, v_scalar, model.sampling_period)
    )
    for _ in range(max_newton_iters):
        grad, hess = _laplace_grad_hess(
            s, cvec, model.B, model.J_mm_inv, v_scalar, model.sampling_period
        )
        if float(grad.abs().max()) < grad_tol:
            break
        delta = torch.linalg.solve(hess, grad)
        # Backtracking: ensure objective decreases (we minimize)
        step = 1.0
        s_trial = s
        f_trial = f_prev
        for _ls in range(25):
            s_trial = s - step * delta
            f_trial = float(
                laplace_diffusion_objective(
                    s_trial, cvec, model.B, model.J_mm_inv, v_scalar, model.sampling_period
                )
            )
            if f_trial < f_prev or step < step_tol:
                break
            step *= 0.5
        s = s_trial
        if abs(f_prev - f_trial) < step_tol * max(1.0, abs(f_prev)):
            f_prev = f_trial
            break
        f_prev = f_trial

    # Laplace covariance: inverse Hessian at the mode.
    smv_star = s - model.v
    aux = torch.exp(-(model.B @ smv_star))
    weight = cvec * aux
    auxMat = (
        1.0 / (2.0 * model.sampling_period) * (model.B.T @ torch.diag(weight) @ model.B)
        + model.J_mm_inv
    )
    auxMat = 0.5 * (auxMat + auxMat.T)
    s_cov = _symmetric_inverse(auxMat)
    s_cov = 0.5 * (s_cov + s_cov.T)
    return {"s_mean": s, "s_cov": s_cov}
