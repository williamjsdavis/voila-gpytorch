"""Variational lower bound for the voila SDE model.

Direct port of `get_lower_bound`, `calculate_E_vector`, and `calculate_ksi_vector`
from voila/src/sde_variational_inferencer.cpp (lines 225-267, 341-368, 376-393).

The bound is structurally:
    L = 0.5 · [
        −⟨E, ξ⟩ / h
        − n·v − Σ_i (B(s − v))_i
        − n·log h − n·log(2π)
        − tr(J⁻¹ Σ_s) − (s − v)ᵀ J⁻¹ (s − v)
        − m·log(2π) + log|J⁻¹|
        − tr(K⁻¹ Σ_f) − fᵀ K⁻¹ f
        − m·log(2π) + log|K⁻¹|
        + log|Σ_f| + log|Σ_s|
        + 2 m · log(2π e)
    ]

with E_i = exp(−v − (B(s−v))_i + 0.5 · ((B Σ_s Bᵀ)_ii + H_ii)),
     ξ_i = ΔX_i² − 2 h ΔX_i (A f)_i + h² · (a_iᵀ (Σ_f + f fᵀ) a_i + Q_ii).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "SparseSDEModel",
    "calculate_E_vector",
    "calculate_ksi_vector",
    "lower_bound",
]


def calculate_E_vector(
    v: float | Tensor,
    s_mean: Tensor,
    s_cov: Tensor,
    B: Tensor,
    H_ii: Tensor,
) -> Tensor:
    """E_i = exp(−v − (B(s−v))_i + 0.5 · ((B Σ_s Bᵀ)_ii + H_ii))."""
    s_minus_v = s_mean - v
    Bg = B @ s_minus_v
    BSBT_diag = torch.einsum("ij,jk,ik->i", B, s_cov, B)
    return torch.exp(-v - Bg + 0.5 * (BSBT_diag + H_ii))


def calculate_ksi_vector(
    dx: Tensor,
    sampling_period: float,
    f_mean: Tensor,
    f_cov: Tensor,
    A: Tensor,
    Q_ii: Tensor,
) -> Tensor:
    """ξ_i = ΔX_i² − 2 h ΔX_i (A f)_i + h² · (a_iᵀ (Σ_f + f fᵀ) a_i + Q_ii)."""
    h = sampling_period
    Af = A @ f_mean
    Fmat = f_cov + torch.outer(f_mean, f_mean)
    aFa = torch.einsum("ij,jk,ik->i", A, Fmat, A)
    return dx * dx - 2.0 * h * dx * Af + h * h * (aFa + Q_ii)


@dataclass
class SparseSDEModel:
    """Bundle of all quantities needed to evaluate the voila ELBO.

    Naming convention mirrors the C++ engine:
        - drift kernel:   K_mm, K_mm_inv, K_nm,  A,  Q_ii
        - diffusion kernel: J_mm, J_mm_inv, J_nm, B, H_ii
        - drift posterior:    f_mean, f_cov  (Gaussian over inducing values)
        - diffusion posterior: s_mean, s_cov (Gaussian over log-diff inducing values)
        - log-normal prior mean of diffusion: v (scalar)
        - data:           dx (n,) target increments, sampling_period h
    """

    dx: Tensor
    sampling_period: float
    f_mean: Tensor
    f_cov: Tensor
    A: Tensor
    Q_ii: Tensor
    K_mm_inv: Tensor
    v: float | Tensor
    s_mean: Tensor
    s_cov: Tensor
    B: Tensor
    H_ii: Tensor
    J_mm_inv: Tensor


def lower_bound(model: SparseSDEModel) -> Tensor:
    """Variational lower bound L (a scalar tensor with autograd flowing back to all inputs)."""
    n = model.dx.shape[0]
    n_pseudo = model.B.shape[1]
    h = model.sampling_period

    log2pi = math.log(2 * math.pi)
    log_ent_constant = n_pseudo * math.log(2 * math.pi * math.e)

    s_minus_v = model.s_mean - model.v

    E = calculate_E_vector(model.v, model.s_mean, model.s_cov, model.B, model.H_ii)
    ksi = calculate_ksi_vector(
        model.dx, h, model.f_mean, model.f_cov, model.A, model.Q_ii
    )

    _, logabsdet_jmm = torch.linalg.slogdet(model.J_mm_inv)
    _, logabsdet_kmm = torch.linalg.slogdet(model.K_mm_inv)
    _, logabsdet_fcov = torch.linalg.slogdet(model.f_cov)
    _, logabsdet_scov = torch.linalg.slogdet(model.s_cov)

    bound = 0.5 * (
        -(E * ksi).sum() / h
        - n * model.v
        - (model.B @ s_minus_v).sum()
        - n * math.log(h)
        - n * log2pi
        - torch.trace(model.J_mm_inv @ model.s_cov)
        - s_minus_v @ model.J_mm_inv @ s_minus_v
        - n_pseudo * log2pi
        + logabsdet_jmm
        - torch.trace(model.K_mm_inv @ model.f_cov)
        - model.f_mean @ model.K_mm_inv @ model.f_mean
        - n_pseudo * log2pi
        + logabsdet_kmm
        + logabsdet_fcov
        + logabsdet_scov
        + 2 * log_ent_constant
    )
    return bound
