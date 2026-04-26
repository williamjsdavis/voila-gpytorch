"""Sparse-GP machinery — direct port of `calculate_kernel_matrices` in
voila/src/sde_variational_inferencer.cpp.

Given a kernel k, n input points X (the time-series base points after dropping
the last sample) and m inducing points Z, computes:

    K_mm     = k(Z, Z) + ε I               (m, m)   auto-cov at inducing points
    K_mm_inv = symmetrized inv(K_mm)        (m, m)
    K_nm     = k(X, Z)                      (n, m)   cross-cov
    A        = K_nm K_mm_inv                (n, m)   sparse projection
    Q_ii     = k(x_i, x_i) − Σ_{j,k} K_nm(i,j) K_mm_inv(j,k) K_nm(i,k)
                                            (n,)    residual variance, clamped ≥ 0

These are the building blocks for both the variational lower bound and the
closed-form posterior updates in `voila_gp.posterior`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels import Kernel


def _symmetric_inverse(A: Tensor) -> Tensor:
    """Inverse of a symmetric PD matrix, then explicitly symmetrized.

    Mirrors voila's `kmmInv = inv_sympd(kmm); kmmInv = (kmmInv + kmmInv.T)/2`.
    Uses Cholesky for numerical stability.
    """
    L = torch.linalg.cholesky(A)
    Ainv = torch.cholesky_inverse(L)
    return 0.5 * (Ainv + Ainv.T)


def sparse_gp_intermediates(
    kernel: Kernel,
    X: Tensor,
    inducing_points: Tensor,
) -> dict[str, Tensor]:
    """Compute K_mm, K_mm_inv, K_nm, A, Q_ii.

    Parameters
    ----------
    kernel : Kernel
    X : (n, d) tensor — base points (time-series with last sample dropped).
    inducing_points : (m, d) tensor — pseudo-inputs.

    Returns
    -------
    dict with keys 'K_mm', 'K_mm_inv', 'K_nm', 'A', 'Q_ii'.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n,d), got shape {tuple(X.shape)}")
    if inducing_points.ndim != 2:
        raise ValueError(
            f"inducing_points must be 2-D (m,d), got shape {tuple(inducing_points.shape)}"
        )
    if X.shape[1] != inducing_points.shape[1]:
        raise ValueError(
            f"X and inducing_points must share input dimension; got {X.shape[1]} and "
            f"{inducing_points.shape[1]}"
        )

    K_mm = kernel.cov(inducing_points)  # epsilon jitter on diagonal
    K_mm_inv = _symmetric_inverse(K_mm)
    K_nm = kernel.cov(X, inducing_points)
    A = K_nm @ K_mm_inv

    diag_kxx = kernel.variances(X)
    aux = torch.einsum("ij,jk,ik->i", K_nm, K_mm_inv, K_nm)
    Q_ii = (diag_kxx - aux).clamp_min(0.0)

    return {"K_mm": K_mm, "K_mm_inv": K_mm_inv, "K_nm": K_nm, "A": A, "Q_ii": Q_ii}
