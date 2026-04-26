"""Compatibility helpers for cross-device support.

`torch.cholesky_inverse` and `torch.special.ndtri` are not implemented on every
backend in PyTorch 2.11 (Apple's Metal/MPS lacks both at time of writing). The
shims here fall back to mathematically equivalent constructions that *are*
universally available.

`_default_dtype_for(device)` codifies the package-wide dtype policy: float64
on CPU and CUDA (matches the published validation), float32 on MPS (Metal has
no FP64 linalg).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _default_dtype_for(device: str | torch.device) -> torch.dtype:
    d = torch.device(device) if not isinstance(device, torch.device) else device
    return torch.float32 if d.type == "mps" else torch.float64


def cholesky_inverse_compat(L: Tensor) -> Tensor:
    """Inverse of L Lᵀ from its lower-triangular Cholesky factor.

    On MPS (PyTorch 2.11) both `torch.cholesky_inverse` and
    `torch.cholesky_solve` are unimplemented (the latter raises
    `NotImplementedError: aten::_cholesky_solve_helper`), so we hand-roll the
    inverse from two `torch.linalg.solve_triangular` calls — which *are*
    available on MPS:

        Solve L · Y = I       for Y      (forward substitution)
        Solve Lᵀ · X = Y      for X      (backward substitution)

    so that X = (L Lᵀ)⁻¹. Mathematically identical to `cholesky_inverse(L)`,
    just expressed via primitives every backend supports. On CPU and CUDA we
    keep the direct LAPACK call, which is faster.
    """
    if L.device.type == "mps":
        eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
        Y = torch.linalg.solve_triangular(L, eye, upper=False)
        return torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
    return torch.cholesky_inverse(L)


def ndtri_compat(p: Tensor) -> Tensor:
    """Inverse standard-normal CDF: Φ⁻¹(p) = √2 · erfinv(2p − 1).

    Equivalent to `torch.special.ndtri` but available on every backend
    (`erfinv` ships on CPU/CUDA/MPS).
    """
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=p.dtype, device=p.device))
    return sqrt2 * torch.erfinv(2 * p - 1)
