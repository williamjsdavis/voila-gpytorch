"""Unit tests for the cross-device compatibility shims."""

from __future__ import annotations

import pytest
import torch
from scipy.stats import norm

from voila_gp._compat import (
    _default_dtype_for,
    cholesky_inverse_compat,
    ndtri_compat,
)


def test_default_dtype_for_cpu_is_float64() -> None:
    assert _default_dtype_for("cpu") is torch.float64
    assert _default_dtype_for(torch.device("cpu")) is torch.float64


def test_default_dtype_for_cuda_is_float64() -> None:
    # FP64 has good throughput on every CUDA SKU we target (A10/H100/A100).
    assert _default_dtype_for("cuda") is torch.float64


def test_default_dtype_for_mps_is_float32() -> None:
    # Metal/MPS lacks FP64 linalg in PyTorch 2.11 — falling back to FP32 is
    # the only practical option and is enforced here so the rest of the
    # pipeline can rely on it.
    assert _default_dtype_for("mps") is torch.float32


def test_cholesky_inverse_compat_matches_torch_on_cpu() -> None:
    """Shim must agree with `torch.cholesky_inverse` on CPU to ~machine eps.

    1e-12 is appropriate for float64 PD inversion of a 6×6 matrix; we keep
    it tight because any drift here is a real bug, not algorithm choice.
    """
    torch.manual_seed(0)
    M = torch.randn(6, 6, dtype=torch.float64)
    A = M @ M.T + 0.5 * torch.eye(6, dtype=torch.float64)  # SPD
    L = torch.linalg.cholesky(A)

    direct = torch.cholesky_inverse(L)
    via_shim = cholesky_inverse_compat(L)

    assert torch.allclose(direct, via_shim, atol=1e-12, rtol=0)


def test_cholesky_inverse_compat_actually_inverts() -> None:
    """A·A⁻¹ ≈ I — sanity check independent of torch.cholesky_inverse."""
    torch.manual_seed(1)
    M = torch.randn(8, 8, dtype=torch.float64)
    A = M @ M.T + torch.eye(8, dtype=torch.float64)
    L = torch.linalg.cholesky(A)
    Ainv = cholesky_inverse_compat(L)
    eye = torch.eye(8, dtype=torch.float64)
    assert torch.allclose(A @ Ainv, eye, atol=1e-10, rtol=0)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
def test_ndtri_compat_matches_scipy(p: float) -> None:
    """Φ⁻¹ via erfinv should agree with scipy.stats.norm.ppf to 1e-12."""
    p_t = torch.tensor(p, dtype=torch.float64)
    ours = float(ndtri_compat(p_t))
    expected = float(norm.ppf(p))
    assert abs(ours - expected) < 1e-12


def test_ndtri_compat_preserves_dtype() -> None:
    p32 = torch.tensor(0.7, dtype=torch.float32)
    out = ndtri_compat(p32)
    assert out.dtype is torch.float32

    p64 = torch.tensor(0.7, dtype=torch.float64)
    out = ndtri_compat(p64)
    assert out.dtype is torch.float64
