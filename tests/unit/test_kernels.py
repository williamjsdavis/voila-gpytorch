"""TDD tests for voila_gp.kernels — match reference C++ forms exactly.

Reference forms (from voila/src/common_kernels.cpp):
- exp_kernel:         k = amp * exp(-Σ_d ((x_d-y_d)/L_d)² / 2)
- rq_kernel:          k = amp * (1 + ||x-y||² / (2 α L²))^(-α)   (single shared L)
- exp_const_kernel:   k = expAmp * exp(...) + (maxAmp - expAmp)
- sum_exp_kernels:    k = A1 * exp(-... / L1²) + (maxAmp - A1) * exp(-... / L2²)
- clamped_exp_lin:    let p = linAmp · Σ_d (x_d - c)(y_d - c);
                      if p > maxAmp: k = maxAmp
                      else:          k = (maxAmp - p) * exp(-Σ_d ((x_d-y_d)/L_d)² / 2) + p

All kernels expose:
- cov(X, Y) -> (n, m) cross covariance
- cov(X)    -> (n, n) auto covariance (with epsilon jitter on diagonal)
- variances(X) -> (n,) diagonal (with epsilon jitter)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from voila_gp.kernels import (
    ClampedExpLinKernel,
    ExpConstKernel,
    ExpKernel,
    RQKernel,
    SumExpKernel,
)


def _grid(n: int, d: int, low: float = -2.0, high: float = 2.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(n, d, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# ExpKernel  (RBF with ARD; reference: exp_kernel)
# ---------------------------------------------------------------------------


def test_exp_kernel_value_matches_formula():
    amp = 0.7
    ls = torch.tensor([0.4, 1.3])
    k = ExpKernel(amplitude=amp, length_scales=ls)
    x = torch.tensor([[0.5, -1.0]])
    y = torch.tensor([[0.1, 0.5]])
    expected = amp * math.exp(-0.5 * (((0.5 - 0.1) / 0.4) ** 2 + ((-1.0 - 0.5) / 1.3) ** 2))
    got = float(k.cov(x, y).detach().squeeze())
    assert got == pytest.approx(expected, rel=1e-12)


def test_exp_kernel_shapes_and_symmetry():
    k = ExpKernel(amplitude=2.0, length_scales=torch.tensor([1.0, 2.0]), epsilon=1e-6)
    X = _grid(7, 2)
    Y = _grid(5, 2, seed=1)
    assert k.cov(X, Y).shape == (7, 5)
    K = k.cov(X)
    assert K.shape == (7, 7)
    assert torch.allclose(K, K.T, atol=1e-14)
    # epsilon adds to the diagonal of auto-covariance
    K_nojit = ExpKernel(amplitude=2.0, length_scales=torch.tensor([1.0, 2.0])).cov(X)
    assert torch.allclose(K - K_nojit, 1e-6 * torch.eye(7), atol=1e-14)


def test_exp_kernel_variances_match_diag():
    k = ExpKernel(amplitude=1.5, length_scales=torch.tensor([0.7]), epsilon=1e-5)
    X = _grid(6, 1)
    v = k.variances(X)
    K = k.cov(X)
    assert torch.allclose(v, torch.diagonal(K), atol=1e-14)


def test_exp_kernel_psd():
    k = ExpKernel(amplitude=1.0, length_scales=torch.tensor([0.5, 1.5]), epsilon=1e-8)
    X = _grid(20, 2)
    K = k.cov(X)
    eigs = torch.linalg.eigvalsh(K)
    assert float(eigs.min()) > -1e-10  # PSD up to float roundoff


def test_exp_kernel_gradients_flow():
    """amplitude is fixed in voila; only length_scales is optimized."""
    k = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.0, 2.0]))
    X = _grid(5, 2)
    K = k.cov(X)
    loss = K.sum()
    loss.backward()
    # amplitude is a plain float, not a Parameter
    assert isinstance(k.amplitude, float)
    assert k.length_scales.grad is not None
    assert torch.all(torch.isfinite(k.length_scales.grad))


# ---------------------------------------------------------------------------
# RQKernel  (single shared lengthscale, no ARD; reference: rq_kernel)
# ---------------------------------------------------------------------------


def test_rq_kernel_value_matches_formula():
    amp, alpha, ls = 1.4, 2.5, 0.9
    k = RQKernel(amplitude=amp, alpha=alpha, length_scale=ls)
    x = torch.tensor([[0.5, -1.0]])
    y = torch.tensor([[0.1, 0.5]])
    sqd = (0.5 - 0.1) ** 2 + (-1.0 - 0.5) ** 2
    expected = amp * (1 + sqd / (2 * alpha * ls * ls)) ** (-alpha)
    got = float(k.cov(x, y).squeeze())
    assert got == pytest.approx(expected, rel=1e-12)


def test_rq_kernel_no_ard():
    # Single shared lengthscale even in higher dims
    k = RQKernel(amplitude=1.0, alpha=1.0, length_scale=1.0)
    assert k.length_scale.numel() == 1


def test_rq_kernel_psd():
    k = RQKernel(amplitude=1.0, alpha=1.5, length_scale=1.2, epsilon=1e-8)
    X = _grid(15, 2)
    eigs = torch.linalg.eigvalsh(k.cov(X))
    assert float(eigs.min()) > -1e-10


# ---------------------------------------------------------------------------
# ExpConstKernel  (used for log-normal diffusion; reference: exp_const_kernel)
# ---------------------------------------------------------------------------


def test_exp_const_kernel_value_matches_formula():
    max_amp, exp_amp = 3.0, 0.6
    ls = torch.tensor([0.5])
    k = ExpConstKernel(max_amplitude=max_amp, exp_amplitude=exp_amp, length_scales=ls)
    x = torch.tensor([[0.4]])
    y = torch.tensor([[1.1]])
    expected = exp_amp * math.exp(-0.5 * ((0.4 - 1.1) / 0.5) ** 2) + (max_amp - exp_amp)
    got = float(k.cov(x, y).squeeze())
    assert got == pytest.approx(expected, rel=1e-12)


def test_exp_const_kernel_diagonal_equals_max_amplitude():
    # When x == y the exp term is 1 → value = exp_amp + (max_amp - exp_amp) = max_amp
    k = ExpConstKernel(max_amplitude=2.5, exp_amplitude=0.7, length_scales=torch.tensor([1.0]))
    X = _grid(8, 1)
    v = k.variances(X)
    assert torch.allclose(v, torch.full((8,), 2.5, dtype=torch.float64), atol=1e-14)


def test_exp_const_kernel_constraint_violation_raises():
    with pytest.raises(ValueError, match="exp_amplitude"):
        ExpConstKernel(max_amplitude=1.0, exp_amplitude=2.0, length_scales=torch.tensor([1.0]))


def test_exp_const_kernel_psd():
    k = ExpConstKernel(
        max_amplitude=2.0, exp_amplitude=1.5, length_scales=torch.tensor([0.7]), epsilon=1e-8
    )
    X = _grid(15, 1)
    eigs = torch.linalg.eigvalsh(k.cov(X))
    assert float(eigs.min()) > -1e-10


# ---------------------------------------------------------------------------
# SumExpKernel  (reference: sum_exp_kernels)
# ---------------------------------------------------------------------------


def test_sum_exp_kernel_value_matches_formula():
    max_amp, a1 = 2.0, 0.6
    ls1 = torch.tensor([0.5, 0.4])
    ls2 = torch.tensor([1.5, 2.0])
    k = SumExpKernel(max_amplitude=max_amp, amplitude1=a1, length_scales1=ls1, length_scales2=ls2)
    x = torch.tensor([[0.4, -0.3]])
    y = torch.tensor([[1.1, 0.2]])
    diff = (x - y).squeeze()
    e1 = math.exp(-0.5 * float(((diff / ls1) ** 2).sum()))
    e2 = math.exp(-0.5 * float(((diff / ls2) ** 2).sum()))
    expected = a1 * e1 + (max_amp - a1) * e2
    got = float(k.cov(x, y).squeeze())
    assert got == pytest.approx(expected, rel=1e-12)


def test_sum_exp_kernel_diagonal_equals_max_amplitude():
    k = SumExpKernel(
        max_amplitude=3.0,
        amplitude1=1.5,
        length_scales1=torch.tensor([0.5]),
        length_scales2=torch.tensor([1.5]),
    )
    X = _grid(6, 1)
    assert torch.allclose(k.variances(X), torch.full((6,), 3.0, dtype=torch.float64), atol=1e-14)


def test_sum_exp_kernel_psd():
    k = SumExpKernel(
        max_amplitude=2.0,
        amplitude1=0.5,
        length_scales1=torch.tensor([0.5, 0.7]),
        length_scales2=torch.tensor([1.5, 2.5]),
        epsilon=1e-8,
    )
    eigs = torch.linalg.eigvalsh(k.cov(_grid(15, 2)))
    assert float(eigs.min()) > -1e-10


# ---------------------------------------------------------------------------
# ClampedExpLinKernel  (non-stationary; reference: clamped_exp_lin_kernel)
# ---------------------------------------------------------------------------


def test_clamped_exp_lin_kernel_value_matches_formula_unclamped():
    max_amp, lin_amp, lin_c = 5.0, 0.1, 0.0
    ls = torch.tensor([1.0])
    k = ClampedExpLinKernel(
        max_amplitude=max_amp, lin_amplitude=lin_amp, lin_center=lin_c, length_scales=ls
    )
    x = torch.tensor([[0.5]])
    y = torch.tensor([[0.7]])
    p = lin_amp * (0.5 - lin_c) * (0.7 - lin_c)  # = 0.035
    assert p < max_amp
    expected = (max_amp - p) * math.exp(-0.5 * ((0.5 - 0.7) / 1.0) ** 2) + p
    got = float(k.cov(x, y).squeeze())
    assert got == pytest.approx(expected, rel=1e-12)


def test_clamped_exp_lin_kernel_clamps_when_lin_product_exceeds_max():
    # Pick large x,y so linAmp · (x-c)(y-c) > maxAmp; result should be exactly maxAmp.
    max_amp, lin_amp, lin_c = 1.0, 5.0, 0.0
    ls = torch.tensor([1.0])
    k = ClampedExpLinKernel(
        max_amplitude=max_amp, lin_amplitude=lin_amp, lin_center=lin_c, length_scales=ls
    )
    x = torch.tensor([[2.0]])
    y = torch.tensor([[3.0]])  # linProd = 5*2*3 = 30 > 1
    got = float(k.cov(x, y).squeeze())
    assert got == pytest.approx(max_amp, rel=1e-14)


def test_clamped_exp_lin_kernel_diagonal_within_bound():
    k = ClampedExpLinKernel(
        max_amplitude=4.0, lin_amplitude=0.05, lin_center=0.0, length_scales=torch.tensor([1.0])
    )
    X = _grid(10, 1, low=-1.0, high=1.0)  # keep |x|<=1 so linProd <= 0.05 << 4
    v = k.variances(X)
    # at x==x: exp term is 1 → value = (max-p)+p = max
    assert torch.allclose(v, torch.full((10,), 4.0, dtype=torch.float64), atol=1e-14)


# ---------------------------------------------------------------------------
# Cross-cutting: all kernels gradient flow + shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_factory",
    [
        lambda: ExpKernel(amplitude=1.0, length_scales=torch.tensor([0.8, 1.2])),
        lambda: RQKernel(amplitude=1.0, alpha=1.5, length_scale=0.9),
        lambda: ExpConstKernel(
            max_amplitude=2.0, exp_amplitude=1.0, length_scales=torch.tensor([0.7, 1.3])
        ),
        lambda: SumExpKernel(
            max_amplitude=2.0,
            amplitude1=1.0,
            length_scales1=torch.tensor([0.5, 0.6]),
            length_scales2=torch.tensor([1.5, 2.0]),
        ),
        lambda: ClampedExpLinKernel(
            max_amplitude=3.0,
            lin_amplitude=0.1,
            lin_center=0.0,
            length_scales=torch.tensor([1.0, 1.0]),
        ),
    ],
)
def test_kernel_gradient_flow(kernel_factory):
    k = kernel_factory()
    X = _grid(6, 2)
    K = k.cov(X)
    K.sum().backward()
    for name, p in k.named_parameters():
        assert p.grad is not None, f"{type(k).__name__}.{name} has no grad"
        assert torch.all(torch.isfinite(p.grad)), f"{type(k).__name__}.{name} grad has NaN/Inf"


def test_kernel_dtype_preserved_float64():
    k = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.0, 1.0]))
    X = torch.zeros(3, 2, dtype=torch.float64)
    assert k.cov(X).dtype is torch.float64


# ---------------------------------------------------------------------------
# Sanity: numerical reference values for a fixed configuration
# These are computed once from the closed-form expressions above and serve as
# a regression checkpoint (any change to kernel math will flip these numbers).
# ---------------------------------------------------------------------------


def test_kernel_numerical_regression_anchors():
    X = torch.tensor([[0.0], [0.5], [1.0]])
    # Exp kernel: amp=1, ls=1
    K_exp = ExpKernel(amplitude=1.0, length_scales=torch.tensor([1.0])).cov(X).detach().numpy()
    expected_exp = np.array(
        [
            [1.0, math.exp(-0.125), math.exp(-0.5)],
            [math.exp(-0.125), 1.0, math.exp(-0.125)],
            [math.exp(-0.5), math.exp(-0.125), 1.0],
        ]
    )
    np.testing.assert_allclose(K_exp, expected_exp, atol=1e-14)

    # RQ kernel: amp=1, alpha=2, ls=1
    K_rq = RQKernel(amplitude=1.0, alpha=2.0, length_scale=1.0).cov(X).detach().numpy()
    expected_rq = np.array(
        [
            [1.0, (1 + 0.0625) ** -2, (1 + 0.25) ** -2],
            [(1 + 0.0625) ** -2, 1.0, (1 + 0.0625) ** -2],
            [(1 + 0.25) ** -2, (1 + 0.0625) ** -2, 1.0],
        ]
    )
    np.testing.assert_allclose(K_rq, expected_rq, atol=1e-14)
