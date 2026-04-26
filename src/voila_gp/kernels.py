"""Kernel functions used by voila — direct ports of voila/src/common_kernels.cpp.

Design:
- Each kernel is a torch.nn.Module so its hyperparameters live in `.parameters()`
  and participate in autograd.
- Hyperparameters are stored in their natural physical units (no log-reparametrization).
  Optimization respects bounds via scipy.optimize.minimize(method="L-BFGS-B"); see
  voila_gp.inference. This matches the R/C++ original which uses Fortran L-BFGS-B
  on bounded parameters.
- Public API (matches reference C++ kernel base class):
    * cov(X)        -> (n,n) auto-covariance, with epsilon jitter on diagonal.
    * cov(X, Y)     -> (n,m) cross-covariance.
    * variances(X)  -> (n,)   diagonal of cov(X).
- Inputs are (n, d) tensors. dtype is preserved (default float64).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "ClampedExpLinKernel",
    "ExpConstKernel",
    "ExpKernel",
    "Kernel",
    "RQKernel",
    "SumExpKernel",
]

_INF = 1e30  # finite stand-in for +inf in scipy L-BFGS-B bounds
# Small positive lower bound on amplitudes / length-scales: prevents the optimizer
# from stepping into a degenerate regime (length_scale → 0 makes the kernel a delta
# function and K_mm singular). Voila's reference uses 0; we use 1e-4 because
# scipy's L-BFGS-B can land *on* the boundary, while the original Fortran in
# voila's setup tends not to. 1e-4 is well below any physically meaningful scale.
_POS_LB = 1e-4


def _as_param(value: Tensor | float, name: str) -> nn.Parameter:
    if not isinstance(value, Tensor):
        value = torch.tensor(value, dtype=torch.float64)
    if value.dtype != torch.float64:
        value = value.to(torch.float64)
    return nn.Parameter(value.clone())


def _sqdist_per_dim(X: Tensor, Y: Tensor) -> Tensor:
    """(n,m,d) tensor of squared coordinate differences."""
    return (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2)


def _sqdist(X: Tensor, Y: Tensor) -> Tensor:
    """(n,m) sum-of-squares Euclidean squared distance."""
    return _sqdist_per_dim(X, Y).sum(dim=-1)


class Kernel(nn.Module):
    """Common base. Subclasses implement `_kernel(X, Y)` -> (n,m) covariance.

    `epsilon` jitter is added to the diagonal in `cov(X)` and `variances(X)`
    (matches reference behavior: jitter applied at matrix-assembly level, not
    inside the pairwise expression).
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__()
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        self.epsilon = float(epsilon)

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _diag(self, X: Tensor) -> Tensor:  # pragma: no cover - default impl
        # Generic but slow fallback: take diagonal of self._kernel(X, X).
        return torch.diagonal(self._kernel(X, X))

    def cov(self, X: Tensor, Y: Tensor | None = None) -> Tensor:
        if Y is None:
            K = self._kernel(X, X)
            if self.epsilon > 0:
                K = K + self.epsilon * torch.eye(X.shape[0], dtype=K.dtype, device=K.device)
            return K
        return self._kernel(X, Y)

    def variances(self, X: Tensor) -> Tensor:
        v = self._diag(X)
        if self.epsilon > 0:
            v = v + self.epsilon
        return v

    # Hyperparameter zip/unzip API — mirrors voila's kernel::get_hyperparams /
    # set_hyperparams / get_lower_bound / get_upper_bound. Each subclass declares
    # an ordered list of trainable parameter names via `_HP_NAMES` and supplies
    # `_hp_bounds()` returning (lower, upper) tensors of the flat vector shape.

    _HP_NAMES: tuple[str, ...] = ()

    def get_hyperparams(self) -> Tensor:
        """Flat vector of trainable kernel hyperparameters (in declaration order)."""
        chunks = [getattr(self, n).reshape(-1) for n in self._HP_NAMES]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)

    def set_hyperparams(self, vec: Tensor) -> None:
        """Inverse of get_hyperparams — reshape and re-assign in place (no autograd graph)."""
        offset = 0
        for n in self._HP_NAMES:
            cur = getattr(self, n)
            size = cur.numel()
            new_val = vec[offset : offset + size].reshape(cur.shape).to(cur.dtype)
            with torch.no_grad():
                cur.copy_(new_val)
            offset += size
        if offset != vec.numel():
            raise ValueError(
                f"hyperparameter vector size mismatch: kernel takes {offset}, got {vec.numel()}"
            )

    def hp_bounds(self) -> tuple[Tensor, Tensor]:
        """Lower/upper bounds (1-D tensors matching get_hyperparams())."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete kernels — one-to-one with voila/src/common_kernels.cpp.
# ---------------------------------------------------------------------------


class ExpKernel(Kernel):
    """k(x,y) = amplitude · exp(-Σ_d ((x_d - y_d)/L_d)² / 2). ARD via L_d.

    Reference: exponential_kernel in voila/src/common_kernels.cpp.

    `amplitude` is a FIXED hyperparameter (set at construction, not trained) to
    match voila's design where amplitude is closure-captured in the kernel lambda.
    Only `length_scales` is optimized.
    """

    _HP_NAMES = ("length_scales",)

    def __init__(
        self, amplitude: float, length_scales: Tensor, epsilon: float = 0.0
    ) -> None:
        super().__init__(epsilon=epsilon)
        if amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        self.amplitude = float(amplitude)  # fixed (non-trainable; matches voila)
        self.length_scales = _as_param(length_scales, "length_scales")
        if self.length_scales.ndim != 1:
            raise ValueError("length_scales must be 1-D (per-dimension ARD)")

    def hp_bounds(self) -> tuple[Tensor, Tensor]:
        d = self.length_scales.numel()
        lower = torch.full((d,), _POS_LB, dtype=torch.float64)
        upper = torch.full((d,), _INF, dtype=torch.float64)
        return lower, upper

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        scaled = _sqdist_per_dim(X, Y) / self.length_scales.pow(2)
        return self.amplitude * torch.exp(-0.5 * scaled.sum(dim=-1))

    def _diag(self, X: Tensor) -> Tensor:
        return torch.full((X.shape[0],), self.amplitude, dtype=X.dtype, device=X.device)


class RQKernel(Kernel):
    """k(x,y) = amplitude · (1 + ||x-y||² / (2 α L²))^(-α).

    Single shared lengthscale L (NO ARD), matches reference rational_quadratic_kernel.

    `amplitude` is FIXED at construction (closure-captured in voila's reference C++);
    only `alpha` and `length_scale` are optimized.
    """

    _HP_NAMES = ("alpha", "length_scale")

    def __init__(
        self,
        amplitude: float,
        alpha: float | Tensor,
        length_scale: float | Tensor,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(epsilon=epsilon)
        if amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        self.amplitude = float(amplitude)  # fixed
        self.alpha = _as_param(alpha, "alpha")
        ls = torch.as_tensor(length_scale, dtype=torch.float64).reshape(())
        self.length_scale = nn.Parameter(ls.clone())

    def hp_bounds(self) -> tuple[Tensor, Tensor]:
        lower = torch.full((2,), _POS_LB, dtype=torch.float64)
        upper = torch.full((2,), _INF, dtype=torch.float64)
        return lower, upper

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        d2 = _sqdist(X, Y)
        return self.amplitude * (1 + d2 / (2 * self.alpha * self.length_scale.pow(2))).pow(
            -self.alpha
        )

    def _diag(self, X: Tensor) -> Tensor:
        return torch.full((X.shape[0],), self.amplitude, dtype=X.dtype, device=X.device)


class ExpConstKernel(Kernel):
    """k(x,y) = expAmp · exp(-Σ_d ((x_d - y_d)/L_d)² / 2) + (maxAmp - expAmp).

    Used for the log-normal diffusion GP. Constant offset ensures positivity is
    well-conditioned after exp() back-transform.

    Reference: exponential_constant_kernel. Constraint: 0 ≤ expAmp ≤ maxAmp.
    """

    _HP_NAMES = ("exp_amplitude", "length_scales")

    def __init__(
        self,
        max_amplitude: float,
        exp_amplitude: float | Tensor,
        length_scales: Tensor,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(epsilon=epsilon)
        if max_amplitude < 0:
            raise ValueError("max_amplitude must be non-negative")
        ea = float(exp_amplitude.item() if isinstance(exp_amplitude, Tensor) else exp_amplitude)
        if ea < 0 or ea > max_amplitude:
            raise ValueError(
                f"exp_amplitude={ea} must satisfy 0 ≤ exp_amplitude ≤ max_amplitude={max_amplitude}"
            )
        self.max_amplitude = float(max_amplitude)  # fixed (non-trainable; matches voila)
        self.exp_amplitude = _as_param(exp_amplitude, "exp_amplitude")
        self.length_scales = _as_param(length_scales, "length_scales")

    def hp_bounds(self) -> tuple[Tensor, Tensor]:
        d = self.length_scales.numel()
        # exp_amplitude lower bound is 0 (vs _POS_LB) — at exp_amp=0 the kernel
        # reduces to a constant maxAmp, which is still PSD (rank-1) but voila's
        # downstream code adds epsilon jitter so K_mm stays non-singular.
        lower = torch.cat(
            [
                torch.zeros(1, dtype=torch.float64),
                torch.full((d,), _POS_LB, dtype=torch.float64),
            ]
        )
        upper = torch.cat(
            [
                torch.tensor([self.max_amplitude], dtype=torch.float64),
                torch.full((d,), _INF, dtype=torch.float64),
            ]
        )
        return lower, upper

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        scaled = _sqdist_per_dim(X, Y) / self.length_scales.pow(2)
        rbf = torch.exp(-0.5 * scaled.sum(dim=-1))
        return self.exp_amplitude * rbf + (self.max_amplitude - self.exp_amplitude)

    def _diag(self, X: Tensor) -> Tensor:
        return torch.full(
            (X.shape[0],),
            self.max_amplitude,
            dtype=X.dtype,
            device=X.device,
        )


class SumExpKernel(Kernel):
    """k(x,y) = A1·exp(-||(x-y)/L1||²/2) + (maxAmp - A1)·exp(-||(x-y)/L2||²/2).

    Two-component RBF mixture with constraint A1 + A2 = maxAmp (so amplitude at
    zero distance equals maxAmp). Reference: sum_exponential_kernels.
    """

    _HP_NAMES = ("amplitude1", "length_scales1", "length_scales2")

    def __init__(
        self,
        max_amplitude: float,
        amplitude1: float | Tensor,
        length_scales1: Tensor,
        length_scales2: Tensor,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(epsilon=epsilon)
        if max_amplitude < 0:
            raise ValueError("max_amplitude must be non-negative")
        a1 = float(amplitude1.item() if isinstance(amplitude1, Tensor) else amplitude1)
        if a1 < 0 or a1 > max_amplitude:
            raise ValueError(
                f"amplitude1={a1} must satisfy 0 ≤ amplitude1 ≤ max_amplitude={max_amplitude}"
            )
        if length_scales1.shape != length_scales2.shape:
            raise ValueError("length_scales1 and length_scales2 must share shape")
        self.max_amplitude = float(max_amplitude)
        self.amplitude1 = _as_param(amplitude1, "amplitude1")
        self.length_scales1 = _as_param(length_scales1, "length_scales1")
        self.length_scales2 = _as_param(length_scales2, "length_scales2")

    def hp_bounds(self) -> tuple[Tensor, Tensor]:
        d = self.length_scales1.numel()
        lower = torch.cat(
            [
                torch.zeros(1, dtype=torch.float64),
                torch.full((2 * d,), _POS_LB, dtype=torch.float64),
            ]
        )
        upper = torch.cat(
            [
                torch.tensor([self.max_amplitude], dtype=torch.float64),
                torch.full((2 * d,), _INF, dtype=torch.float64),
            ]
        )
        return lower, upper

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        s1 = _sqdist_per_dim(X, Y) / self.length_scales1.pow(2)
        s2 = _sqdist_per_dim(X, Y) / self.length_scales2.pow(2)
        e1 = torch.exp(-0.5 * s1.sum(dim=-1))
        e2 = torch.exp(-0.5 * s2.sum(dim=-1))
        return self.amplitude1 * e1 + (self.max_amplitude - self.amplitude1) * e2

    def _diag(self, X: Tensor) -> Tensor:
        return torch.full((X.shape[0],), self.max_amplitude, dtype=X.dtype, device=X.device)


class ClampedExpLinKernel(Kernel):
    """Non-stationary kernel mixing a clamped linear and a modulated RBF.

    Let p(x,y) = linAmp · Σ_d (x_d - c)(y_d - c).
    If p > maxAmp:        k = maxAmp.
    Else:                 k = (maxAmp - p) · exp(-Σ_d ((x_d-y_d)/L_d)² / 2) + p.

    Reference: clamped_exponential_linear_kernel.
    Note: lin_center is a single scalar broadcast across all dimensions
    (matches Armadillo `dot(x - linCenter, y - linCenter)` semantics).
    """

    _HP_NAMES = ("lin_amplitude", "lin_center", "length_scales")

    def __init__(
        self,
        max_amplitude: float,
        lin_amplitude: float | Tensor,
        lin_center: float | Tensor,
        length_scales: Tensor,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(epsilon=epsilon)
        if max_amplitude < 0:
            raise ValueError("max_amplitude must be non-negative")
        self.max_amplitude = float(max_amplitude)
        self.lin_amplitude = _as_param(lin_amplitude, "lin_amplitude")
        lc = torch.as_tensor(lin_center, dtype=torch.float64).reshape(())
        self.lin_center = nn.Parameter(lc.clone())
        self.length_scales = _as_param(length_scales, "length_scales")

    def hp_bounds(self) -> tuple[Tensor, Tensor]:
        d = self.length_scales.numel()
        lower = torch.cat(
            [
                torch.tensor([0.0, -_INF], dtype=torch.float64),
                torch.full((d,), _POS_LB, dtype=torch.float64),
            ]
        )
        upper = torch.full((2 + d,), _INF, dtype=torch.float64)
        return lower, upper

    def _kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        Xc = X - self.lin_center
        Yc = Y - self.lin_center
        # Σ_d (x_d - c)(y_d - c)  →  (n,m)
        lin_dot = (Xc.unsqueeze(1) * Yc.unsqueeze(0)).sum(dim=-1)
        p = self.lin_amplitude * lin_dot
        scaled = _sqdist_per_dim(X, Y) / self.length_scales.pow(2)
        rbf = torch.exp(-0.5 * scaled.sum(dim=-1))
        unclamped = (self.max_amplitude - p) * rbf + p
        # torch.where preserves grads on both branches; clamped branch is constant
        return torch.where(
            p > self.max_amplitude,
            torch.full_like(unclamped, self.max_amplitude),
            unclamped,
        )

    def _diag(self, X: Tensor) -> Tensor:
        Xc = X - self.lin_center
        # p(x,x) = linAmp · ||x - c||²
        p = self.lin_amplitude * Xc.pow(2).sum(dim=-1)
        # at x==y, exp() is 1, so unclamped = (max - p) · 1 + p = max
        return torch.where(
            p > self.max_amplitude,
            torch.full_like(p, self.max_amplitude),
            torch.full_like(p, self.max_amplitude),
        )
