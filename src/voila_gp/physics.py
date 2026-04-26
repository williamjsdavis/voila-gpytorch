"""Physical interpretation of fitted 1-D Langevin SDEs.

For an Itô SDE
    dX = f(X) dt + g(X) dW       (with diffusion coefficient D(x) := g²(x)/2)

we define two potentials, both useful but distinct:

  - **Drift potential** V(x) = −∫ f(x') dx'.
    Has the same units as the drift integral. Local minima of V are stable
    equilibria of the *deterministic* dynamics ẋ = f(x). This is the right
    potential to use in the **Kramers formula** for mean first-passage times.

  - **Log-stationary potential** U(x) = −∫ f(x')/D(x') dx'.
    The stationary density of the Fokker–Planck equation satisfies
        p_s(x) ∝ (1/D(x)) · exp(−U(x)).
    For additive noise (D constant) U = V/D and the two are equivalent up to
    scale; for state-dependent D they generally differ.

Both potentials are returned by `effective_potential`. The Kramers helper uses
V together with the local diffusion D at the saddle, which is the rate-
limiting region — this is the standard Hänggi–Talkner–Borkovec result and
agrees with the analytical answer on standard double-well benchmarks.

References
----------
Hänggi P., Talkner P., Borkovec M. *Reaction-rate theory: fifty years after
Kramers.* Rev. Mod. Phys. 62 (1990), 251.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PotentialAnalysis:
    """Output of `effective_potential`.

    Attributes
    ----------
    x      : grid points (regular).
    V      : drift potential, V(x) = −∫f(x')dx'  (sets baseline to zero).
    U      : log-stationary potential, U(x) = −∫(f/D)dx'  (sets baseline to zero).
    drift  : f(x) on the grid.
    D      : g²(x)/2 on the grid.
    minima : indices of local minima of V (stable equilibria).
    maxima : indices of local maxima of V (saddles / barriers).
    """

    x: np.ndarray
    V: np.ndarray
    U: np.ndarray
    drift: np.ndarray
    D: np.ndarray
    minima: np.ndarray
    maxima: np.ndarray


def _cumulative_trapezoid(integrand: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral; output has same length as x and starts at 0."""
    dx = np.diff(x)
    inc = 0.5 * (integrand[1:] + integrand[:-1]) * dx
    return np.concatenate([[0.0], np.cumsum(inc)])


def effective_potential(
    x_grid: np.ndarray,
    drift: np.ndarray,
    diffusion_g2: np.ndarray,
) -> PotentialAnalysis:
    """Compute V, U on a regular 1-D grid, locate minima and maxima of V.

    Parameters
    ----------
    x_grid       : (N,) grid (must be 1-D, monotonic increasing).
    drift        : (N,) posterior mean of f(x).
    diffusion_g2 : (N,) posterior mean of g²(x). Internally D = g²/2.
    """
    x = np.asarray(x_grid, dtype=np.float64).ravel()
    f = np.asarray(drift, dtype=np.float64).ravel()
    g2 = np.asarray(diffusion_g2, dtype=np.float64).ravel()
    if not (x.shape == f.shape == g2.shape):
        raise ValueError("x_grid, drift, diffusion_g2 must share shape")
    D = 0.5 * g2

    V = -_cumulative_trapezoid(f, x)
    V = V - V.min()
    U = -_cumulative_trapezoid(f / np.maximum(D, 1e-12), x)
    U = U - U.min()

    # Local extrema of V via sign of dV/dx
    dV = np.diff(V)
    sign = np.sign(dV)
    sign[sign == 0] = 1
    transitions = np.where(np.diff(sign) != 0)[0] + 1

    minima, maxima = [], []
    for idx in transitions:
        if 1 <= idx <= len(V) - 2:
            if V[idx] < V[idx - 1] and V[idx] < V[idx + 1]:
                minima.append(idx)
            elif V[idx] > V[idx - 1] and V[idx] > V[idx + 1]:
                maxima.append(idx)
    return PotentialAnalysis(
        x=x, V=V, U=U, drift=f, D=D,
        minima=np.array(minima, dtype=int),
        maxima=np.array(maxima, dtype=int),
    )


@dataclass
class KramersEstimate:
    """Output of `kramers_escape_time` (units of time = sampling-period units)."""

    barrier_height: float          # ΔV = V(saddle) - V(minimum)
    curvature_minimum: float       # V''(x_min) > 0
    curvature_saddle: float        # |V''(x_saddle)|
    diffusion_at_saddle: float     # D(x_saddle) used in the exponential
    mean_first_passage_time: float # τ̄
    rate: float                    # 1 / τ̄
    note: str = ""


def _second_derivative(y: np.ndarray, x: np.ndarray, i: int) -> float:
    """Three-point second-derivative estimate on a possibly non-uniform grid."""
    if i <= 0 or i >= len(x) - 1:
        raise ValueError("Cannot compute U''(x) at boundary index")
    h_l = x[i] - x[i - 1]
    h_r = x[i + 1] - x[i]
    return float(2 * (y[i + 1] * h_l - y[i] * (h_l + h_r) + y[i - 1] * h_r) /
                 (h_l * h_r * (h_l + h_r)))


def kramers_escape_time(
    pot: PotentialAnalysis,
    minimum_idx: int,
    saddle_idx: int,
) -> KramersEstimate:
    """Mean first-passage time over a barrier (Hänggi-Talkner-Borkovec).

        τ̄ = (2π / sqrt(V''(x_min) · |V''(x_saddle)|))  ·  exp(ΔV / D̄)

    with D̄ = D(x_saddle). Valid in the low-noise regime ΔV ≫ D̄. With state-
    dependent D this is the leading-order asymptote.
    """
    Upp_min = _second_derivative(pot.V, pot.x, minimum_idx)
    Upp_sad = _second_derivative(pot.V, pot.x, saddle_idx)
    delta_V = float(pot.V[saddle_idx] - pot.V[minimum_idx])
    D_saddle = float(pot.D[saddle_idx])

    curv_min = abs(Upp_min)
    curv_sad = abs(Upp_sad)
    if delta_V <= 0 or curv_min <= 0 or curv_sad <= 0 or D_saddle <= 0:
        return KramersEstimate(
            barrier_height=delta_V,
            curvature_minimum=curv_min,
            curvature_saddle=curv_sad,
            diffusion_at_saddle=D_saddle,
            mean_first_passage_time=float("inf"),
            rate=0.0,
            note="degenerate inputs — Kramers formula not applicable",
        )

    tau = (2 * np.pi / np.sqrt(curv_min * curv_sad)) * np.exp(delta_V / D_saddle)
    return KramersEstimate(
        barrier_height=delta_V,
        curvature_minimum=curv_min,
        curvature_saddle=curv_sad,
        diffusion_at_saddle=D_saddle,
        mean_first_passage_time=float(tau),
        rate=1.0 / float(tau),
    )


def stationary_density(pot: PotentialAnalysis) -> np.ndarray:
    """Normalized stationary density of the Fokker–Planck equation:

        p_s(x) ∝ (1/D(x)) · exp(−U(x)).

    Useful for visualizing where the stochastic dynamics spend time.
    """
    raw = np.exp(-pot.U) / np.maximum(pot.D, 1e-12)
    dx = np.diff(pot.x)
    Z = float(np.sum(0.5 * (raw[1:] + raw[:-1]) * dx))
    return raw / Z
