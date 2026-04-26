"""Tests for physics module: effective potential, Kramers rate, stationary density."""

from __future__ import annotations

import numpy as np

from voila_gp.physics import (
    effective_potential,
    kramers_escape_time,
    stationary_density,
)


def test_effective_potential_recovers_quadratic_for_OU():
    """For OU dX = -x dt + sqrt(2D₀) dW, V(x) = x²/2 and U(x) = V/D₀ = x²/2 (D₀=1)."""
    x = np.linspace(-3, 3, 401)
    drift = -x
    g2 = np.full_like(x, 2.0)  # so D = 1.0
    pot = effective_potential(x, drift, g2)
    expected_V = x**2 / 2
    expected_V = expected_V - expected_V.min()
    assert np.allclose(pot.V, expected_V, atol=1e-3)
    # U = V / D = V (since D=1)
    assert np.allclose(pot.U, expected_V, atol=1e-3)
    assert len(pot.minima) == 1
    assert abs(x[pot.minima[0]]) < 0.05
    assert len(pot.maxima) == 0


def test_effective_potential_state_dependent_diffusion_distinguishes_V_and_U():
    """When D(x) varies, V and U should differ."""
    x = np.linspace(-2, 2, 401)
    drift = -x
    # D(x) = 1 + 0.5 x²  → state-dependent
    g2 = 2.0 * (1.0 + 0.5 * x**2)
    pot = effective_potential(x, drift, g2)
    # V is still x²/2; U is different (smaller in magnitude where D is larger)
    expected_V = x**2 / 2
    expected_V -= expected_V.min()
    assert np.allclose(pot.V, expected_V, atol=1e-3)
    # U at x=±2: ∫₀² x/(1+0.5 x²) dx = ln(1 + 0.5·4)/0.5·0.5 ... let me just check that U != V
    assert not np.allclose(pot.U, pot.V, atol=1e-2)


def test_effective_potential_finds_double_well():
    """Drift f(x) = x - x³ has potential U(x) = -x²/2 + x⁴/4 (double well)."""
    x = np.linspace(-2.0, 2.0, 401)
    drift = x - x**3
    D = 0.5
    g2 = np.full_like(x, 2 * D)
    pot = effective_potential(x, drift, g2)
    # Two minima at ±1, one maximum at 0
    assert len(pot.minima) == 2
    assert len(pot.maxima) == 1
    min_xs = np.sort(x[pot.minima])
    max_xs = x[pot.maxima]
    assert abs(min_xs[0] - (-1.0)) < 0.05
    assert abs(min_xs[1] - 1.0) < 0.05
    assert abs(max_xs[0] - 0.0) < 0.05


def test_kramers_rate_double_well():
    """Symmetric double well x⁴/4 - x²/2, additive noise D=0.05.
    Kramers' barrier height ΔU = 0.25; curvatures at min are 2 (at ±1), at saddle 1 (at 0).
    τ̄ = 2π/sqrt(2·1) · exp(0.25/0.05) = 2π/√2 · e⁵ ≈ 659.5
    """
    x = np.linspace(-2.0, 2.0, 801)
    drift = x - x**3
    D = 0.05
    g2 = np.full_like(x, 2 * D)
    pot = effective_potential(x, drift, g2)
    # Use the right-hand minimum (x≈+1) and the saddle (x≈0)
    right_min_idx = pot.minima[np.argmax(x[pot.minima])]
    saddle_idx = pot.maxima[0]
    est = kramers_escape_time(pot, minimum_idx=right_min_idx, saddle_idx=saddle_idx)
    expected_tau = 2 * np.pi / np.sqrt(2.0 * 1.0) * np.exp(0.25 / 0.05)
    assert abs(est.barrier_height - 0.25) < 0.01
    assert abs(est.curvature_minimum - 2.0) < 0.05
    assert abs(est.curvature_saddle - 1.0) < 0.05
    # Kramers tau within 5% of analytical
    assert abs(est.mean_first_passage_time - expected_tau) / expected_tau < 0.05


def test_stationary_density_normalizes_and_matches_OU():
    """For OU, stationary density is N(0, D₀) = N(0, 1).  ∫ p dx = 1, peak at 0."""
    x = np.linspace(-5, 5, 601)
    drift = -x
    g2 = np.full_like(x, 2.0)
    pot = effective_potential(x, drift, g2)
    p = stationary_density(pot)
    # Normalized
    dx = np.diff(x)
    integral = float(np.sum(0.5 * (p[1:] + p[:-1]) * dx))
    assert abs(integral - 1.0) < 1e-3
    # Peak at zero
    assert abs(x[np.argmax(p)]) < 0.05
    # Variance should be ≈ 1
    mean_x = float(np.sum(0.5 * (x[1:] * p[1:] + x[:-1] * p[:-1]) * dx))
    var_x = float(np.sum(0.5 * ((x[1:] - mean_x) ** 2 * p[1:] + (x[:-1] - mean_x) ** 2 * p[:-1]) * dx))
    assert abs(mean_x) < 0.01
    assert abs(var_x - 1.0) < 0.05
