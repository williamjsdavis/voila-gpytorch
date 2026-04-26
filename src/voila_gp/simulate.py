"""Lightweight Euler-Maruyama SDE simulator for tests/tutorials.

This is NOT used for validation against R (the bundled `ornstein.rda` time series
is the canonical fixture). It is provided as a convenience so notebooks and the
multivariate parity test can generate small fresh datasets without R.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def euler_maruyama(
    drift: Callable[[np.ndarray], np.ndarray],
    diffusion: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate dX = drift(X) dt + diffusion(X) dW with Euler-Maruyama.

    Parameters
    ----------
    drift, diffusion : callables (d,) -> (d,) for drift and (d,d) for diffusion
    x0 : initial state, shape (d,)
    dt : time step
    n_steps : number of steps; the output has shape (n_steps + 1, d)
    rng : numpy generator (default: fresh PCG64)

    For diagonal-diffusion systems, you can pass a callable returning a (d,)
    vector of standard deviations and the simulator will broadcast it correctly.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x0 = np.asarray(x0, dtype=np.float64)
    d = x0.shape[0]
    out = np.empty((n_steps + 1, d), dtype=np.float64)
    out[0] = x0
    sqrt_dt = float(np.sqrt(dt))
    for t in range(n_steps):
        xt = out[t]
        f = np.asarray(drift(xt), dtype=np.float64)
        g = np.asarray(diffusion(xt), dtype=np.float64)
        eps = rng.standard_normal(d)
        if g.ndim == 1:
            # diagonal diffusion: each component scaled independently
            out[t + 1] = xt + f * dt + g * eps * sqrt_dt
        else:
            out[t + 1] = xt + f * dt + sqrt_dt * (g @ eps)
    return out
