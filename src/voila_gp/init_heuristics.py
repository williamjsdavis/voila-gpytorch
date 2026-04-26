"""Initialization heuristics — direct port of voila/R/select_diffusion_parameters.R.

The diffusion GP is parametrized in log-space (log g²(x)) with prior mean v.
`select_diffusion_parameters` calibrates v and the kernel amplitude from the
empirical variance of the differentiated time series so that the prior matches
the observed scale of fluctuations.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

# R's mad() default constant: 1 / Φ⁻¹(0.75) ≈ 1.4826022185056. Hard-coded so the
# port is independent of scipy version-specific defaults.
_R_MAD_CONSTANT = 1.4826


def mad(x: np.ndarray | torch.Tensor) -> float:
    """Median absolute deviation, scaled like R's `mad()` default (constant=1.4826).

    Returns a Python float so it composes cleanly with downstream scalar arithmetic.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    return _R_MAD_CONSTANT * float(np.median(np.abs(arr - np.median(arr))))


def select_diffusion_parameters(
    x: Any,
    sampling_period: float,
    prior_on_sd: float,
    target_index: int = 0,
    var_x: float | None = None,
) -> dict:
    """Return the diffusion-GP prior mean v and kernel amplitude.

    Reference R:
        varX = mad(diff(x))^2
        kernelAmplitude = log(1 + (priorOnSd * h / varX)^2)
        v = log(varX / h) - kernelAmplitude/2

    Parameters
    ----------
    x : 1-D or 2-D array-like (n,) or (n, d)
        Densely-observed time series.
    sampling_period : float
        Δt between consecutive samples.
    prior_on_sd : float
        Prior belief about the maximum amplitude change of the diffusion term.
    target_index : int
        Which column of `x` to use when 2-D (matches R's `responseVariableIndex`,
        but 0-indexed).
    var_x : float, optional
        Override the empirical variance estimate (matches R's `varX` argument).
    """
    arr = np.asarray(x, dtype=np.float64)
    col = arr[:, target_index] if arr.ndim == 2 else arr.ravel()
    if var_x is None:
        var_x = mad(np.diff(col)) ** 2
    kernel_amplitude = math.log(1 + (prior_on_sd * sampling_period / var_x) ** 2)
    v = math.log(var_x / sampling_period) - kernel_amplitude / 2
    return {"v": v, "kernel_amplitude": kernel_amplitude}
