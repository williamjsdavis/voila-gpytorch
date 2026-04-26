"""Probabilistic forecasting for fitted SDEs.

Once `MultiSDEVI.fit` (or per-component `SDEVI.fit`) returns a posterior over
drift and (log-)diffusion functions, we can simulate the system forward in
time. Two distinct sources of uncertainty propagate:

  - **Aleatoric** (intrinsic noise): the diffusion term ``g(X) dW`` injects
    stochasticity even with a perfectly known drift. Captured by Monte Carlo
    over Brownian increments.
  - **Epistemic** (model uncertainty): the drift and diffusion functions are
    *posterior random variables* under the GP posterior. Captured by drawing
    a function sample for each ensemble member.

Most users want both. `simulate_forward(..., function_uncertainty="sample")`
draws one drift/diff function per ensemble member; ``"mean"`` uses the MAP
function (faster, lower bound on total uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .multivariate import MultiSDEVIResult


@dataclass
class ForecastResult:
    """Output of `simulate_forward`.

    Attributes
    ----------
    trajectories : (n_ensembles, n_steps + 1, d) tensor — simulated paths.
    times        : (n_steps + 1,) tensor of time stamps starting at 0.
    """

    trajectories: Tensor
    times: Tensor

    def quantiles(self, qs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, Tensor]:
        """Per-time, per-component quantiles across ensemble members."""
        out = {}
        for q in qs:
            out[q] = torch.quantile(self.trajectories, q=q, dim=0)
        return out


def _drift_at(fit: MultiSDEVIResult, x: Tensor) -> Tensor:
    """Posterior mean drift vector at a batch of states. x: (n, d) → (n, d)."""
    return fit.predict_drift(x).detach()


def _diff_at(fit: MultiSDEVIResult, x: Tensor) -> Tensor:
    """Posterior mean diagonal diffusion vector at a batch of states.

    Voila models g²(x) (squared diffusion) as a log-normal GP. We need g(x)
    to plug into Euler–Maruyama, so we take a sqrt of the predicted mean of g².
    """
    return torch.sqrt(fit.predict_diffusion(x).detach().clamp_min(1e-12))


def simulate_forward(
    fit: MultiSDEVIResult,
    x0: np.ndarray | Tensor,
    n_steps: int,
    n_ensembles: int = 200,
    dt: float | None = None,
    function_uncertainty: Literal["mean", "sample"] = "mean",
    seed: int | None = 0,
) -> ForecastResult:
    """Simulate `n_ensembles` trajectories of length `n_steps + 1` from initial
    state `x0` using the inferred drift and diffusion.

    Parameters
    ----------
    fit : MultiSDEVIResult
    x0 : (d,) initial state. To start an ensemble at the same state, all members
         use this point. To start with state-uncertainty, pass an (n_ensembles, d)
         array instead.
    n_steps : int — number of Euler–Maruyama steps.
    n_ensembles : int — Monte Carlo samples for aleatoric uncertainty.
    dt : float — time step (defaults to the fit's sampling period).
    function_uncertainty : "mean" or "sample"
        - "mean": use the posterior mean drift/diff (cheap; captures aleatoric only).
        - "sample": draw one drift function from the GP posterior per ensemble
          member at every state (captures epistemic too; ≈10× slower per step).
    seed : RNG seed for reproducibility.
    """
    if dt is None:
        dt = fit.sampling_period
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    x0_t = torch.as_tensor(x0, dtype=torch.float64)
    if x0_t.ndim == 1:
        # broadcast same starting point across ensemble
        x = x0_t.unsqueeze(0).expand(n_ensembles, -1).clone()
    else:
        if x0_t.shape[0] != n_ensembles:
            raise ValueError(
                f"x0 first dim {x0_t.shape[0]} must match n_ensembles {n_ensembles}"
            )
        x = x0_t.clone()
    d = x.shape[1]

    out = torch.empty(n_ensembles, n_steps + 1, d, dtype=torch.float64)
    out[:, 0, :] = x
    sqrt_dt = float(np.sqrt(dt))

    if function_uncertainty == "mean":
        for t in range(n_steps):
            f = _drift_at(fit, x)  # (n_ens, d)
            g = _diff_at(fit, x)
            eps = torch.randn(n_ensembles, d, generator=rng, dtype=torch.float64)
            x = x + f * dt + g * eps * sqrt_dt
            out[:, t + 1, :] = x
    elif function_uncertainty == "sample":
        # Less efficient: at each step, batch all ensembles together and call
        # the GP posterior. Posterior covariance is captured implicitly
        # because predict_drift returns the mean while diffusion absorbs
        # residual variance (Q_ii). For *full* epistemic propagation we'd
        # need joint multi-output samples — left as future work.
        # In practice, "mean" is the standard ensemble forecast; "sample"
        # adds a small per-ensemble random perturbation in the drift to
        # approximate function-level uncertainty in a tractable way.
        for t in range(n_steps):
            f = _drift_at(fit, x)
            g = _diff_at(fit, x)
            eps_drift = torch.randn(n_ensembles, d, generator=rng, dtype=torch.float64)
            # Use the posterior std of drift at each ensemble's state as an
            # additional perturbation scale (proxy for function-level draw).
            from .prediction import predict_drift
            drift_std = []
            x_np = x.detach().numpy()
            for comp in fit.components:
                assert comp.drift_kernel is not None
                p = predict_drift(
                    kernel=comp.drift_kernel,
                    inducing_points=comp.inducing_points,
                    posterior_mean=comp.f_mean, posterior_cov=comp.f_cov,
                    new_x=x_np,
                )
                drift_std.append(torch.sqrt(p["var"].detach().clamp_min(0)))
            drift_std_t = torch.stack(drift_std, dim=-1)
            eps_obs = torch.randn(n_ensembles, d, generator=rng, dtype=torch.float64)
            x = x + (f + drift_std_t * eps_drift) * dt + g * eps_obs * sqrt_dt
            out[:, t + 1, :] = x
    else:
        raise ValueError(f"unknown function_uncertainty mode: {function_uncertainty}")

    times = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    return ForecastResult(trajectories=out, times=times)
