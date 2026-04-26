"""Posterior diagnostics: held-out predictive log-likelihood and model comparison.

The ELBO is a lower bound on the marginal likelihood, so a *higher* ELBO can
be used as a Bayesian model-selection criterion when comparing fits trained
on the same data. For a more rigorous comparison the predictive log-likelihood
of a held-out trajectory is preferred — this is computed by walking through
the held-out data and accumulating the conditional Gaussian likelihood of
each increment under the inferred posterior drift and diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import SDEVIResult
from .prediction import predict_diffusion, predict_drift


@dataclass
class ModelComparison:
    """Output of `compare_models` — sorted from best to worst."""

    names: list[str]
    elbos: list[float]
    held_out_log_likelihoods: list[float]


def predictive_log_likelihood(
    fit: SDEVIResult,
    held_out_time_series: np.ndarray,
    sampling_period: float | None = None,
    target_index: int = 0,
) -> float:
    """Sum of log p(ΔX_i | X_i, fit) under the inferred posterior on the
    target component, evaluated on a *held-out* trajectory.

    Increment i is modelled as
        ΔX_i ~ Normal(h · f(X_i), h · g²(X_i))
    using posterior-mean drift and diffusion (the standard Euler–Maruyama
    approximation). This gives an honest predictive score that penalizes
    overfitting (high score on training data is easy; on held-out data it's
    informative).
    """
    if sampling_period is None:
        sampling_period = fit.sampling_period
    h = float(sampling_period)
    x = np.asarray(held_out_time_series, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    head = x[:-1]
    dx = (x[1:] - x[:-1])[:, target_index]

    assert fit.drift_kernel is not None and fit.diff_kernel is not None
    drift = predict_drift(
        kernel=fit.drift_kernel, inducing_points=fit.inducing_points,
        posterior_mean=fit.f_mean, posterior_cov=fit.f_cov,
        new_x=head,
    )
    diff = predict_diffusion(
        kernel=fit.diff_kernel, inducing_points=fit.inducing_points,
        posterior_mean=fit.s_mean, posterior_cov=fit.s_cov,
        new_x=head, v=fit.v,
    )
    f_mu = drift["mean"].detach().cpu().numpy()
    g2 = diff["mean"].detach().cpu().numpy().clip(min=1e-12)

    var = h * g2
    mu = h * f_mu
    # log N(dx | mu, var)
    ll = -0.5 * (np.log(2 * np.pi * var) + (dx - mu) ** 2 / var)
    return float(ll.sum())


def compare_models(
    fits: dict[str, SDEVIResult],
    held_out_time_series: np.ndarray | None = None,
    sampling_period: float | None = None,
    target_index: int = 0,
) -> ModelComparison:
    """Compare multiple fits by their final ELBO and (optionally) by their
    predictive log-likelihood on a held-out trajectory.

    Returns the comparison sorted by held-out log-likelihood (or ELBO if no
    held-out data is provided), best first.
    """
    names = list(fits.keys())
    elbos = [fits[n].lower_bound_history[-1] for n in names]
    if held_out_time_series is not None:
        lls = [
            predictive_log_likelihood(
                fits[n], held_out_time_series,
                sampling_period=sampling_period, target_index=target_index,
            )
            for n in names
        ]
        order = np.argsort(lls)[::-1]
    else:
        lls = [float("nan")] * len(names)
        order = np.argsort(elbos)[::-1]
    return ModelComparison(
        names=[names[i] for i in order],
        elbos=[elbos[i] for i in order],
        held_out_log_likelihoods=[lls[i] for i in order],
    )
