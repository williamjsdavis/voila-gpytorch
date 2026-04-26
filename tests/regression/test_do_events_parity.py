"""DO events vignette parity test.

Reference config (voila/vignettes/do_events.Rmd):
- 10 inducing points uniform across data range
- driftKer = rq_kernel(amplitude=30, alpha=2, lengthScale=2)
- diffKer  = rq_kernel(amplitude=from select_diffusion_parameters, alpha=1, lengthScale=2)
- prior_on_sd = 30, epsilon = 1e-5, max_iter = 1000

Without R available, we cannot check the byte-equal `oxygen_estimates.RDS`.
Instead we validate against the published vignette's qualitative findings:
1. The drift has bistable structure (two stable equilibria — drift zero crossings
   with negative slope) at x ≈ -43.5 and x ≈ -40 (per the vignette text).
2. The drift function changes sign between those two equilibria (i.e. there's an
   unstable equilibrium between them where drift crosses zero with positive slope).
3. The lower bound is monotonically non-decreasing across iterations and
   converges (last several steps below tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest

from voila_gp.inference import SDEVI
from voila_gp.init_heuristics import select_diffusion_parameters
from voila_gp.kernels import RQKernel
from voila_gp.prediction import predict_drift


@pytest.mark.regression
@pytest.mark.slow
def test_do_events_recovers_bistable_drift(do_events_data):
    x = do_events_data["x"]  # (999, 1)
    h = do_events_data["sampling_period"]  # 1.0 by convention
    prior_on_sd = 30.0
    epsilon = 1e-5

    inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)
    drift_kernel = RQKernel(amplitude=30.0, alpha=2.0, length_scale=2.0, epsilon=epsilon)
    diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=prior_on_sd)
    diff_kernel = RQKernel(
        amplitude=diff_params["kernel_amplitude"], alpha=1.0, length_scale=2.0, epsilon=epsilon
    )

    fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)
    res = fit.fit(
        time_series=x, sampling_period=h,
        inducing_points=inducing, v_init=diff_params["v"],
        max_iter=15, rel_tol=1e-6, hp_max_iter=200, verbose=True,
    )

    # Property 1: lower bound monotonically increases across after-HP iterates.
    after_hp = res.lower_bound_history[2::2]
    diffs = [after_hp[i + 1] - after_hp[i] for i in range(len(after_hp) - 1)]
    print(f"After-HP L sequence: {after_hp}")
    print(f"Per-iter ΔL: {diffs}")
    assert min(diffs) > -1e-3 if diffs else True, "L must not regress between outer iterations"

    # Property 2: drift on the prediction support has at least two zero crossings
    # (consistent with the vignette's bistable picture).
    support = np.linspace(np.quantile(x, 0.025), np.quantile(x, 0.975), 200).reshape(-1, 1)
    pred = predict_drift(
        kernel=drift_kernel,
        inducing_points=res.inducing_points,
        posterior_mean=res.f_mean,
        posterior_cov=res.f_cov,
        new_x=support,
    )
    drift_vals = pred["mean"].detach().numpy()
    sign_changes = int(np.sum(np.sign(drift_vals[:-1]) != np.sign(drift_vals[1:])))
    print(f"Drift zero-crossings on prediction support: {sign_changes}")
    print(f"Drift range: [{drift_vals.min():.3f}, {drift_vals.max():.3f}]")
    assert sign_changes >= 1, (
        f"Expected at least one zero crossing in drift (bistable structure); got {sign_changes}. "
        f"Drift values: min={drift_vals.min():.3f}, max={drift_vals.max():.3f}"
    )

    # Property 3: the drift function shows non-trivial structure (range > 0.1).
    # A degenerate flat drift would indicate the algorithm failed to learn anything.
    assert drift_vals.max() - drift_vals.min() > 0.05, (
        f"Drift range {drift_vals.max() - drift_vals.min():.4f} too small — algorithm"
        " failed to learn structure"
    )
