# Test data fixtures

These NPZ files vendor the bundled R `voila` data so the Python test suite has no R dependency.

| File | Source | Shape | Notes |
|--|--|--|--|
| `do_events.npz` | `voila/data/do_events.rda` (NGRIP δ18O paleoclimate series) | (999, 1) | sampling period 1 by convention |
| `ornstein.npz` | `voila/data/ornstein.rda` (bundled OU realization) | (20001, 1) | drift `f(x) = -x`, diffusion `g(x) = sqrt(1.5)`, dt = 0.001 |

Regenerate with `uv run python scripts/dump_r_reference.py`.

## Reference numerical anchors (from the original R package)

These are the values a correct port should reproduce. They are documented in
`voila/README.md` (quoted execution log) and are expected at convergence on the
bundled OU realization.

| Quantity | Value | Tolerance |
|--|--|--|
| OU lower bound L at convergence | 36698.475 | ±0.1 |
| Drift kernel amplitude | ≈ 1.0 | ±5% |
| Drift kernel lengthscale | ≈ 1.5 | ±5% |
| Diffusion kernel parametrization | hyperparameter trajectory recorded | trajectory match per phase gate |

The `.RDS` regression checkpoints in `voila/inst/extdata/` (oxygen_estimates.RDS,
multivariate_inference.RDS) contain S4 `sgp_sde` objects that pyreadr cannot
decode. Recovering those would require a working R install with the original
voila package — out-of-band per the project plan.
