# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`voila-gp` is a Python/GPyTorch port of [`voila`](https://github.com/citiususc/voila), an R package for sparse-VI inference of Langevin SDE drift and diffusion (García et al., *Phys. Rev. E* 96, 022104, 2017). The Python module reproduces the algorithm verbatim against the R/C++ reference and validates against the published OU example to within ~0.5 ELBO units (scientific equivalence).

## Common commands

All commands use [`uv`](https://docs.astral.sh/uv/) — never `pip` or `conda`:

```bash
uv sync                                              # install dependencies
uv run pytest                                        # full test suite
uv run pytest tests/unit                             # unit tests only (fast)
uv run pytest -m regression                          # OU/DO/multivariate parity tests
uv run pytest tests/regression/test_ou_parity.py -s  # OU walk-through with verbose log
uv run python scripts/dump_r_reference.py            # regenerate tests/data fixtures from voila/data/*.rda
uv run python scripts/build_notebooks.py             # rebuild tutorial notebooks from source
uv run jupyter execute --inplace notebooks/*.ipynb   # execute notebooks and persist outputs
uv run ruff check src tests scripts                  # lint
uv run pyright src                                   # type-check
```

## Architecture

Module-by-module mapping to the R/C++ reference (under `voila/`, gitignored locally):

### Core (ports of voila R/C++)

| Python module | Reference | Purpose |
|--|--|--|
| `voila_gp.kernels`         | `voila/src/common_kernels.{h,cpp}`         | 5 kernels with shared `cov(X)` / `cov(X,Y)` / `variances(X)` API |
| `voila_gp.sparse_gp`       | `calculate_kernel_matrices` in `sde_variational_inferencer.cpp` | K_mm, K_mm_inv, K_nm, A, Q_ii |
| `voila_gp.elbo`            | `get_lower_bound`, `calculate_E_vector`, `calculate_ksi_vector` | Variational bound and its sub-quantities |
| `voila_gp.posterior`       | `update_distributions`                     | Closed-form drift + Laplace diffusion (Newton with analytical gradient/Hessian) |
| `voila_gp.inference`       | `do_inference`                             | Outer loop: alternation + scipy L-BFGS-B on hyperparameters |
| `voila_gp.prediction`      | `predict.sgp_sde` in `voila/R/sde_prediction.R` | Posterior drift / log-normal-diffusion at new points |
| `voila_gp.init_heuristics` | `select_diffusion_parameters` in R         | Calibrates v and kernel amplitude from data |
| `voila_gp.simulate`        | (new helper)                               | Lightweight Euler-Maruyama for tests/notebooks |

### Research extensions (new in voila-gp, not in original voila)

| Python module | Purpose |
|--|--|
| `voila_gp.multivariate` | `MultiSDEVI` fits all components in one call; aggregated `predict_drift` / `predict_diffusion`; `sample_drift_functions` for full posterior function samples |
| `voila_gp.forecast`     | `simulate_forward(fit, x0, n_steps, n_ensembles, ...)` — Monte Carlo ensemble forecast; `quantiles()` helper for prediction bands |
| `voila_gp.physics`      | `effective_potential(x, drift, g²)` returns drift potential V, log-stationary potential U, minima/maxima; `kramers_escape_time(...)` returns mean first-passage time; `stationary_density(...)` |
| `voila_gp.diagnostics`  | `predictive_log_likelihood(fit, held_out)` for held-out scoring; `compare_models(fits, held_out)` for Bayesian model selection |

The Lorenz '63 demo (`notebooks/04_lorenz63.ipynb`) is the headline use of the
extension stack: noisy 3-D chaotic trajectory → `MultiSDEVI.fit` → drift
correlations 0.999/0.995/0.998 with truth → `simulate_forward` → recovered
butterfly attractor with median 19 wing-switches per forecast (truth: 21).

### Important design decisions (non-obvious from the code)

- **Kernel `amplitude` is FIXED, not trained** — this matches voila's reference C++ where `amplitude` is closure-captured in the kernel lambda. Only shape parameters (`length_scales`, `alpha`, `exp_amplitude`, etc.) appear in `_HP_NAMES` and get optimized. Changing this affects parity with R.
- **Posterior tensors are detached during HP optimization** — the hyperparameter L-BFGS-B step holds `f_mean, f_cov, s_mean, s_cov` constant. Autograd flows only through kernel parameters, inducing points, and `v`.
- **scipy L-BFGS-B, not torch.optim.LBFGS** — voila links Fortran L-BFGS-B; scipy wraps the same routine. This minimizes cross-implementation drift in iterate counts.
- **Lower-bound tolerance is ±0.5 absolute on the OU benchmark** (≈ 1.5×10⁻⁵ relative). Tighter tolerances would falsely flag scipy-vs-Fortran convergence differences as bugs. See `tests/regression/test_ou_parity.py` for the rationale.
- **Diffusion updates use a Laplace approximation, NOT the optimal KL-minimizing Gaussian** — therefore individual diffusion updates can decrease L slightly. This matches voila's behavior. Tests should not require monotone L improvement on every diffusion step; they should require eventual convergence.
- **No R install required** — vendored fixtures live in `tests/data/*.npz` (regenerated from `voila/data/*.rda` via `scripts/dump_r_reference.py`, which uses `pyreadr`). The S4 `.RDS` regression checkpoints (`oxygen_estimates.RDS`, `multivariate_inference.RDS`) cannot be decoded without R; we validate qualitatively against the published vignette claims.

## Testing strategy

- `tests/unit/test_<module>.py` — one file per `src/voila_gp/<module>.py`. Each test names a specific quantity and tolerance, with comments justifying the value. No smoke-only tests in this layer.
- `tests/regression/test_*_parity.py` — full pipeline runs against the R reference's published anchors. Marked `@pytest.mark.regression` and `@pytest.mark.slow`.
- The OU regression test (`test_ou_parity.py`) checks per-iteration trajectory anchors against R's quoted log; the DO events / multivariate tests check qualitative properties (bistable structure, restoring force) since their `.RDS` regression checkpoints are unreachable without R.

## Reference material on disk (gitignored)

- `voila/` — original R/C++ package. Read this before changing kernel formulas, the ELBO, or the inference loop. Key files: `R/sde_vi.R`, `src/sde_variational_inferencer.{h,cpp}`, `src/common_kernels.{h,cpp}`, `R/sde_prediction.R`, `R/select_diffusion_parameters.R`, `vignettes/*.Rmd`.
- `paper/` — the published paper and its LaTeX source. Definitive notation for the lower-bound derivation.

## Out of scope / deferred

- KBR (`fit_kbr_sde`) and polynomial (`fit_polynomial_sde`) alternatives in `voila/demo/` — not part of the core VI method.
- A high-fidelity Python port of `simulate_sde` (yuima wrapper) — `voila_gp.simulate.euler_maruyama` is a minimal convenience, not a faithful yuima reproduction.
- Multi-component joint inference — currently each component is fit independently (`target_index` selects one), matching voila.
