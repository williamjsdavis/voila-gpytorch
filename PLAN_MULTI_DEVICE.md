# Plan: multi-device support + performance comparison

## Context

`voila-gp` currently runs on CPU only. All tensors are allocated with the
default device, and `torch.float64` is hard-coded throughout. This is fine for
the published validation problems (OU n=20k converges in ~5 s on M-series CPU)
but leaves performance on the table for larger problems and provides no answer
to the practical question users ask: "which device should I run this on?"

This plan implements **Option 1** from our earlier design discussion: thread
the user-chosen device through the codebase so the GP linear algebra (which is
where ~75% of the wall-clock lives) executes on GPU when one is available. The
scipy L-BFGS-B optimizer stays on CPU; the per-iteration boundary crossing is
17 float64s and is negligible.

The plan also lays out a **three-machine performance comparison** —
MacBook CPU, MacBook MPS, A10 CUDA — plus an R baseline, collated into a
single notebook so users can see what each device buys them.

## Confirmed scope

**In scope:**

- `device` and `dtype` parameters on `SDEVI`, `MultiSDEVI`, `_State`, and the
  user-facing entry points.
- MPS compatibility shims for `torch.cholesky_inverse` and `torch.special.ndtri`
  (likely missing/buggy on MPS in PyTorch 2.11).
- A per-device validation tier in the test suite (CPU/float64 stays strict;
  MPS/float32 and CUDA/float64 each get their own tolerance band).
- A reproducible benchmark harness (`scripts/run_device_benchmark.py`) that
  dumps timing JSON for a fixed problem set on whatever device is requested.
- An R-side benchmark script (`scripts/run_r_benchmark.R`) that times the same
  problems through the original `voila` package.
- A collation notebook (`notebooks/07_device_comparison.ipynb`) that loads the
  JSONs from each machine and produces a single comparison chart.

**Out of scope** (deferred):

- Replacing scipy L-BFGS-B with `torch.optim.LBFGS` (would break R parity).
- Adam / natural-gradient alternatives.
- Batched fitting (one GPU kernel for many time series).
- Re-implementing L-BFGS-B in pure torch.

## Architecture: how device threading works

Every long-lived tensor (kernel parameters, posterior means/covariances,
inducing points, pre-computed K_mm etc.) lives on the chosen device with the
chosen dtype. New tensors are allocated using `device=ref.device, dtype=ref.dtype`
borrowed from a parent tensor. The scipy bridge is the single point where
data crosses to CPU/numpy:

```
   GPU (where the math lives)        CPU (scipy iterates)
   ─────────────────────────         ────────────────────
   forward  → K_mm, K_nm, A, Q_ii
   forward  → ELBO scalar  ───────► .item()
   backward → 17-float grad ──────► .cpu().numpy()  ──► scipy step
                                    np.array(x_new) ──► back to GPU
```

The boundary cost is one tensor of length ≈ `n_drift_hp + n_diff_hp + m·d + 1`
per L-BFGS-B function evaluation. For OU that's 16 floats × ~50 evals × 5 outer
iters ≈ 4000 floats over the full fit — utterly negligible.

### Dtype policy

| Device | Dtype | Notes |
|--|--|--|
| CPU | float64 | matches existing R-parity validation (no change) |
| CUDA | float64 | A10 has full FP64 support (slower than FP32 but still ~hundreds of GFLOPS — fine for our small linalg) |
| MPS | float32 | Apple Silicon Metal backend has no FP64 for linalg ops |

A small helper `_default_dtype_for(device) -> torch.dtype` codifies this.

## Phase-by-phase execution

### Phase 1 — Device threading (≈ 2 h)

Make every public entry point accept `device=` and `dtype=` (with sensible
defaults), thread them through, and verify nothing breaks on the existing
CPU path.

Files modified:

- `src/voila_gp/kernels.py`
  - `_as_param(value, name, *, device, dtype)` — propagate kwargs.
  - `Kernel.__init__` accepts `device, dtype`.
  - All `torch.zeros/full/eye/...` calls in `hp_bounds`, `_diag`, `cov` already
    use `K.dtype/device` — verify and fix any stragglers.
  - Kernel subclasses' `__init__` accept and forward `device, dtype`.
- `src/voila_gp/sparse_gp.py`
  - `_symmetric_inverse(A)` — already device-clean, but add MPS fallback (see
    Phase 2).
  - `sparse_gp_intermediates` is device-clean; `clamp_min(0.0)` works on all
    devices.
- `src/voila_gp/elbo.py`
  - `lower_bound(model)` — verify `slogdet`, `einsum`, `trace` all work on
    target devices. The constants `math.log(2*math.pi)` etc. are Python floats
    and broadcast correctly.
- `src/voila_gp/posterior.py`
  - `update_drift_closed_form`, `update_diffusion_laplace` — `torch.diag(E)`
    inherits device, `torch.linalg.solve` works on MPS/CUDA.
- `src/voila_gp/inference.py`
  - `SDEVI.__init__(self, drift_kernel, diff_kernel, *, device=None, dtype=None)`.
    If `device is None`, pick the first available among `cuda > mps > cpu` or
    just default to CPU; if `dtype is None`, use `_default_dtype_for(device)`.
  - `_State` dataclass gets `device: torch.device, dtype: torch.dtype` fields.
  - `initialize` allocates `f_mean`, `f_cov`, `s_mean`, `s_cov`, `inducing_points`
    on `device` with `dtype`.
  - `_zip_hp` returns numpy arrays as it does today (numpy is the scipy
    interface). `_unzip_hp` and the inner `fun(x_np, layout)` callback push
    inputs back to `device, dtype` before assigning.
  - `_hp_objective`'s `fun` callback: gradients gathered as `.detach().cpu().numpy()`.
  - `SDEVIResult` adds `device: torch.device` and `dtype: torch.dtype` for
    record-keeping.
- `src/voila_gp/multivariate.py`
  - `MultiSDEVI.__init__(..., *, device=None, dtype=None)` — passes through to
    each per-component `SDEVI`.
  - `MultiSDEVIResult.predict_drift / predict_diffusion` — push input `new_x`
    onto the components' device.
- `src/voila_gp/forecast.py`
  - `simulate_forward(...)`: allocate `out`, `times`, `eps` on the fit's device.
  - **Cache `K_mm_inv` once** at simulator construction. Currently each Euler
    step calls `predict_drift` which rebuilds sparse-GP intermediates — this is
    wasteful on any device and will dominate on GPU due to kernel-launch overhead.
- `src/voila_gp/prediction.py`
  - `_qnorm(p, mean, sd)`: see Phase 2 (`ndtri` may not be on MPS).
- `src/voila_gp/diagnostics.py`
  - Threaded device handling for `predictive_log_likelihood` (currently calls
    `.detach().numpy()` which forces CPU — keep that since the formula is
    pure numpy reduction, but ensure the input tensors can come from any device).

Public API examples after the change:

```python
# Default — CPU, float64 (existing behavior, no breakage)
fit = SDEVI(drift_kernel, diff_kernel)

# CUDA on A10 — float64 by default
fit = SDEVI(drift_kernel, diff_kernel, device="cuda")

# MPS on Mac — float32 (chosen automatically because MPS lacks FP64)
fit = SDEVI(drift_kernel, diff_kernel, device="mps")

# Explicit override
fit = SDEVI(drift_kernel, diff_kernel, device="cuda", dtype=torch.float32)
```

### Phase 2 — MPS compatibility shims (≈ 30 min)

Add `src/voila_gp/_compat.py` with two shims:

```python
def cholesky_inverse_compat(L: Tensor) -> Tensor:
    """Inverse of L L^T from its Cholesky factor.

    `torch.cholesky_inverse` is unavailable on MPS in some PyTorch builds; fall
    back to `torch.cholesky_solve(I, L)` which is.
    """
    if L.device.type == "mps":
        I = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device)
        return torch.cholesky_solve(I, L)
    return torch.cholesky_inverse(L)


def ndtri_compat(p: Tensor) -> Tensor:
    """Inverse standard-normal CDF.

    `torch.special.ndtri` may not be on MPS; fall back to `erfinv` form:
        Φ⁻¹(p) = sqrt(2) · erfinv(2p - 1)
    """
    return torch.sqrt(torch.tensor(2.0, dtype=p.dtype, device=p.device)) * \
           torch.erfinv(2 * p - 1)
```

Replace direct calls in `sparse_gp.py` and `prediction.py` with the shim
versions. Add a tiny test (`tests/unit/test_compat.py`) that verifies the
shim outputs match the direct calls on CPU to ~1e-12.

### Phase 3 — Validation per device (≈ 30 min)

Add `tests/unit/test_devices.py`:

```python
@pytest.mark.parametrize("device", _available_devices())
def test_ou_runs_on_device(device, ornstein_data):
    """Smoke test: full OU fit completes on each available device."""
    ...

@pytest.mark.parametrize("device", _available_devices())
def test_kernel_cov_matches_cpu_reference(device):
    """Numerical match: kernel covariance on `device` agrees with CPU/float64
    within the device's tolerance band."""
    ...
```

`_available_devices()` returns `["cpu"]` plus `["mps"]` if MPS available plus
`["cuda"]` if CUDA available. Tolerance bands codified in a single dict:

```python
DEVICE_TOLERANCES = {
    "cpu":  {"kernel_atol": 1e-12, "elbo_atol": 0.5},   # the existing R-parity bar
    "cuda": {"kernel_atol": 1e-12, "elbo_atol": 0.5},   # FP64 = same bar as CPU
    "mps":  {"kernel_atol": 1e-5,  "elbo_atol": 50.0},  # FP32 — looser
}
```

Add a documented note in `tests/regression/test_ou_parity.py` that the strict
±1.0 ELBO band applies on CPU/float64 only; the same test on MPS would need
the looser band.

### Phase 4 — Benchmark harness (≈ 1 h)

Create `scripts/run_device_benchmark.py`:

```python
"""
Run a fixed benchmark suite on the requested device and dump timing JSON.

Usage (run on each machine separately):
    uv run python scripts/run_device_benchmark.py --device cpu  --output bench_macbook_cpu.json
    uv run python scripts/run_device_benchmark.py --device mps  --output bench_macbook_mps.json
    uv run python scripts/run_device_benchmark.py --device cuda --output bench_a10_cuda.json
"""
```

Benchmark suite — three problems, each run 3 times (median reported):

1. **OU (small)**: bundled `ornstein.npz` (n=20001, m=10, 1-D),
   `RQKernel + ExpConstKernel`, max_iter=5.
2. **OU (medium)**: same but with m=30 inducing points and concatenate the
   trajectory 4× (n≈80000) — exercises the bigger m and longer n that GPUs
   actually benefit from.
3. **Lorenz (multivariate)**: stochastic Lorenz '63 (n=30000, m=30, 3
   components fit independently via `MultiSDEVI`), max_iter=4 — the
   embarrassingly-parallel-across-components case.

Each run records:

```json
{
  "device": "cuda",
  "dtype": "float64",
  "torch_version": "2.11.0",
  "machine": {"hostname": "...", "platform": "Linux", "cpu_count": ...},
  "results": [
    {"problem": "ou_small",  "wall_clock_s": [4.81, 4.79, 4.83], "final_L": 36698.91, "iterations": 5},
    {"problem": "ou_medium", "wall_clock_s": [...], "final_L": ..., "iterations": ...},
    {"problem": "lorenz",    "wall_clock_s": [...], "final_L_per_component": [...], "iterations_per_component": [...]}
  ]
}
```

Strict reproducibility: each run uses the same `torch.manual_seed(0)`, fresh
kernels/inducing points, identical hyperparameter init.

### Phase 5 — R baseline collection (≈ 1 h, on the A10 machine)

`scripts/run_r_benchmark.R`:

```r
# Times the original voila::sde_vi on the same OU and DO datasets and dumps
# wall-clock to JSON for collation with the Python device benchmarks.
#
# Setup (one-time, on Linux instance):
#   sudo apt install r-base r-base-dev libarmadillo-dev
#   R -e 'install.packages(c("Rcpp", "RcppArmadillo", "yuima", "jsonlite"))'
#   R CMD INSTALL voila/
#
# Run:
#   Rscript scripts/run_r_benchmark.R --output bench_r_a10.json
```

Same problem sizes as the Python benchmark (OU small, medium). DO and Lorenz
require multivariate plumbing that voila R doesn't expose neatly — skip those
on the R side and document why. The OU small/medium cases are enough to
calibrate the absolute scale.

If R install fails on the A10 box (e.g. no apt access), fall back to: run R on
the MacBook CPU and just report that one number; document the absolute scale
difference between Mac CPU and A10 CPU separately so we can normalize.

### Phase 6 — Comparison notebook (≈ 30 min)

`notebooks/07_device_comparison.ipynb` loads all JSON outputs from
`bench_*.json` in the repo root, produces:

- **Bar chart per problem**: wall-clock time on R-Mac, Py-Mac-CPU, Py-Mac-MPS,
  Py-A10-CUDA.
- **Speedup table**: each Python device vs R baseline, vs Mac CPU baseline.
- **Final-L table**: for the OU small case, every device should reproduce the
  R anchor (36698.475) within its tolerance band — visualizes that going to
  MPS/float32 trades a tiny bit of accuracy for speed.

The notebook is *runnable on any machine* once the JSONs are collected — it's
just a loader + plotter, no new fits.

## Concrete file inventory

| Action | File | Notes |
|--|--|--|
| **Modify** | `src/voila_gp/kernels.py`         | device/dtype kwargs |
| **Modify** | `src/voila_gp/sparse_gp.py`       | use cholesky_inverse_compat |
| **Modify** | `src/voila_gp/elbo.py`            | verify device-clean |
| **Modify** | `src/voila_gp/posterior.py`       | use cholesky_inverse_compat |
| **Modify** | `src/voila_gp/inference.py`       | _State carries device/dtype; SDEVI accepts them; SDEVIResult records them |
| **Modify** | `src/voila_gp/multivariate.py`    | thread through MultiSDEVI |
| **Modify** | `src/voila_gp/forecast.py`        | simulate_forward device-aware + cache K_mm_inv |
| **Modify** | `src/voila_gp/prediction.py`      | use ndtri_compat |
| **Modify** | `src/voila_gp/diagnostics.py`     | accept inputs from any device |
| **Add**    | `src/voila_gp/_compat.py`         | cholesky_inverse_compat, ndtri_compat |
| **Add**    | `tests/unit/test_compat.py`       | shim sanity checks |
| **Add**    | `tests/unit/test_devices.py`      | device-parametrized tests |
| **Add**    | `scripts/run_device_benchmark.py` | timing harness, dumps JSON |
| **Add**    | `scripts/run_r_benchmark.R`       | R baseline timing |
| **Add**    | `scripts/build_device_comparison_notebook.py` | JSON-loading notebook builder |
| **Add**    | `notebooks/07_device_comparison.ipynb` | collated charts + tables |
| **Update** | `README.md`                       | new device-selection section |
| **Update** | `CLAUDE.md`                       | architecture note about device threading + scipy bridge |

Existing **70 + 15 = 85 tests** must continue to pass on CPU/float64 with no
relaxation. The new device tests run on CPU always and skip cleanly when the
GPU isn't available.

## Workflow on the A10 instance

Suggested order once you're on the A10:

```bash
# 1. Pull the repo (with this plan file) and the existing fixtures
git clone <repo>; cd voila-gpytorch
uv sync

# 2. Implement Phases 1–3 (device threading + compat + tests). Run:
uv run pytest                                   # 85 + new device tests pass
uv run ruff check src tests scripts && uv run pyright src

# 3. Collect Python benchmarks on the A10 + Mac
#    (Mac runs:)   uv run python scripts/run_device_benchmark.py --device cpu  --output bench_mac_cpu.json
#    (Mac runs:)   uv run python scripts/run_device_benchmark.py --device mps  --output bench_mac_mps.json
#    (A10 runs:)   uv run python scripts/run_device_benchmark.py --device cuda --output bench_a10_cuda.json

# 4. Set up R on the A10 (or Mac) and run the R baseline
sudo apt install r-base r-base-dev libarmadillo-dev
R -e 'install.packages(c("Rcpp", "RcppArmadillo", "yuima", "jsonlite"))'
R CMD INSTALL voila/
Rscript scripts/run_r_benchmark.R --output bench_r.json

# 5. Collate
uv run python scripts/build_device_comparison_notebook.py
uv run jupyter execute --inplace notebooks/07_device_comparison.ipynb
```

Each JSON is small (a few KB), so they can be checked into the repo or just
shared between machines via scp / Drive.

## Expected results (predictions)

These are predictions to validate against once the benchmarks run. If reality
diverges substantially we should investigate before publishing.

| Problem | Mac CPU (FP64) | Mac MPS (FP32) | A10 CUDA (FP64) | R (Mac CPU) |
|--|--|--|--|--|
| OU small (n=20k, m=10) | ~5 s | ~5–8 s* | ~5–10 s* | ~2-4 s |
| OU medium (n=80k, m=30) | ~30 s | ~10-20 s | **~3-8 s** | ~30-60 s |
| Lorenz (3-comp, n=30k, m=30) | ~60 s | ~30-60 s | **~10-20 s** | n/a |

\* The small problem is dominated by L-BFGS-B + Python overhead; GPU might
actually be slower due to kernel-launch latency. This is expected.

The headline narrative for users:

- **Small / 1-D problems**: any device is fine; CPU is often fastest because
  L-BFGS-B doesn't benefit from GPU and overhead dominates.
- **Medium / multivariate**: A10/CUDA gives ~5-10× over CPU; MPS gives ~2-3×
  if you can tolerate float32.
- **R parity**: every Python device hits the OU anchor within its documented
  tolerance band (CPU/CUDA: ±0.5 ELBO; MPS: ±50 ELBO due to float32, but the
  recovered drift function is unaffected).

## Verification (whole-feature exit criteria)

A reviewer pulling main should be able to:

1. `uv sync && uv run pytest` — 85 baseline + device tests all green.
2. `uv run pytest -m regression` — OU/DO/multivariate parity tests pass on
   CPU; device tests pass on whatever GPU the box has.
3. `uv run python scripts/run_device_benchmark.py --device cpu --output _bench.json`
   completes in <2 min.
4. Open `notebooks/07_device_comparison.ipynb` and see populated charts (or
   placeholders if some JSONs aren't present, with clear "missing data" notes).
5. README has a "Choosing a device" section with a one-paragraph
   recommendation per device class.

## Risks and mitigations

- **MPS PyTorch ops missing**: the compat shim handles known gaps, but PyTorch
  versions vary. Mitigation: each shim has a CPU fallback path (run that op on
  CPU, push result back). Acceptable if it's only `cholesky_inverse` once per
  fit (tiny cost).
- **MPS numerical instability with float32**: K_mm Cholesky on tight kernels
  may fail PD with float32. Mitigation: jitter `epsilon` is already a kernel
  parameter; document that MPS users should pass `epsilon=1e-4` (vs the
  `1e-5/1e-6` we use on CPU) for stability. Add a smoke test that exercises
  this.
- **CUDA kernel launch latency dominating small problems**: this is just
  reality, not a bug. Document it clearly so users don't expect 10× speedups
  on n=1000 problems.
- **R baseline reproducibility**: `voila::sde_vi` uses Fortran L-BFGS-B with a
  default convergence tolerance that may stop earlier or later than scipy's,
  inflating or deflating its wall-clock unfairly. Mitigation: align
  `relTol` and `maxIterations` between Python and R as closely as possible;
  document the residual gap in the comparison notebook.

## Notes for the future you / future me

- The `forecast.simulate_forward` cache for `K_mm_inv` is a ~free perf win
  even on CPU. Worth doing as part of Phase 1 because it makes the GPU
  benchmark numbers honest (otherwise per-step kernel-rebuild dominates).
- If after Phase 4 the A10/CUDA numbers are disappointing, the next move is
  **Option 3 from our discussion**: an Adam-based fit path that runs
  GPU-native and supports batched fitting across many time series. That's the
  real GPU win story; the current plan just makes single fits device-portable.
- If the user wants the *L-BFGS-B itself* to run on GPU (Option 4 from our
  discussion), that's a separate ~300-line pure-torch implementation. Skip
  unless profiling on the A10 says boundary crossings are dominating, which
  we don't expect.
