# Pickup list for the MacBook

This file captures what's left to do on the Apple Silicon MacBook to complete
the multi-device validation. The A10/Lambda side is already done (commits on
`main`); these steps gather the missing CPU + MPS data points and (optionally)
the R baseline.

## Prerequisites

```bash
git pull
uv sync
```

## 1. Sanity-check tests on the Mac

The device-threading work changed how every long-lived tensor is allocated. A
quick way to confirm nothing regressed on the original Mac CPU/float64 path:

```bash
uv run pytest                    # expect: 85 baseline + 13 compat + 6 device = 104 green
                                  # (the device tests on the Mac will include
                                  # cpu + mps parametrizations; cuda skips)
```

If anything fails, **stop here** and report the failure — it's a regression
introduced by Phase 1, not a Mac-specific quirk.

## 2. Collect the Mac CPU benchmark

```bash
uv run python scripts/run_device_benchmark.py --device cpu --output bench_mac_cpu.json
```

Three problems × 3 reps; expect 5–15 minutes wall-clock total on a recent
MacBook.

## 3. Collect the MPS benchmark

```bash
uv run python scripts/run_device_benchmark.py --device mps --output bench_mac_mps.json
```

**Watch for:**

- `cholesky_solve` PD failures on tight kernels under float32. If `lorenz`
  blows up, bump the kernel `epsilon` from `1e-4` → `1e-3` in
  `scripts/run_device_benchmark.py` (in both `drift_kernel_factory` and
  `diff_kernel_factory`) and re-run. Document the change.
- ELBO values that look "off" by tens of units relative to the CPU/CUDA
  numbers. That's expected — MPS is FP32. The drift correlation (in the
  Lorenz notebook) is the better fidelity check.

## 4. (Optional) R baseline

Only worth doing if you want a head-to-head against the original `voila`:

```bash
brew install r            # or: sudo apt install r-base r-base-dev libarmadillo-dev
R -e 'install.packages(c("Rcpp", "RcppArmadillo", "yuima", "jsonlite"))'
R CMD INSTALL voila/      # builds the vendored R package

Rscript scripts/run_r_benchmark.R --output bench_r.json
```

Allow ~30–60 minutes for the install path. The R script times only `ou_small`
and `ou_medium` (Lorenz/multivariate aren't exposed cleanly by `voila`).

## 5. Re-render the comparison notebook

Once the JSONs are in place (either committed locally on the Mac or scp'd from
the Lambda box), regenerate the chart:

```bash
uv run python scripts/build_device_comparison_notebook.py
uv run jupyter execute --inplace notebooks/07_device_comparison.ipynb
```

The notebook is just a JSON loader + plotter — running it does not re-run any
fits. It will silently skip any device whose JSON is absent.

## 6. Commit and push

```bash
git add bench_mac_cpu.json bench_mac_mps.json notebooks/07_device_comparison.ipynb
# (and bench_r.json if you ran the R baseline)
git commit -m "Add Mac CPU + MPS benchmark results and re-rendered comparison notebook"
git push
```

## Open questions left for the Mac session

- **MPS PyTorch op coverage:** the compat shim covers `cholesky_inverse` and
  `ndtri`, but PyTorch 2.11 may have other gaps. If `pytest` on the Mac trips
  a "not implemented for MPS" error, add the missing op to
  `src/voila_gp/_compat.py` following the same pattern.
- **Float32 stability on tight kernels:** the `epsilon=1e-5` we use on
  CPU/CUDA may not be enough on MPS. Document any bumps.
