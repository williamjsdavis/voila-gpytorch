# voila-gpytorch

A Python/[GPyTorch](https://gpytorch.ai/) reimplementation of [`voila`](https://github.com/citiususc/voila), an R package for non-parametric estimation of Langevin equations (stochastic differential equations) from densely-observed time series.

## Background

The original `voila` package estimates the drift and diffusion terms of a Langevin equation

$$\mathrm{d}X_t = f(X_t)\,\mathrm{d}t + g(X_t)\,\mathrm{d}W_t$$

by modelling $f$ and $g$ as Gaussian processes and learning sparse approximations via variational inference with inducing variables. The method is described in:

> García, C.A., Otero, A., Félix, P., Presedo, J. & Márquez D.G. **Non-parametric Estimation of Stochastic Differential Equations with Sparse Gaussian Processes.** *Phys. Rev. E 96 (2017), 022104.* [Article](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.022104) · [Preprint](https://arxiv.org/abs/1704.04375)

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for environment management:

```bash
uv sync
uv run pytest                                   # full test suite
uv run jupyter execute --inplace notebooks/*.ipynb   # render tutorials
```

Python 3.12 is pinned via `.python-version`. All dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

## Quickstart

```python
import numpy as np
import torch
from voila_gp.inference import SDEVI
from voila_gp.init_heuristics import select_diffusion_parameters
from voila_gp.kernels import RQKernel, ExpConstKernel
from voila_gp.prediction import predict_drift, predict_diffusion

torch.set_default_dtype(torch.float64)

# 1. Load (or simulate) a time series; shape (n, d).
x = np.load("tests/data/ornstein.npz")["x"]
h = 0.001  # sampling period

# 2. Configure kernels (matches voila/README.md).
drift_kernel = RQKernel(amplitude=5.0, alpha=1.0, length_scale=1.5, epsilon=1e-5)
diff_params  = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=5.0)
diff_kernel  = ExpConstKernel(
    max_amplitude=diff_params["kernel_amplitude"],
    exp_amplitude=diff_params["kernel_amplitude"] * 1e-3,
    length_scales=torch.tensor([1.5]),
    epsilon=1e-5,
)

# 3. Run sparse variational inference.
inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)
fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)
res = fit.fit(
    time_series=x, sampling_period=h, inducing_points=inducing,
    v_init=diff_params["v"], max_iter=10, rel_tol=1e-6,
)
print(f"Final L = {res.lower_bound_history[-1]:.4f}")

# 4. Predict drift and diffusion at new points.
support = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
drift = predict_drift(
    kernel=drift_kernel, inducing_points=res.inducing_points,
    posterior_mean=res.f_mean, posterior_cov=res.f_cov, new_x=support,
)
diff = predict_diffusion(
    kernel=diff_kernel, inducing_points=res.inducing_points,
    posterior_mean=res.s_mean, posterior_cov=res.s_cov, new_x=support, v=res.v,
)
```

## Tutorials

Three Jupyter notebooks under `notebooks/` mirror the original R vignettes:

| Notebook | Topic |
|--|--|
| `01_quickstart_ou.ipynb` | Ornstein–Uhlenbeck — reproduces voila/README.md |
| `02_do_events.ipynb`     | Dansgaard–Oeschger paleoclimate events (NGRIP δ¹⁸O) |
| `03_multivariate.ipynb`  | 2-D harmonic-oscillator-analogue example |

Build them from the source script and execute with outputs:

```bash
uv run python scripts/build_notebooks.py
uv run jupyter execute --inplace notebooks/*.ipynb
```

## Validation against the R reference

Numerical anchors (taken from `voila/README.md`'s execution log on the bundled OU realization):

| Quantity | R reference | voila-gp | Δ |
|--|--|--|--|
| Initial L | -59222.9 | -59222.94 | 0.04 |
| Iter 1 distribution L | 36691.347 | 36691.347 | 0.0003 |
| Iter 2 distribution L | 36698.412 | 36698.527 | 0.115 |
| Iter 3 hyperparam L (R declares convergence) | 36698.475 | 36698.844 | 0.369 |
| Final L | 36698.475 | 36698.917 | 0.442 |

The tight match through the early iterations confirms algorithmic identity with voila. The small late-iteration drift (≈ +0.4) reflects scipy's L-BFGS-B running slightly more thoroughly than voila's Fortran solver before declaring convergence — both implementations are correct.

## Architecture

| Module | Role |
|--|--|
| `voila_gp.kernels`         | Five kernels ported from `voila/src/common_kernels.cpp`: `ExpKernel`, `RQKernel`, `ExpConstKernel`, `SumExpKernel`, `ClampedExpLinKernel`. |
| `voila_gp.sparse_gp`       | `K_mm`, `K_mm_inv`, `K_nm`, `A = K_nm K_mm⁻¹`, `Q_ii` — the sparse-GP intermediates. |
| `voila_gp.elbo`            | Variational lower bound, `E` and `ξ` vectors. |
| `voila_gp.posterior`       | Closed-form drift update; Laplace diffusion update with analytical Newton iteration. |
| `voila_gp.inference`       | `SDEVI.fit(...)` — outer alternating loop with scipy L-BFGS-B for hyperparameters. |
| `voila_gp.prediction`      | Posterior drift/diffusion at new points; log-normal back-transform. |
| `voila_gp.init_heuristics` | `select_diffusion_parameters` (verbatim port of the R helper). |
| `voila_gp.simulate`        | Lightweight Euler-Maruyama for tutorials/tests. |

The R `voila/` and `paper/` directories remain present locally as the canonical specification (gitignored). Run `scripts/dump_r_reference.py` to regenerate `tests/data/*.npz` fixtures from the bundled `.rda` files.

## License

GPL-3.0-or-later (matches the original voila R package).
