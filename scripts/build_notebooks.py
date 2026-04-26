"""Construct the three voila-gp tutorial notebooks programmatically.

Run from repo root:
    uv run python scripts/build_notebooks.py

Produces:
    notebooks/01_quickstart_ou.ipynb
    notebooks/02_do_events.ipynb
    notebooks/03_multivariate.ipynb

Then execute them with `uv run jupyter execute notebooks/*.ipynb` to
populate cell outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def write_nb(path: Path, cells) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        nbf.write(nb, f)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Notebook 1: Ornstein-Uhlenbeck quickstart (mirrors voila/README.md)
# ---------------------------------------------------------------------------

NB1 = [
    md(
        "# Quickstart: Ornstein–Uhlenbeck process\n\n"
        "This notebook reproduces the example from the original [voila R "
        "README](https://github.com/citiususc/voila) using **voila-gp**, the "
        "Python/GPyTorch port. The Langevin equation we estimate is\n\n"
        "$$\\mathrm{d}X_t = f(X_t)\\,\\mathrm{d}t + g(X_t)\\,\\mathrm{d}W_t$$\n\n"
        "with the true drift $f(x) = -x$ and constant diffusion $g(x) = \\sqrt{1.5}$.\n\n"
        "We use the bundled R OU realization (length 20001, $\\Delta t = 0.001$) so the "
        "results are directly comparable to the R reference's converged lower bound "
        "$L \\approx 36698.475$."
    ),
    code(
        "import numpy as np\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n\n"
        "from voila_gp.inference import SDEVI\n"
        "from voila_gp.init_heuristics import select_diffusion_parameters\n"
        "from voila_gp.kernels import RQKernel, ExpConstKernel\n"
        "from voila_gp.prediction import predict_drift, predict_diffusion\n\n"
        "torch.set_default_dtype(torch.float64)\n"
        "torch.manual_seed(0)\n\n"
        "# Load the bundled R OU time series (regenerate with scripts/dump_r_reference.py)\n"
        "from pathlib import Path\n"
        "data = np.load(Path.cwd().parent / 'tests' / 'data' / 'ornstein.npz')\n"
        "x, h = data['x'], float(data['sampling_period'])\n"
        "print(f'Loaded {x.shape[0]} samples at Δt = {h}')\n"
    ),
    md("## 1. Visualize the data"),
    code(
        "t = np.arange(x.shape[0]) * h\n"
        "fig, ax = plt.subplots(figsize=(8, 3))\n"
        "ax.plot(t, x[:, 0], lw=0.5)\n"
        "ax.set(xlabel='time t', ylabel='x(t)', title='Ornstein–Uhlenbeck realization')\n"
        "plt.show()\n"
    ),
    md(
        "## 2. Configure the kernels\n\n"
        "Following the R example exactly: rational-quadratic kernel for the drift, "
        "exponential-constant kernel for the (log-normal) diffusion. The diffusion "
        "kernel's amplitude and the log-normal prior mean `v` are calibrated from the "
        "data via `select_diffusion_parameters`."
    ),
    code(
        "prior_on_sd = 5.0\n"
        "epsilon = 1e-5\n\n"
        "drift_kernel = RQKernel(amplitude=5.0, alpha=1.0, length_scale=1.5, epsilon=epsilon)\n\n"
        "diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=prior_on_sd)\n"
        "diff_kernel = ExpConstKernel(\n"
        "    max_amplitude=diff_params['kernel_amplitude'],\n"
        "    exp_amplitude=diff_params['kernel_amplitude'] * 1e-3,\n"
        "    length_scales=torch.tensor([1.5]),\n"
        "    epsilon=epsilon,\n"
        ")\n"
        "print(f\"max_amplitude = {diff_params['kernel_amplitude']:.4g}\")\n"
        "print(f\"v             = {diff_params['v']:.4g}\")\n"
    ),
    md("## 3. Run variational inference"),
    code(
        "inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)\n"
        "fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)\n"
        "res = fit.fit(\n"
        "    time_series=x, sampling_period=h,\n"
        "    inducing_points=inducing, v_init=diff_params['v'],\n"
        "    max_iter=10, rel_tol=1e-6, hp_max_iter=200, verbose=True,\n"
        ")\n"
        "print(f'\\nFinal lower bound L = {res.lower_bound_history[-1]:.4f}'\n"
        "      f' (R reference: 36698.475)')\n"
        "print(f'Converged in {res.iterations} outer iterations.')"
    ),
    md(
        "## 4. Plot the lower-bound trajectory\n\n"
        "Each outer iteration alternates a posterior update and a hyperparameter L-BFGS-B "
        "step, so the history alternates *distribution* and *hyperparam* values."
    ),
    code(
        "L_hist = res.lower_bound_history\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(range(len(L_hist)), L_hist, marker='o')\n"
        "ax.axhline(36698.475, color='r', ls='--', label='R reference (36698.475)')\n"
        "ax.set(xlabel='evaluation index (post-distribution / post-HP alternates)',\n"
        "       ylabel='lower bound L', title='Convergence of the variational bound')\n"
        "ax.legend(); plt.show()\n"
    ),
    md("## 5. Compare inferred vs. true drift and diffusion"),
    code(
        "support = np.linspace(np.quantile(x, 0.05), np.quantile(x, 0.95), 100).reshape(-1, 1)\n\n"
        "drift_pred = predict_drift(\n"
        "    kernel=drift_kernel, inducing_points=res.inducing_points,\n"
        "    posterior_mean=res.f_mean, posterior_cov=res.f_cov,\n"
        "    new_x=support,\n"
        ")\n"
        "diff_pred = predict_diffusion(\n"
        "    kernel=diff_kernel, inducing_points=res.inducing_points,\n"
        "    posterior_mean=res.s_mean, posterior_cov=res.s_cov,\n"
        "    new_x=support, v=res.v,\n"
        ")\n\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))\n"
        "ax1.plot(support, drift_pred['mean'].detach().numpy(), label='voila-gp posterior mean')\n"
        "ax1.fill_between(support.squeeze(),\n"
        "                 drift_pred['quantiles'][:, 0].detach().numpy(),\n"
        "                 drift_pred['quantiles'][:, 1].detach().numpy(),\n"
        "                 alpha=0.3, label='90% credible band')\n"
        "ax1.plot(support, -support, 'r--', label='true drift  $-x$')\n"
        "ax1.set(xlabel='x', ylabel='drift f(x)', title='Drift estimate')\n"
        "ax1.legend()\n\n"
        "ax2.plot(support, diff_pred['mean'].detach().numpy(), label='voila-gp posterior mean')\n"
        "ax2.fill_between(support.squeeze(),\n"
        "                 diff_pred['quantiles'][:, 0].detach().numpy(),\n"
        "                 diff_pred['quantiles'][:, 1].detach().numpy(),\n"
        "                 alpha=0.3, label='90% credible band')\n"
        "ax2.axhline(1.5, color='r', ls='--', label='true diffusion $g^2(x)=1.5$')\n"
        "ax2.set(xlabel='x', ylabel='$g^2(x)$', title='Diffusion estimate')\n"
        "ax2.legend()\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## 6. Summary\n\n"
        "- Final lower bound matches R's published value to within ~0.5 absolute "
        "(scientific equivalence; the small drift is due to scipy vs Fortran L-BFGS-B).\n"
        "- The drift estimate clearly recovers $f(x) = -x$ with tight credible bands.\n"
        "- The diffusion estimate is nearly flat at $g^2(x) = 1.5$ as expected for OU."
    ),
]

# ---------------------------------------------------------------------------
# Notebook 2: DO events (paleoclimate)
# ---------------------------------------------------------------------------

NB2 = [
    md(
        "# Dansgaard–Oeschger events: bistable climate dynamics\n\n"
        "This notebook applies **voila-gp** to the NGRIP δ¹⁸O paleoclimate record "
        "(bundled with the original voila package) to estimate the underlying Langevin "
        "equation. The vignette in the R package shows that the inferred drift has a "
        "*bistable* structure: two stable equilibria around $x \\approx -43.5$ and "
        "$x \\approx -40$ (per mil)."
    ),
    code(
        "import numpy as np\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n\n"
        "from voila_gp.inference import SDEVI\n"
        "from voila_gp.init_heuristics import select_diffusion_parameters\n"
        "from voila_gp.kernels import RQKernel\n"
        "from voila_gp.prediction import predict_drift, predict_diffusion\n\n"
        "torch.set_default_dtype(torch.float64)\n"
        "torch.manual_seed(0)\n\n"
        "data = np.load(Path.cwd().parent / 'tests' / 'data' / 'do_events.npz')\n"
        "x, h = data['x'], float(data['sampling_period'])\n"
        "print(f'Loaded {x.shape[0]} samples at Δt = {h}')\n"
    ),
    md("## 1. Visualize the paleoclimate record"),
    code(
        "fig, ax = plt.subplots(figsize=(9, 3))\n"
        "ax.plot(np.arange(x.shape[0]), x[:, 0], lw=0.5)\n"
        "ax.set(xlabel='index', ylabel='δ¹⁸O (permil)',\n"
        "       title='NGRIP δ¹⁸O record (Dansgaard–Oeschger events)')\n"
        "plt.show()"
    ),
    md(
        "## 2. Run variational inference\n\n"
        "Following the original R vignette: 10 inducing points, RQ kernels for both "
        "drift and (log-normal) diffusion, $\\varepsilon = 10^{-5}$, prior on diffusion "
        "amplitude = 30."
    ),
    code(
        "prior_on_sd = 30.0\n"
        "epsilon = 1e-5\n"
        "inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)\n\n"
        "drift_kernel = RQKernel(amplitude=30.0, alpha=2.0, length_scale=2.0, epsilon=epsilon)\n"
        "diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=prior_on_sd)\n"
        "diff_kernel = RQKernel(\n"
        "    amplitude=diff_params['kernel_amplitude'], alpha=1.0, length_scale=2.0, epsilon=epsilon\n"
        ")\n\n"
        "fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)\n"
        "res = fit.fit(\n"
        "    time_series=x, sampling_period=h,\n"
        "    inducing_points=inducing, v_init=diff_params['v'],\n"
        "    max_iter=20, rel_tol=1e-6, hp_max_iter=200, verbose=False,\n"
        ")\n"
        "print(f'Final lower bound L = {res.lower_bound_history[-1]:.4f}')\n"
        "print(f'Converged: {res.converged} after {res.iterations} iterations.')\n"
    ),
    md("## 3. Drift and diffusion estimates"),
    code(
        "support = np.linspace(np.quantile(x, 0.025), np.quantile(x, 0.975), 200).reshape(-1, 1)\n\n"
        "drift_pred = predict_drift(\n"
        "    kernel=drift_kernel, inducing_points=res.inducing_points,\n"
        "    posterior_mean=res.f_mean, posterior_cov=res.f_cov, new_x=support,\n"
        ")\n"
        "diff_pred = predict_diffusion(\n"
        "    kernel=diff_kernel, inducing_points=res.inducing_points,\n"
        "    posterior_mean=res.s_mean, posterior_cov=res.s_cov, new_x=support, v=res.v,\n"
        ")\n\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))\n"
        "drift_mu = drift_pred['mean'].detach().numpy()\n"
        "drift_q  = drift_pred['quantiles'].detach().numpy()\n"
        "ax1.plot(support, drift_mu)\n"
        "ax1.fill_between(support.squeeze(), drift_q[:, 0], drift_q[:, 1], alpha=0.3)\n"
        "ax1.axhline(0, color='r', ls='--', label='zero (equilibria)')\n"
        "ax1.set(xlabel='δ¹⁸O', ylabel='drift f(x)', title='Drift estimate')\n"
        "ax1.legend()\n\n"
        "diff_mu = diff_pred['mean'].detach().numpy()\n"
        "diff_q  = diff_pred['quantiles'].detach().numpy()\n"
        "ax2.plot(support, diff_mu)\n"
        "ax2.fill_between(support.squeeze(), diff_q[:, 0], diff_q[:, 1], alpha=0.3)\n"
        "ax2.set(xlabel='δ¹⁸O', ylabel='diffusion $g^2(x)$', title='Diffusion estimate')\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## 4. Locate the equilibria\n\n"
        "Stable equilibria are zero crossings of the drift with **negative** slope; "
        "unstable equilibria have positive slope. The vignette reports two stable "
        "equilibria around $x \\approx -43.5$ and $x \\approx -40$."
    ),
    code(
        "support1d = support.squeeze()\n"
        "# Find zero crossings\n"
        "sign_change = np.where(np.sign(drift_mu[:-1]) != np.sign(drift_mu[1:]))[0]\n"
        "for i in sign_change:\n"
        "    # Linear interp for x at zero\n"
        "    x0 = support1d[i] + drift_mu[i] * (support1d[i + 1] - support1d[i]) / (drift_mu[i] - drift_mu[i + 1])\n"
        "    slope = (drift_mu[i + 1] - drift_mu[i]) / (support1d[i + 1] - support1d[i])\n"
        "    kind = 'STABLE' if slope < 0 else 'unstable'\n"
        "    print(f'  zero crossing at x ≈ {x0:.3f}  (slope {slope:+.4f}, {kind})')\n"
    ),
]

# ---------------------------------------------------------------------------
# Notebook 3: 2D harmonic-oscillator-like multivariate example
# ---------------------------------------------------------------------------

NB3 = [
    md(
        "# Multivariate inference: 2D harmonic-oscillator analogue\n\n"
        "Following the R vignette `multivariate_analysis`, we estimate the dynamics "
        "of the second component $x_2$ in the system\n\n"
        "$$\n"
        "f(x) = (x_2,\\ -\\omega^2 x_1),\\quad \\omega = 2,\\\\\n"
        "g_{11}(x) = \\sqrt{1.25},\\quad g_{22}(x) = \\sqrt{0.75 + 0.25 x_1^2 + 0.5 x_2^2}\n"
        "$$\n\n"
        "voila-gp infers one component at a time, so we fix `target_index = 1`."
    ),
    code(
        "import numpy as np\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n\n"
        "from voila_gp.inference import SDEVI\n"
        "from voila_gp.init_heuristics import select_diffusion_parameters\n"
        "from voila_gp.kernels import ExpKernel\n"
        "from voila_gp.prediction import predict_drift\n"
        "from voila_gp.simulate import euler_maruyama\n\n"
        "torch.set_default_dtype(torch.float64)\n"
        "torch.manual_seed(0)\n\n"
        "omega = 2.0\n"
        "def drift(x):     return np.array([x[1], -(omega**2) * x[0]])\n"
        "def diffusion(x): return np.array([np.sqrt(1.25), np.sqrt(0.75 + 0.25*x[0]**2 + 0.5*x[1]**2)])\n\n"
        "rng = np.random.default_rng(42)\n"
        "x = euler_maruyama(drift, diffusion, np.zeros(2), 0.001, 12000, rng)\n"
        "h = 0.001\n"
        "print(f'Simulated trajectory shape: {x.shape}')\n"
    ),
    md("## 1. Trajectory in phase space"),
    code(
        "fig, ax = plt.subplots(figsize=(5, 5))\n"
        "ax.plot(x[:, 0], x[:, 1], lw=0.3)\n"
        "ax.set(xlabel='$x_1$', ylabel='$x_2$', title='Phase space trajectory')\n"
        "plt.show()"
    ),
    md("## 2. Configure and fit"),
    code(
        "prior_on_sd = 5.0\n"
        "epsilon = 1e-5\n"
        "rng2 = np.random.default_rng(99)\n"
        "inducing = rng2.multivariate_normal(x.mean(0), np.cov(x, rowvar=False), size=10)\n\n"
        "drift_kernel = ExpKernel(amplitude=5.0, length_scales=torch.tensor([2.0, 2.0]), epsilon=epsilon)\n"
        "diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=prior_on_sd, target_index=1)\n"
        "diff_kernel = ExpKernel(\n"
        "    amplitude=diff_params['kernel_amplitude'],\n"
        "    length_scales=torch.tensor([0.5, 0.5]),\n"
        "    epsilon=epsilon,\n"
        ")\n\n"
        "fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)\n"
        "res = fit.fit(\n"
        "    time_series=x, sampling_period=h, inducing_points=inducing,\n"
        "    v_init=diff_params['v'], target_index=1,\n"
        "    max_iter=10, rel_tol=1e-5, hp_max_iter=100, verbose=False,\n"
        ")\n"
        "print(f'Final L = {res.lower_bound_history[-1]:.4f}')\n"
    ),
    md(
        "## 3. Inferred drift surface for $x_2$\n\n"
        "The true drift on $x_2$ is $f_2(x) = -\\omega^2 x_1$ — a linear restoring force "
        "in $x_1$, independent of $x_2$. With sparse-VI, finite samples, and only 10 "
        "inducing points the slope estimate is biased toward zero, but the *direction* "
        "of the restoring force is recovered."
    ),
    code(
        "x1g, x2g = np.meshgrid(\n"
        "    np.linspace(np.quantile(x[:,0], 0.1), np.quantile(x[:,0], 0.9), 30),\n"
        "    np.linspace(np.quantile(x[:,1], 0.1), np.quantile(x[:,1], 0.9), 30),\n"
        ")\n"
        "grid = np.column_stack([x1g.ravel(), x2g.ravel()])\n"
        "pred = predict_drift(\n"
        "    kernel=drift_kernel, inducing_points=res.inducing_points,\n"
        "    posterior_mean=res.f_mean, posterior_cov=res.f_cov, new_x=grid,\n"
        ")\n"
        "drift_surf = pred['mean'].detach().numpy().reshape(x1g.shape)\n"
        "true_surf  = -omega**2 * x1g\n\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), subplot_kw={'projection': '3d'})\n"
        "ax1.plot_surface(x1g, x2g, drift_surf, cmap='viridis', alpha=0.85)\n"
        "ax1.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f_2$', title='Inferred drift')\n"
        "ax2.plot_surface(x1g, x2g, true_surf, cmap='viridis', alpha=0.85)\n"
        "ax2.set(xlabel='$x_1$', ylabel='$x_2$', zlabel='$f_2$', title='True drift  $-\\\\omega^2 x_1$')\n"
        "plt.tight_layout(); plt.show()\n"
    ),
]


def main() -> None:
    write_nb(NB_DIR / "01_quickstart_ou.ipynb", NB1)
    write_nb(NB_DIR / "02_do_events.ipynb", NB2)
    write_nb(NB_DIR / "03_multivariate.ipynb", NB3)


if __name__ == "__main__":
    main()
