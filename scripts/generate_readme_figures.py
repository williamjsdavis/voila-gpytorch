"""Render the static images used by the top-level README.

Three figures, all written under `static/`:

  - `ou_estimation.png`        — drift + diffusion posteriors on the bundled
                                 Ornstein–Uhlenbeck realization. The "core
                                 algorithm in one picture" image.
  - `lorenz_reconstruction.png` — true vs voila-gp-reconstructed Lorenz '63
                                 butterfly attractor. The "this also handles
                                 3-D chaos" showpiece.
  - `device_comparison.png`     — wall-clock bar chart across CPU / MPS /
                                 CUDA / R reference, loaded from
                                 `benchmarks/bench_*.json`.

Run from repo root:
    uv run python scripts/generate_readme_figures.py

Re-run after any benchmark JSON changes; the comparison plot will update.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "static"
DATA = REPO_ROOT / "tests" / "data"
BENCH = REPO_ROOT / "benchmarks"


def _ou_estimation_figure() -> None:
    """OU drift+diffusion posterior vs ground truth."""
    from voila_gp.inference import SDEVI
    from voila_gp.init_heuristics import select_diffusion_parameters
    from voila_gp.kernels import ExpConstKernel, RQKernel
    from voila_gp.prediction import predict_diffusion, predict_drift

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    npz = np.load(DATA / "ornstein.npz")
    x, h = npz["x"], float(npz["sampling_period"])
    inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)

    drift_kernel = RQKernel(amplitude=5.0, alpha=1.0, length_scale=1.5, epsilon=1e-5)
    diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=5.0)
    diff_kernel = ExpConstKernel(
        max_amplitude=diff_params["kernel_amplitude"],
        exp_amplitude=diff_params["kernel_amplitude"] * 1e-3,
        length_scales=torch.tensor([1.5]),
        epsilon=1e-5,
    )
    fit = SDEVI(drift_kernel=drift_kernel, diff_kernel=diff_kernel)
    res = fit.fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        v_init=diff_params["v"], max_iter=10, rel_tol=1e-6, hp_max_iter=200,
    )

    support = np.linspace(np.quantile(x, 0.05), np.quantile(x, 0.95), 100).reshape(-1, 1)
    drift_pred = predict_drift(
        kernel=drift_kernel, inducing_points=res.inducing_points,
        posterior_mean=res.f_mean, posterior_cov=res.f_cov, new_x=support,
    )
    diff_pred = predict_diffusion(
        kernel=diff_kernel, inducing_points=res.inducing_points,
        posterior_mean=res.s_mean, posterior_cov=res.s_cov, new_x=support, v=res.v,
    )
    drift_mu = drift_pred["mean"].detach().numpy()
    drift_q = drift_pred["quantiles"].detach().numpy()
    diff_mu = diff_pred["mean"].detach().numpy()
    diff_q = diff_pred["quantiles"].detach().numpy()
    s = support.squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
    ax1.fill_between(s, drift_q[:, 0], drift_q[:, 1], alpha=0.3, color="steelblue",
                     label="90% credible band")
    ax1.plot(s, drift_mu, color="steelblue", lw=2, label="voila-gp posterior mean")
    ax1.plot(s, -s, "r--", lw=1.4, label="true drift  $f(x) = -x$")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set(xlabel="x", ylabel="drift  $f(x)$", title="Drift estimate")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.fill_between(s, diff_q[:, 0], diff_q[:, 1], alpha=0.3, color="darkorange",
                     label="90% credible band")
    ax2.plot(s, diff_mu, color="darkorange", lw=2, label="voila-gp posterior mean")
    ax2.axhline(1.5, color="r", ls="--", lw=1.4, label="true diffusion  $g^2(x) = 1.5$")
    ax2.set(xlabel="x", ylabel="$g^2(x)$", title="Diffusion estimate")
    ax2.set_ylim(0.5, 2.5)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("voila-gp on a 1-D Ornstein–Uhlenbeck process", fontsize=12, y=1.02)
    plt.tight_layout()
    out = STATIC / "ou_estimation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def _lorenz_reconstruction_figure() -> None:
    """Stochastic Lorenz '63 attractor — truth vs voila-gp-fit reconstruction."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers 3D projection

    from voila_gp.forecast import simulate_forward
    from voila_gp.kernels import ExpKernel
    from voila_gp.multivariate import MultiSDEVI
    from voila_gp.simulate import euler_maruyama

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def drift(x: np.ndarray) -> np.ndarray:
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])

    def diffusion(_x: np.ndarray) -> np.ndarray:
        return np.full(3, 1.0)

    dt = 0.005
    n_steps = 12_000
    rng = np.random.default_rng(0)
    ts = euler_maruyama(drift, diffusion, np.array([1.0, 1.0, 1.0]), dt, n_steps, rng)
    ts = ts[1000:]  # drop transient

    rng2 = np.random.default_rng(1)
    inducing = rng2.multivariate_normal(ts.mean(0), np.cov(ts, rowvar=False), size=25)
    fit = MultiSDEVI(
        drift_kernel_factory=lambda *_: ExpKernel(
            amplitude=200.0, length_scales=torch.tensor([6.0, 6.0, 6.0]), epsilon=1e-4,
        ),
        diff_kernel_factory=lambda *_: ExpKernel(
            amplitude=2.0, length_scales=torch.tensor([6.0, 6.0, 6.0]), epsilon=1e-4,
        ),
        prior_on_sd=2.0,
    )
    res = fit.fit(
        time_series=ts, sampling_period=dt, inducing_points=inducing,
        max_iter=4, hp_max_iter=60, verbose=False,
    )
    recon = simulate_forward(
        res, x0=ts[0].copy(), n_steps=20_000, n_ensembles=1, dt=dt, seed=7,
    ).trajectories[0].numpy()

    fig = plt.figure(figsize=(11, 4.5), dpi=110)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(ts[:, 0], ts[:, 1], ts[:, 2], lw=0.25, color="steelblue", alpha=0.75)
    ax1.set(xlabel="x", ylabel="y", zlabel="z", title="Truth (noisy Lorenz '63)")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(recon[:, 0], recon[:, 1], recon[:, 2], lw=0.25, color="darkorange", alpha=0.75)
    ax2.set(xlabel="x", ylabel="y", zlabel="z", title="voila-gp reconstruction")
    fig.suptitle(
        "Stochastic Lorenz '63 attractor reconstructed from a single noisy trajectory",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    out = STATIC / "lorenz_reconstruction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def _device_comparison_figure() -> None:
    """Wall-clock bar chart across CPU / MPS / CUDA / R."""
    bench_files = sorted(BENCH.glob("bench_*.json"))
    if not bench_files:
        print(f"WARNING: no bench files in {BENCH}; skipping device_comparison.png")
        return

    rows = []
    for p in bench_files:
        payload = json.loads(p.read_text())
        label = p.stem.removeprefix("bench_").replace("_", " ").upper()
        for r in payload.get("results", []):
            rows.append({
                "label": label,
                "problem": r["problem"],
                "median_s": float(r["median_s"]),
            })
    if not rows:
        print("WARNING: bench files contained no results")
        return

    problems = sorted({r["problem"] for r in rows}, key=lambda p: ["ou_small", "ou_medium", "lorenz"].index(p))
    fig, axes = plt.subplots(1, len(problems), figsize=(4.0 * len(problems), 3.6), dpi=110)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems, strict=True):
        sub = sorted(
            [r for r in rows if r["problem"] == problem], key=lambda r: r["median_s"]
        )
        labels = [r["label"] for r in sub]
        times = [r["median_s"] for r in sub]
        # Color GPU / accelerated devices distinctly
        colors = []
        for lab in labels:
            up = lab.upper()
            if "CUDA" in up:
                colors.append("#2ca02c")  # green for CUDA
            elif "MPS" in up:
                colors.append("#1f77b4")  # blue for MPS
            elif up.strip() == "R" or up.startswith("R "):
                colors.append("#7f7f7f")  # gray for R baseline
            else:
                colors.append("#ff7f0e")  # orange for CPU
        bars = ax.barh(labels, times, color=colors)
        ax.set(xlabel="wall-clock (s)", title=problem)
        ax.set_xlim(0, max(times) * 1.18)
        for bar, t in zip(bars, times, strict=True):
            ax.text(t, bar.get_y() + bar.get_height() / 2, f" {t:.2f}s",
                    va="center", fontsize=9)
    fig.suptitle("Median wall-clock per device (3 runs each)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = STATIC / "device_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    _ou_estimation_figure()
    _device_comparison_figure()
    _lorenz_reconstruction_figure()


if __name__ == "__main__":
    main()
