"""Time voila-gp end-to-end on a fixed problem suite and dump JSON.

Run separately on every machine you want to compare:

    uv run python scripts/run_device_benchmark.py --device cpu  --output benchmarks/bench_a10_cpu.json
    uv run python scripts/run_device_benchmark.py --device cuda --output benchmarks/bench_a10_cuda.json
    uv run python scripts/run_device_benchmark.py --device cpu  --output benchmarks/bench_mac_cpu.json   # on Mac
    uv run python scripts/run_device_benchmark.py --device mps  --output benchmarks/bench_mac_mps.json   # on Mac

Three problems, each run REPS=3 times (median wall-clock reported):

    ou_small  : bundled `ornstein.npz`, n=20001, m=10,  RQ + ExpConst, max_iter=5
    ou_medium : concatenated OU,        n≈80004, m=30, RQ + ExpConst, max_iter=5
    lorenz    : stochastic Lorenz '63,  n≈29000, m=30, 3 components, max_iter=4

The output JSON is collated by `notebooks/07_device_comparison.ipynb`.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from voila_gp.inference import SDEVI
from voila_gp.init_heuristics import select_diffusion_parameters
from voila_gp.kernels import ExpConstKernel, ExpKernel, RQKernel
from voila_gp.multivariate import MultiSDEVI
from voila_gp.simulate import euler_maruyama

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
REPS = 3


@dataclass
class ProblemResult:
    problem: str
    wall_clock_s: list[float]
    median_s: float
    final_L: float | list[float]
    iterations: int | list[int]


def _sync(device: torch.device) -> None:
    """Block until pending GPU work is done, so wall-clock measures actual compute.

    Both CUDA and MPS dispatch asynchronously: without an explicit synchronize,
    `time.perf_counter()` can stop while GPU work is still in flight, making
    the device look artificially fast. CPU is naturally synchronous and needs
    no barrier.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _machine_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": _cpu_count(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def _cpu_count() -> int:
    import os
    try:
        return len(os.sched_getaffinity(0))  # honors taskset / cgroups
    except AttributeError:
        return os.cpu_count() or 0


def _run_ou(
    *,
    device: torch.device,
    dtype: torch.dtype,
    x: np.ndarray,
    h: float,
    m: int,
    max_iter: int,
) -> tuple[float, int]:
    torch.manual_seed(0)
    drift = RQKernel(amplitude=5.0, alpha=1.0, length_scale=1.5, epsilon=1e-5,
                     device=device, dtype=dtype)
    diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=5.0)
    diff = ExpConstKernel(
        max_amplitude=diff_params["kernel_amplitude"],
        exp_amplitude=diff_params["kernel_amplitude"] * 1e-3,
        length_scales=torch.tensor([1.5]),
        epsilon=1e-5,
        device=device, dtype=dtype,
    )
    inducing = np.linspace(x.min(), x.max(), m).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff, device=device, dtype=dtype)
    res = fit.fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        v_init=diff_params["v"], max_iter=max_iter, rel_tol=1e-6,
    )
    return float(res.lower_bound_history[-1]), int(res.iterations)


def _problem_ou_small(device: torch.device, dtype: torch.dtype) -> ProblemResult:
    npz = np.load(DATA_DIR / "ornstein.npz")
    x = npz["x"]
    h = float(npz["sampling_period"])

    times, finals, iters = [], [], []
    for _ in range(REPS):
        _sync(device)
        t0 = time.perf_counter()
        L, n_iter = _run_ou(device=device, dtype=dtype, x=x, h=h, m=10, max_iter=5)
        _sync(device)
        times.append(time.perf_counter() - t0)
        finals.append(L)
        iters.append(n_iter)
    return ProblemResult(
        problem="ou_small",
        wall_clock_s=times,
        median_s=float(np.median(times)),
        final_L=finals[-1],
        iterations=iters[-1],
    )


def _problem_ou_medium(device: torch.device, dtype: torch.dtype) -> ProblemResult:
    npz = np.load(DATA_DIR / "ornstein.npz")
    x_base = npz["x"]
    h = float(npz["sampling_period"])
    # Concat 4x — same OU process repeated, gives n≈80004. Inflates per-step
    # work without changing the algorithmic problem.
    x = np.tile(x_base, (4, 1)) if x_base.ndim == 2 else np.tile(x_base, 4)

    times, finals, iters = [], [], []
    for _ in range(REPS):
        _sync(device)
        t0 = time.perf_counter()
        L, n_iter = _run_ou(device=device, dtype=dtype, x=x, h=h, m=30, max_iter=5)
        _sync(device)
        times.append(time.perf_counter() - t0)
        finals.append(L)
        iters.append(n_iter)
    return ProblemResult(
        problem="ou_medium",
        wall_clock_s=times,
        median_s=float(np.median(times)),
        final_L=finals[-1],
        iterations=iters[-1],
    )


def _simulate_lorenz(n_steps: int, dt: float, seed: int) -> np.ndarray:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    noise = 1.0

    def drift(x: np.ndarray) -> np.ndarray:
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])

    def diffusion(_x: np.ndarray) -> np.ndarray:
        return np.full(3, noise)

    rng = np.random.default_rng(seed)
    ts = euler_maruyama(drift, diffusion, np.array([1.0, 1.0, 1.0]), dt, n_steps, rng)
    # Drop transient so we sit on the attractor.
    return ts[1000:]


def _problem_lorenz(device: torch.device, dtype: torch.dtype) -> ProblemResult:
    ts = _simulate_lorenz(n_steps=30_000, dt=0.005, seed=0)
    n = ts.shape[0]
    h = 0.005

    rng2 = np.random.default_rng(1)
    mean = ts.mean(0)
    cov = np.cov(ts, rowvar=False)
    inducing = rng2.multivariate_normal(mean, cov, size=30)

    # Cholesky on the Lorenz K_mm (m=30) loses positive-definiteness under FP32
    # at the original epsilon=1e-4 — the kernel hyperparameters drift into a
    # near-degenerate region during L-BFGS-B and round-off pushes the matrix
    # below PD. Bumping the jitter to 1e-3 stabilizes the FP32 path without
    # measurably affecting the FP64 (CPU/CUDA) numbers. See BENCH_TODO_MAC.md
    # for the original failure mode and the documented mitigation.
    lorenz_eps = 1e-3 if dtype == torch.float32 else 1e-4

    def drift_kernel_factory(_x, _h, _i):
        return ExpKernel(
            amplitude=200.0, length_scales=torch.tensor([6.0, 6.0, 6.0]),
            epsilon=lorenz_eps, device=device, dtype=dtype,
        )

    def diff_kernel_factory(_x, _h, _i):
        return ExpKernel(
            amplitude=2.0, length_scales=torch.tensor([6.0, 6.0, 6.0]),
            epsilon=lorenz_eps, device=device, dtype=dtype,
        )

    times: list[float] = []
    final_L_per: list[list[float]] = []
    iters_per: list[list[int]] = []
    for _ in range(REPS):
        torch.manual_seed(0)
        fit = MultiSDEVI(
            drift_kernel_factory=drift_kernel_factory,
            diff_kernel_factory=diff_kernel_factory,
            prior_on_sd=2.0,
            device=device, dtype=dtype,
        )
        _sync(device)
        t0 = time.perf_counter()
        res = fit.fit(
            time_series=ts, sampling_period=h, inducing_points=inducing,
            max_iter=4, rel_tol=1e-5, hp_max_iter=80,
        )
        _sync(device)
        times.append(time.perf_counter() - t0)
        final_L_per.append([float(c.lower_bound_history[-1]) for c in res.components])
        iters_per.append([int(c.iterations) for c in res.components])
    # Use last rep's per-component finals/iters; n is implicit in the problem definition.
    _ = n  # silence pyright if unused; documented below
    return ProblemResult(
        problem="lorenz",
        wall_clock_s=times,
        median_s=float(np.median(times)),
        final_L=final_L_per[-1],
        iterations=iters_per[-1],
    )


PROBLEMS = {
    "ou_small": _problem_ou_small,
    "ou_medium": _problem_ou_medium,
    "lorenz": _problem_lorenz,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path. Defaults to stdout-only.")
    parser.add_argument("--problems", nargs="*", choices=list(PROBLEMS.keys()),
                        default=list(PROBLEMS.keys()),
                        help="Subset of problems to run (default: all).")
    parser.add_argument("--dtype", choices=["float32", "float64", "default"], default="default",
                        help="Override dtype. 'default' uses _default_dtype_for(device).")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.dtype == "default":
        from voila_gp._compat import _default_dtype_for
        dtype = _default_dtype_for(device)
    else:
        dtype = getattr(torch, args.dtype)

    print(f"Running on device={device}, dtype={dtype}, REPS={REPS}")
    results: list[dict[str, Any]] = []
    for name in args.problems:
        print(f"\n--- {name} ---")
        result = PROBLEMS[name](device, dtype)
        print(f"  wall_clock_s = {result.wall_clock_s}")
        print(f"  median_s     = {result.median_s:.3f}")
        print(f"  final_L      = {result.final_L}")
        print(f"  iterations   = {result.iterations}")
        results.append(asdict(result))

    payload = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "machine": _machine_info(device),
        "results": results,
    }
    payload_str = json.dumps(payload, indent=2, default=str)
    print("\n" + payload_str)
    if args.output is not None:
        args.output.write_text(payload_str + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
