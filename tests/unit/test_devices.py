"""Per-device validation tests + tolerance bands.

These tests are parametrized over the devices available on the current
machine. CPU is always present; CUDA and MPS are added when their PyTorch
backend is available. Each device gets its own tolerance band — float64 on
CPU/CUDA is held to a tight bar, MPS/float32 to a substantially looser one.

Tolerance rationale (DEVICE_TOLERANCES):
- CPU/float64 is the existing R-parity baseline (kernel cov bit-identical
  to the reference within 1e-12; ELBO within ±0.5 of the published log).
- CUDA/float64 differs from CPU only via BLAS algorithm choice, which
  introduces ~1e-10 drift on small matrices. Same ELBO band as CPU.
- MPS/float32 has ~7 decimal digits, so 1e-5 on individual kernel entries
  and ±50 on the integrated ELBO over n=20k points is the realistic bar.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voila_gp.inference import SDEVI
from voila_gp.init_heuristics import select_diffusion_parameters
from voila_gp.kernels import ExpConstKernel, RQKernel


def _available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


DEVICE_TOLERANCES: dict[str, dict[str, float]] = {
    "cpu":  {"kernel_atol": 1e-12, "elbo_atol": 0.5},
    "cuda": {"kernel_atol": 1e-10, "elbo_atol": 0.5},
    "mps":  {"kernel_atol": 1e-5,  "elbo_atol": 50.0},
}


@pytest.mark.parametrize("device", _available_devices())
def test_kernel_cov_matches_cpu_reference(device: str) -> None:
    """RQKernel.cov(X) on `device` agrees with the CPU/float64 reference within
    that device's tolerance band."""
    torch.manual_seed(0)
    x_np = np.linspace(-2, 2, 7).reshape(-1, 1)

    # CPU reference at float64 — the gold standard.
    cpu = RQKernel(amplitude=2.0, alpha=1.5, length_scale=0.7, epsilon=1e-5, device="cpu")
    x_cpu = torch.as_tensor(x_np, dtype=cpu.dtype, device=cpu.device)
    K_ref = cpu.cov(x_cpu).detach().cpu().numpy()

    # Same kernel + input on the target device.
    other = RQKernel(amplitude=2.0, alpha=1.5, length_scale=0.7, epsilon=1e-5, device=device)
    x_other = torch.as_tensor(x_np, dtype=other.dtype, device=other.device)
    K_other = other.cov(x_other).detach().cpu().numpy()

    atol = DEVICE_TOLERANCES[device]["kernel_atol"]
    np.testing.assert_allclose(K_other, K_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("device", _available_devices())
def test_ou_runs_on_device(device: str, ornstein_data: dict) -> None:
    """SDEVI.fit completes on every available device and recovers a sensible L.

    We only check that L lands above 36000 (well past the random-init bound
    of ≈ -59000 and within striking distance of the ≈ 36698 anchor) — tight
    parity is the regression test's job, not the device test's.
    """
    x = ornstein_data["x"]
    h = ornstein_data["sampling_period"]
    drift = RQKernel(amplitude=5.0, alpha=1.0, length_scale=1.5, epsilon=1e-5, device=device)
    diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=5.0)
    diff = ExpConstKernel(
        max_amplitude=diff_params["kernel_amplitude"],
        exp_amplitude=diff_params["kernel_amplitude"] * 1e-3,
        length_scales=torch.tensor([1.5]),
        epsilon=1e-5,
        device=device,
    )
    inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff, device=device)
    res = fit.fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        v_init=diff_params["v"], max_iter=2, rel_tol=1e-6,
    )
    final_L = res.lower_bound_history[-1]
    assert final_L > 36000.0, f"final L {final_L} too low on {device}"
    # All result tensors should live on the chosen device.
    assert res.f_mean.device.type == torch.device(device).type
    assert res.inducing_points.device.type == torch.device(device).type
    assert res.device.type == torch.device(device).type


@pytest.mark.parametrize("device", _available_devices())
def test_simulate_forward_lives_on_device(device: str, ornstein_data: dict) -> None:
    """`simulate_forward` produces trajectories on the fit's device."""
    from voila_gp.forecast import simulate_forward
    from voila_gp.multivariate import MultiSDEVIResult

    x = ornstein_data["x"]
    h = ornstein_data["sampling_period"]
    drift = RQKernel(amplitude=5.0, alpha=1.0, length_scale=1.5, epsilon=1e-5, device=device)
    diff_params = select_diffusion_parameters(x, sampling_period=h, prior_on_sd=5.0)
    diff = ExpConstKernel(
        max_amplitude=diff_params["kernel_amplitude"],
        exp_amplitude=diff_params["kernel_amplitude"] * 1e-3,
        length_scales=torch.tensor([1.5]),
        epsilon=1e-5,
        device=device,
    )
    inducing = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)
    fit = SDEVI(drift_kernel=drift, diff_kernel=diff, device=device)
    res = fit.fit(
        time_series=x, sampling_period=h, inducing_points=inducing,
        v_init=diff_params["v"], max_iter=1, rel_tol=1e-6,
    )
    multi = MultiSDEVIResult(components=[res], sampling_period=h, target_indices=[0])
    forecast = simulate_forward(
        multi, x0=np.array([0.0]), n_steps=20, n_ensembles=8, seed=0,
    )
    assert forecast.trajectories.device.type == torch.device(device).type
    assert forecast.times.device.type == torch.device(device).type
    assert forecast.trajectories.shape == (8, 21, 1)
