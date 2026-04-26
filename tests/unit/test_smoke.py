"""Phase 0 gate: pytest framework wires up; fixtures load."""

import numpy as np
import torch

import voila_gp


def test_package_importable():
    assert voila_gp.__version__ == "0.1.0"


def test_torch_default_dtype_is_double():
    # conftest autouse fixture sets float64; correctness of all numerical work below depends on it
    assert torch.empty(1).dtype is torch.float64


def test_ornstein_fixture_loads(ornstein_data):
    x = ornstein_data["x"]
    assert x.shape == (20001, 1)
    assert ornstein_data["sampling_period"] == 0.001
    # Initial value of the bundled realization is 0 (per simulate_sde default x0)
    assert x[0, 0] == 0.0
    # Stationary OU with drift -x and diff sqrt(1.5) has variance ~ 1.5/(2*1) = 0.75
    sample_var = float(np.var(x[1000:]))
    assert 0.5 < sample_var < 1.0


def test_do_events_fixture_loads(do_events_data):
    x = do_events_data["x"]
    assert x.shape == (999, 1)
    # δ18O values for Greenland ice core are typically in the range [-46, -36]
    assert -46.0 < float(np.min(x)) < -40.0
    assert -42.0 < float(np.max(x)) < -34.0
