"""Shared test fixtures: R-data loaders, tolerance bands, deterministic seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _set_torch_defaults():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="session")
def ornstein_data():
    """Bundled R OU realization (length 20001, dt=0.001)."""
    npz = np.load(DATA_DIR / "ornstein.npz")
    return {"x": npz["x"], "sampling_period": float(npz["sampling_period"])}


@pytest.fixture(scope="session")
def do_events_data():
    """Bundled NGRIP δ18O paleoclimate series (length 999, dt=1)."""
    npz = np.load(DATA_DIR / "do_events.npz")
    return {"x": npz["x"], "sampling_period": float(npz["sampling_period"])}
