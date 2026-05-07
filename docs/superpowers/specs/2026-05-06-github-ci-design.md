# GitHub CI Design

**Date:** 2026-05-06  
**Repo:** williamjsdavis/voila-gpytorch

## Goal

Add GitHub Actions CI that gives fast PR feedback and full regression coverage on `main`.

## Workflow: `.github/workflows/ci.yml`

Single workflow file, two jobs.

### Job 1: `ci`

**Triggers:** push to any branch, pull_request to any branch  
**Runner:** `ubuntu-latest`

Steps:
1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v5` (with `enable-cache: true`)
3. `uv sync` (installs all deps including dev group)
4. `uv run ruff check src tests scripts`
5. `uv run pyright src`
6. `uv run pytest tests/unit -q`

### Job 2: `regression`

**Triggers:** push to `main` only  
**Runner:** `ubuntu-latest`

Steps:
1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v5` (with `enable-cache: true`)
3. `uv sync`
4. `uv run pytest -m regression -q`

## Rationale

- **No version matrix** — `pyproject.toml` pins `requires-python = ">=3.12,<3.13"`, so a single 3.12 runner covers all supported versions.
- **Regression on `main` only** — the two regression tests run full inference loops (minutes each); running them on every PR would make CI impractically slow.
- **`setup-uv` caching** — caches the uv package store between runs, so repeated installs of torch/gpytorch don't re-download.
- **No secrets required** — all test fixtures are vendored `.npz` files in `tests/data/`.
