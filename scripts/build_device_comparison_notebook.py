"""Construct the device comparison notebook from collected `bench_*.json` files.

Run:
    uv run python scripts/build_device_comparison_notebook.py
    uv run jupyter execute --inplace notebooks/07_device_comparison.ipynb

The notebook is *pure plotter*: it loads every `bench_*.json` it finds in the
repo root and renders bar charts + speedup tables. It is meant to be re-run
each time a new JSON arrives (e.g. after the Mac runs).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "notebooks" / "07_device_comparison.ipynb"


def md(s: str):
    return nbf.v4.new_markdown_cell(s)


def code(s: str):
    return nbf.v4.new_code_cell(s)


CELLS = [
    md("""# Choosing a device: CPU, MPS, CUDA, and the R baseline

`voila-gp` runs on any device PyTorch supports. This notebook collates timing
data collected on multiple machines into a single comparison so users can see
what each device actually buys them on the same fixed problem set.

The data comes from `scripts/run_device_benchmark.py` (Python) and
`scripts/run_r_benchmark.R` (R). Each JSON is one machine × one device; this
notebook glues them together. Re-run after any new JSON lands.
"""),
    code("""import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

REPO_ROOT = Path('.').resolve()
# Notebook lives in notebooks/, JSONs live in repo root.
if REPO_ROOT.name == 'notebooks':
    REPO_ROOT = REPO_ROOT.parent

bench_files = sorted(REPO_ROOT.glob('bench_*.json'))
print(f'Found {len(bench_files)} benchmark file(s):')
for p in bench_files:
    print(f'  - {p.name}')
"""),
    md("""## Load all benchmark JSONs"""),
    code("""def load_bench(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload['source_file'] = path.name
    # Derive a short label like 'A10 CUDA' from filename + payload.
    stem = path.stem.removeprefix('bench_')
    payload['label'] = stem.replace('_', ' ').upper()
    return payload

benches = [load_bench(p) for p in bench_files]

# Long-form table: one row per (label, problem) with median wall-clock.
rows = []
for b in benches:
    for r in b['results']:
        rows.append({
            'label': b['label'],
            'device': b['device'],
            'dtype': b.get('dtype', 'unknown'),
            'problem': r['problem'],
            'median_s': r['median_s'],
            'final_L': r['final_L'],
        })
df = pd.DataFrame(rows)
df
"""),
    md("""## Wall-clock per problem, per device

Lower is better. Bars are the median of three runs."""),
    code("""if df.empty:
    print('No benchmark data found yet — populate by running:')
    print('  uv run python scripts/run_device_benchmark.py --device cpu  --output bench_a10_cpu.json')
    print('  uv run python scripts/run_device_benchmark.py --device cuda --output bench_a10_cuda.json')
    print('  (and similar on Mac CPU / MPS)')
else:
    problems = df['problem'].unique()
    fig, axes = plt.subplots(1, len(problems), figsize=(4.2 * len(problems), 4),
                             squeeze=False)
    for ax, problem in zip(axes[0], problems):
        sub = df[df['problem'] == problem].sort_values('median_s')
        ax.barh(sub['label'], sub['median_s'], color='steelblue')
        ax.set(xlabel='wall-clock (s)', title=problem)
        for i, v in enumerate(sub['median_s']):
            ax.text(v, i, f' {v:.1f}s', va='center', fontsize=9)
    plt.tight_layout(); plt.show()
"""),
    md("""## Speedup table

Each Python device's wall-clock divided by the slowest CPU baseline on the
same problem. Higher = faster."""),
    code("""if not df.empty and 'CPU' in ' '.join(df['label'].unique()).upper():
    # Pick the slowest CPU run as baseline (typically the Mac CPU when both
    # are present; falls back to A10 CPU).
    cpu_mask = df['label'].str.contains('CPU', case=False)
    cpu_baseline = df[cpu_mask].groupby('problem')['median_s'].max()
    spd = df.copy()
    spd['speedup_vs_slowest_cpu'] = spd.apply(
        lambda r: cpu_baseline.get(r['problem'], np.nan) / r['median_s'], axis=1
    )
    pivot = spd.pivot(index='label', columns='problem', values='speedup_vs_slowest_cpu')
    display(pivot.round(2))
else:
    print('Not enough data to build a speedup table yet — need at least one CPU run.')
"""),
    md("""## Final ELBO across devices on `ou_small`

Every Python device should land on the OU anchor (≈ 36698) within its
documented tolerance band. Sanity check that nothing is silently broken."""),
    code("""ou_small = df[df['problem'] == 'ou_small'][['label', 'dtype', 'final_L']].sort_values('label')
if ou_small.empty:
    print('No ou_small data yet.')
else:
    display(ou_small)
    spread = ou_small['final_L'].max() - ou_small['final_L'].min()
    print(f'\\nELBO spread across devices: {spread:.3f}')
    print('Expected: <0.5 across CPU + CUDA (FP64), <50 if MPS (FP32) is included.')
"""),
    md("""## Reading the chart

- **Small / 1-D problems (`ou_small`)** are dominated by L-BFGS-B + Python
  overhead. CPU is often the *fastest* option here because GPU kernel-launch
  latency exceeds the per-step compute on n=20k, m=10 matrices.
- **Medium / multivariate problems (`ou_medium`, `lorenz`)** are where the
  GPU pays off: bigger inducing matrices and longer trajectories give the
  Cholesky/MM kernels enough work to amortize launch overhead.
- **R baseline (when present)** anchors the absolute scale. The Python
  port at FP64 should be within a small constant factor of the original
  Fortran-backed R implementation on the same problem.
- **MPS (when present)** runs at FP32, so its ELBO will differ from the
  CPU/CUDA values by a documented (loose) tolerance. The recovered drift
  function is unaffected — verified by the device-validation tests in
  `tests/unit/test_devices.py`.
"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    return nb


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), NB_PATH)
    print(f"Wrote {NB_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
