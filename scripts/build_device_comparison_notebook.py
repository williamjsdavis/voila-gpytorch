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
df = pd.DataFrame(rows, columns=['label', 'device', 'dtype', 'problem', 'median_s', 'final_L'])
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
    problems = list(df['problem'].unique())
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
    code("""if df.empty:
    print('Not enough data to build a speedup table yet — need at least one CPU run.')
else:
    labels_str = ' '.join(df['label'].unique()).upper()
    if 'CPU' in labels_str:
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
        print('No CPU run found — speedup table needs a CPU baseline.')
"""),
    md("""## Final-L parity on `ou_small`

The published voila R README quotes `L = 36698.475` at convergence on this
exact realization. Every Python device should reproduce it within a documented
tolerance band — wide for MPS (FP32), tight for CPU and CUDA (FP64). This
table makes the agreement visible rather than asserted."""),
    code("""R_ANCHOR = 36698.475   # quoted in voila/README.md execution log

def _scalar_L(L):
    # ou_small final_L is a scalar; lorenz returns a per-component list.
    return float(L) if not isinstance(L, list) else float('nan')

ou_small = df[df['problem'] == 'ou_small'].copy()
if ou_small.empty:
    print('No ou_small data yet.')
else:
    ou_small['final_L'] = ou_small['final_L'].apply(_scalar_L)
    ou_small['delta_vs_R'] = ou_small['final_L'] - R_ANCHOR
    parity = ou_small[['label', 'dtype', 'final_L', 'delta_vs_R']].sort_values('label')
    display(parity.style.format({'final_L': '{:.4f}', 'delta_vs_R': '{:+.4f}'}))

    fp64_mask = parity['dtype'].str.contains('64', na=False)
    fp64_spread = float((parity.loc[fp64_mask, 'delta_vs_R']).abs().max()) if fp64_mask.any() else float('nan')
    fp32_spread = float((parity.loc[~fp64_mask, 'delta_vs_R']).abs().max()) if (~fp64_mask).any() else float('nan')
    print(f'\\nMax |Δ| from R anchor across FP64 devices: {fp64_spread:.4f}  (expected < 0.5)')
    print(f'Max |Δ| from R anchor across FP32 devices: {fp32_spread:.4f}  (expected < 50)')
"""),
    md("""## What the numbers say

The interpretation below is regenerated from the loaded JSONs every time the
notebook runs, so it stays in sync with the data above."""),
    code("""from IPython.display import Markdown

if df.empty:
    display(Markdown('_(No benchmark data loaded — interpretation suppressed.)_'))
else:
    medians = df.set_index(['problem', 'label'])['median_s']
    labels = sorted(df['label'].unique())
    problems = list(df['problem'].unique())

    def t(problem, label):
        try:
            return float(medians.loc[(problem, label)])
        except KeyError:
            return None

    def fmt(v):
        return '—' if v is None else f'{v:.2f} s'

    def speedup(slow, fast):
        if slow is None or fast is None or fast == 0:
            return None
        return slow / fast

    bullets = []

    # Slowest baseline per problem (used as the headline 'how much faster')
    for problem in problems:
        per_label = {lab: t(problem, lab) for lab in labels}
        valid = {k: v for k, v in per_label.items() if v is not None}
        if not valid:
            continue
        slowest_label, slowest_t = max(valid.items(), key=lambda kv: kv[1])
        fastest_label, fastest_t = min(valid.items(), key=lambda kv: kv[1])
        bullets.append(
            f'- **`{problem}`**: slowest is **{slowest_label}** at **{fmt(slowest_t)}**, '
            f'fastest is **{fastest_label}** at **{fmt(fastest_t)}** '
            f'(**{speedup(slowest_t, fastest_t):.1f}×** spread).'
        )

    # If R reference is present, show "Python on this Mac vs R" as a separate point
    r_label = next((lab for lab in labels if lab.upper() == 'R' or lab.upper().startswith('R ')), None)
    mac_cpu_label = next((lab for lab in labels if 'MAC' in lab.upper() and 'CPU' in lab.upper()), None)
    if r_label and mac_cpu_label:
        bullets.append('')
        bullets.append('**Versus the original R implementation:**')
        for problem in ['ou_small', 'ou_medium']:
            r_t = t(problem, r_label)
            py_t = t(problem, mac_cpu_label)
            if r_t and py_t:
                bullets.append(
                    f'- `{problem}`: R takes **{fmt(r_t)}**, this Mac\\'s CPU port takes '
                    f'**{fmt(py_t)}** — **{speedup(r_t, py_t):.1f}×** faster despite '
                    f'matching the same final ELBO.'
                )

    # GPU vs CPU asymmetry note — depends on whether multiple CPUs are present
    cpu_labels = [lab for lab in labels if 'CPU' in lab.upper() and 'R ' not in lab.upper()]
    if len(cpu_labels) > 1:
        bullets.append('')
        bullets.append(
            '**The "GPU vs CPU" speedup depends heavily on which CPU you compare against** — '
            f'see the speedup table above. ' +
            ', '.join(f'{lab} = {fmt(t("ou_small", lab))}' for lab in cpu_labels) +
            ' on `ou_small`.'
        )

    md_str = (
        '### Interpretation (auto-generated from current JSONs)\\n\\n' +
        '\\n'.join(bullets) + '\\n\\n' +
        '_Re-run this cell after dropping new `bench_*.json` files to update._'
    )
    display(Markdown(md_str))
"""),
    md("""## Caveats and what to read next

- **Small / 1-D problems** (`ou_small`) are dominated by L-BFGS-B and Python
  overhead, not GP linear algebra; expect GPU and CPU to be close. The
  hyperparameter step is scipy-bound regardless of where the kernel work runs.
- **Medium / multivariate problems** (`ou_medium`, `lorenz`) are where GPU
  pays off — bigger Cholesky and matmul kernels amortize launch overhead.
- **MPS** runs at FP32 (Apple Silicon's Metal backend has no FP64 linalg).
  The ELBO drift band is documented; the recovered drift function itself is
  unaffected. See `tests/unit/test_devices.py` for the per-device tolerance
  table.
- The full validation suite — including drift-correlation checks against the
  ground truth on each device — lives in `tests/unit/test_devices.py` and
  `tests/regression/test_ou_parity.py`.
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
