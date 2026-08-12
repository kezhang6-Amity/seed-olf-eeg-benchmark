"""Build and execute a reader-facing Phase 3 results notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "phase3_fewshot")
    parser.add_argument("--notebook", type=Path, default=Path("notebooks") / "phase3_fewshot.ipynb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    incremental = pd.read_csv(args.output_dir / "incremental_value.csv")
    negative = pd.read_csv(args.output_dir / "negative_control_summary.csv")
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text())
    p2 = incremental[incremental.candidate == "P2"].set_index("calibration_size")
    tldr = f"""## tl;dr

- **Classification: `{metadata['classification']}`.** P2 passes the full gate at **4 calibration trials** (one per odor).
- At 4 trials, P2 improves held-out Sessions-2/3 log loss by **{p2.loc[4, 'mean_log_loss_improvement']:.6f}** (95% CI **{p2.loc[4, 'log_loss_ci95_lower']:.6f} to {p2.loc[4, 'log_loss_ci95_upper']:.6f}**) over the leave-target pooled odor prior.
- The improvement is small but observed for **{p2.loc[4, 'participants_improved_rate']:.1%}** of participants and exceeds all 20 permuted-label controls at each nonzero calibration size.
- Direct target-only estimates and the fixed EEG comparator are worse; pooled shrinkage, not EEG, drives the supported conclusion.
"""
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell("# SEED-OLF Phase 3: Few-Shot Personalization"),
        nbf.v4.new_markdown_cell(tldr),
        nbf.v4.new_markdown_cell("""## Methods

Session 1 supplies participant-specific calibration labels. Sessions 2–3 are held out intact. For 4, 8, and 16 trials, each calibration subset is exactly balanced across the four odors and sampled independently of labels. P2 updates a leave-target-participant-out, odor-specific Beta prior. Intervals use a three-level paired bootstrap.

The negative control permutes each participant's Session-1 labels across odors. It retains overall individual response tendency while breaking odor-specific correspondence; therefore the observed effect must exceed its permutation distribution rather than merely exceed zero.
"""),
        nbf.v4.new_code_cell(f"""from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style='whitegrid')
OUTPUT_DIR = Path({args.output_dir.as_posix()!r})
summary = pd.read_csv(OUTPUT_DIR / 'prediction_summary.csv')
incremental = pd.read_csv(OUTPUT_DIR / 'incremental_value.csv')
negative = pd.read_csv(OUTPUT_DIR / 'negative_control_summary.csv')
metadata = json.loads((OUTPUT_DIR / 'run_metadata.json').read_text())
validation = json.loads((OUTPUT_DIR / 'validation_checks.json').read_text())
"""),
        nbf.v4.new_markdown_cell("## Primary calibration curve"),
        nbf.v4.new_code_cell("""p2 = incremental[incremental.candidate == 'P2'].query('calibration_size > 0').copy()
display(p2.round(6))
fig, ax = plt.subplots(figsize=(7, 4))
errors = [p2.mean_log_loss_improvement - p2.log_loss_ci95_lower, p2.log_loss_ci95_upper - p2.mean_log_loss_improvement]
ax.errorbar(p2.calibration_size, p2.mean_log_loss_improvement, yerr=errors, fmt='o-', capsize=4, label='P2 observed')
null_max = negative[negative.calibration_size > 0].groupby('calibration_size').mean_log_loss_improvement.max()
ax.plot(null_max.index, null_max.values, 's--', label='max permuted control')
ax.axhline(0, color='black', linewidth=1)
ax.set(xlabel='Session-1 calibration trials', ylabel='Log-loss improvement over P0', title='Few-shot personalized posterior')
ax.legend()
plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("## Comparator and audit checks"),
        nbf.v4.new_code_cell("""display(incremental.query("calibration_size in [4, 24]").round(6))
display(pd.DataFrame(metadata['success_gate']['by_size']).T)
display(pd.Series(validation, name='value').to_frame())
"""),
        nbf.v4.new_markdown_cell("""## Takeaways

1. Four odor-balanced labeled trials are enough for a statistically stable, though small, cross-session calibration gain in this public dataset.
2. Direct target-only estimates overfit; empirical-Bayes pooling is essential.
3. The fixed EEG representation adds no value here, so a future deep model must beat P2 under the same grouped and negative-control audits before it is considered useful.
4. Future collection should prospectively reproduce this transparent posterior baseline while treating multimodal physiology as an incremental, not assumed, source of information.
"""),
    ]
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    executed = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}}).execute()
    nbf.write(executed, args.notebook)
    print(f"Executed notebook written to {args.notebook}")


if __name__ == "__main__":
    main()
