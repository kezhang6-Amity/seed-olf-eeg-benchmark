"""Build and execute the reader-facing Phase 2 results notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts") / "phase2_channel"
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks") / "phase2_channel.ipynb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.output_dir / "prediction_summary.csv")
    incremental = pd.read_csv(args.output_dir / "incremental_value.csv")
    profile = json.loads(
        (args.output_dir / "data_quality_profile.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (args.output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    loso = summary[summary.protocol == "loso"].set_index("model")
    primary = incremental[
        (incremental.protocol == "loso") & (incremental.candidate == "M2")
    ].iloc[0]
    best_ranking_model = loso.roc_auc.idxmax()
    negative = pd.read_csv(args.output_dir / "negative_control_summary.csv")

    tldr = f"""## tl;dr

- **Data gate:** {profile['n_trials']:,} paired trials from {profile['n_subjects']} participants passed schema and finite-value checks; all {profile['feature_count']:,} channel features were finite with no zero-variance feature.
- **Primary result:** LOSO odor prior M0 log loss was **{loso.loc['M0', 'log_loss']:.4f}**. Odor + residualized channel log-power M2 was **{loso.loc['M2', 'log_loss']:.4f}**.
- M2 mean paired log-loss improvement was **{primary.mean_log_loss_improvement:.4f}** (95% CI **{primary.log_loss_ci95_lower:.4f} to {primary.log_loss_ci95_upper:.4f}**); positive would favor EEG.
- The strongest LOSO ranking model was **{best_ranking_model}** with AUROC **{loso.loc[best_ranking_model, 'roc_auc']:.4f}**, but no channel model beat M0 on probabilistic loss.
- All {len(negative)} odor-stratified label-permutation controls failed the success gate; median permuted LOSO improvement was **{negative.loso_log_loss_improvement.median():.4f}**.
- **Classification: `{metadata['classification']}`.** Ranking signal is present, but reliable incremental probability value is not established.
"""

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell("# SEED-OLF Phase 2: Odor-Controlled Channel Features"),
        nbf.v4.new_markdown_cell(tldr),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The question is whether channel-resolved EEG improves held-out prediction of the public binary subjective-valence proxy beyond odor identity. M2 is the single primary EEG comparison. All odor residualization, scaling, clipping, and tuning are fit inside training folds.

### Key Assumptions

- Sampling rate is assumed to be 200 Hz from 3,000 samples over the reported 15-second stage.
- Channels are array indices `ch00`–`ch61`; no electrode or topographic claim is made.
- `clean_breathing` is post-stimulus recovery, not a causal pre-stimulus baseline.
- `y_emotion` is a methodological proxy, not relaxation or a clinical outcome.
"""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
OUTPUT_DIR = Path({args.output_dir.as_posix()!r})
profile = json.loads((OUTPUT_DIR / "data_quality_profile.json").read_text())
metadata = json.loads((OUTPUT_DIR / "run_metadata.json").read_text())
summary = pd.read_csv(OUTPUT_DIR / "prediction_summary.csv")
incremental = pd.read_csv(OUTPUT_DIR / "incremental_value.csv")
equivalence = pd.read_csv(OUTPUT_DIR / "logbp_de_equivalence.csv")
negative = pd.read_csv(OUTPUT_DIR / "negative_control_summary.csv")
validation = json.loads((OUTPUT_DIR / "validation_checks.json").read_text())
"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Check dataset and feature quality"),
        nbf.v4.new_code_cell(
            """quality = pd.DataFrame(profile["signal_qc_by_phase"]).T
display(pd.Series({
    "trials": profile["n_trials"],
    "subjects": profile["n_subjects"],
    "participant_sessions": profile["n_sessions"],
    "feature_count": profile["feature_count"],
    "finite_feature_rate": profile["finite_feature_rate"],
    "zero_variance_features": profile["zero_variance_features"],
}, name="value").to_frame())
display(quality)
"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. Compare held-out probabilistic performance"),
        nbf.v4.new_code_cell(
            """display(summary.round(4))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(data=summary, x="model", y="log_loss", hue="protocol", ax=axes[0])
sns.barplot(data=summary, x="model", y="roc_auc", hue="protocol", ax=axes[1])
axes[0].set_title("Held-out log loss (lower is better)")
axes[1].set_title("Held-out AUROC (higher is better)")
for axis in axes:
    axis.legend(title="Protocol")
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell("### 3. Inspect paired incremental value and uncertainty"),
        nbf.v4.new_code_cell(
            """display(incremental.round(4))
loso = incremental[incremental.protocol == "loso"].copy()
fig, ax = plt.subplots(figsize=(8, 4.5))
errors = np.vstack([
    loso.mean_log_loss_improvement - loso.log_loss_ci95_lower,
    loso.log_loss_ci95_upper - loso.mean_log_loss_improvement,
])
ax.errorbar(loso.candidate, loso.mean_log_loss_improvement, yerr=errors, fmt="o", capsize=4)
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("Log-loss improvement over M0")
ax.set_title("LOSO paired hierarchical-bootstrap intervals")
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell("### 4. Quantify log-power and DE equivalence"),
        nbf.v4.new_code_cell(
            """display(equivalence.describe().round(4))
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(equivalence.median_correlation, bins=10, ax=ax)
ax.set_title("Fold-wise median correlation: corresponding log-power and DE features")
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell("### 5. Check odor-stratified label permutations"),
        nbf.v4.new_code_cell(
            """display(negative.round(4))
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(negative.loso_log_loss_improvement, bins=10, ax=ax)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Negative-control LOSO improvement distribution")
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Channel-level EEG contains some held-out ranking information, but the preregistered M2 probability-value gate fails.
2. Robust clipping/scaling reduces M2's damage but does not establish positive incremental log loss.
3. Within-participant cross-session performance is substantially worse than odor-only prediction, so the current 310-dimensional representation is not ready for few-shot personalization without stronger shrinkage or lower-dimensional structure.
4. DE and log-power are highly correlated and should remain separate comparisons rather than a concatenated feature expansion.
5. These results justify completing the planned Phase 3 calibration-curve baseline, but they do not justify promoting EEGNet or TSception as likely solutions yet.
"""
        ),
    ]
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    executed = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
    nbf.write(executed, args.notebook)
    print(f"Executed notebook written to {args.notebook}")


if __name__ == "__main__":
    main()
