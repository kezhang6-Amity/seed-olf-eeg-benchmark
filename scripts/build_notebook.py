"""Build and execute the reader-facing SEED-OLF baseline results notebook."""

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
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "baseline_v1",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks") / "baseline_effects.ipynb",
    )
    return parser.parse_args()


def fmt_metric(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.3f}"


def build_summary(output_dir: Path) -> tuple[str, str]:
    manifest = pd.read_csv(output_dir / "manifest.csv")
    effects = pd.read_csv(output_dir / "paired_spectral_effects.csv")
    reliability = pd.read_csv(output_dir / "cross_session_reliability.csv")
    predictions = pd.read_csv(output_dir / "prediction_summary.csv")
    incremental = pd.read_csv(output_dir / "incremental_value.csv")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    loso = predictions[(predictions.protocol == "loso") & (predictions.subset == "all")]
    odor = loso[loso.model == "odor_prior"].iloc[0]
    candidate = loso[loso.model == "odor_eeg_paired_logit"].iloc[0]
    riemann_candidate = loso[loso.model == "odor_riemann_stim_logit"].iloc[0]
    candidate_increment = incremental[
        (incremental.protocol == "loso")
        & (incremental.subset == "all")
        & (incremental.model == "odor_eeg_paired_logit")
    ].iloc[0]
    significant_effects = effects[effects.wilcoxon_q_value < 0.05]
    reliable = reliability[reliability.icc2_1 >= 0.5]

    tldr = f"""## tl;dr

- The executed dataset contains **{len(manifest):,} stimulation trials from {manifest.subject.nunique()} participants and 3 sessions per participant**.
- The subjective class-1 rate is **{manifest.y_emotion.mean():.1%}**; odor identity is therefore evaluated as a mandatory shortcut baseline.
- Under leave-one-subject-out evaluation, the odor-only prior achieved log loss **{fmt_metric(odor.log_loss)}** and balanced accuracy **{fmt_metric(odor.balanced_accuracy)}**.
- Adding paired stimulation–recovery EEG features produced log loss **{fmt_metric(candidate.log_loss)}** and balanced accuracy **{fmt_metric(candidate.balanced_accuracy)}**. Its participant-block mean log-loss improvement over odor prior was **{candidate_increment.mean_log_loss_improvement:.3f}** (95% CI **{candidate_increment.ci95_lower:.3f} to {candidate_increment.ci95_upper:.3f}**). Positive values favor EEG.
- The spatial OAS covariance + Log-Euclidean model produced LOSO log loss **{fmt_metric(riemann_candidate.log_loss)}** and balanced accuracy **{fmt_metric(riemann_candidate.balanced_accuracy)}**; it also did not beat the odor prior.
- **{len(significant_effects)}/{len(effects)}** global spectral stimulation–recovery effects passed within-family FDR control; **{len(reliable)}/{len(reliability)}** odor-band features reached ICC(2,1) ≥ 0.5.
- Frequency interpretation currently uses **{metadata['sampling_rate_hz']:.0f} Hz inferred from 3,000 samples over the reported 15-second stage**. This remains a metadata assumption, and clean breathing is a post-stimulus recovery condition rather than a pre-stimulus baseline.
"""

    improvement = odor.log_loss - candidate.log_loss
    direction = "improved" if improvement > 0 else "did not improve"
    takeaway = f"""## Takeaways

1. The leakage-safe baseline is operational at the participant, session, and trial levels.
2. The paired EEG model {direction} leave-one-subject-out log loss relative to odor identity by **{abs(improvement):.3f}** in this first feature set.
3. Any EEG benefit must remain after participant-level uncertainty analysis and on odor-prior disagreement trials before it is interpreted as personalized response information.
4. Cross-session reliability results should guide feature selection and future repetition counts; low reliability is evidence against using a feature for personalization.
5. The next justified work is an odor-controlled disagreement/residual target plus channel-resolved differential-entropy features. EEGNet and TSception are appropriate compact deep comparisons after that reframing; DSEN remains deferred until authoritative code or a complete specification is available.
"""
    return tldr, takeaway


def main() -> None:
    args = parse_args()
    tldr, takeaway = build_summary(args.output_dir)
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}

    notebook["cells"] = [
        nbf.v4.new_markdown_cell("# SEED-OLF Public-Data Baseline Effects"),
        nbf.v4.new_markdown_cell(tldr),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This notebook is an executed reader-facing companion to `scripts/run_baselines.py`. The primary unit is a participant-session-trial. Leave-one-subject-out and within-participant leave-one-session-out splits are assigned before fitted transformations. Odor priors are learned inside each training fold.

### Key Assumptions

- Sampling rate is provisionally 200 Hz, inferred from the reported 15-second stage and 3,000 samples.
- `clean_breathing` is interpreted as post-stimulus recovery, not a pre-stimulus causal baseline.
- Channel order is not embedded in the files, so this first notebook reports channel-aggregated spectral effects and does not make scalp-topography claims.
"""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
OUTPUT_DIR = Path({args.output_dir.as_posix()!r})
manifest = pd.read_csv(OUTPUT_DIR / "manifest.csv")
effects = pd.read_csv(OUTPUT_DIR / "paired_spectral_effects.csv")
reliability = pd.read_csv(OUTPUT_DIR / "cross_session_reliability.csv")
prediction_summary = pd.read_csv(OUTPUT_DIR / "prediction_summary.csv")
incremental_value = pd.read_csv(OUTPUT_DIR / "incremental_value.csv")
metadata = json.loads((OUTPUT_DIR / "run_metadata.json").read_text())
"""
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Confirm trial structure and labels"),
        nbf.v4.new_code_cell(
            """dataset_summary = pd.DataFrame({
    "value": [
        len(manifest), manifest.subject.nunique(), manifest.session.nunique(),
        manifest.trial.nunique(), manifest.y_emotion.mean()
    ]
}, index=["trials", "subjects", "sessions", "trials_per_session", "class_1_rate"])
display(dataset_summary)
display(pd.crosstab(manifest.y_odor, manifest.y_emotion, margins=True))
"""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. Paired stimulation–recovery spectral effects"),
        nbf.v4.new_code_cell(
            """plot_effects = effects[effects.feature_family == "logbp"].copy()
fig, ax = plt.subplots(figsize=(8, 4.5))
positions = np.arange(len(plot_effects))
errors = np.vstack([
    plot_effects.mean_stim_minus_recovery - plot_effects.ci95_lower,
    plot_effects.ci95_upper - plot_effects.mean_stim_minus_recovery,
])
ax.errorbar(positions, plot_effects.mean_stim_minus_recovery, yerr=errors, fmt="o", capsize=4)
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(positions, plot_effects.band)
ax.set_ylabel("Stimulation − recovery log power (dB)")
ax.set_title("Participant-level paired spectral effects with bootstrap 95% CI")
plt.tight_layout()
plt.show()
display(effects.sort_values("wilcoxon_q_value").round(4))
"""
        ),
        nbf.v4.new_markdown_cell("### 3. Cross-session reliability"),
        nbf.v4.new_code_cell(
            """icc_heatmap = reliability[reliability.feature_family == "logbp"].pivot(
    index="odor", columns="band", values="icc2_1"
)
icc_heatmap = icc_heatmap.reindex(columns=["delta", "theta", "alpha", "beta", "gamma"])
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.heatmap(icc_heatmap, annot=True, fmt=".2f", center=0, cmap="vlag", ax=ax)
ax.set_title("Cross-session absolute-agreement ICC of stimulation–recovery log power")
plt.tight_layout()
plt.show()
display(reliability.sort_values("icc2_1", ascending=False).head(15).round(4))
"""
        ),
        nbf.v4.new_markdown_cell("### 4. Leakage-safe prediction baselines"),
        nbf.v4.new_code_cell(
            """overall = prediction_summary[prediction_summary.subset == "all"].copy()
display(overall.sort_values(["protocol", "log_loss"]).round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=overall, x="model", y="log_loss", hue="protocol", ax=axes[0])
sns.barplot(data=overall, x="model", y="balanced_accuracy", hue="protocol", ax=axes[1])
for ax, title in zip(axes, ["Lower is better", "Higher is better"], strict=True):
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Protocol")
plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell("### 5. Odor-prior disagreement trials"),
        nbf.v4.new_code_cell(
            """disagreement = prediction_summary[
    prediction_summary.subset == "odor_prior_disagreement"
].copy()
display(disagreement.sort_values(["protocol", "log_loss"]).round(4))
display(incremental_value.sort_values(["protocol", "subset", "mean_log_loss_improvement"], ascending=[True, True, False]).round(4))
"""
        ),
        nbf.v4.new_markdown_cell(takeaway),
    ]

    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    executed = client.execute()
    nbf.write(executed, args.notebook)
    print(f"Executed notebook written to {args.notebook}")


if __name__ == "__main__":
    main()
