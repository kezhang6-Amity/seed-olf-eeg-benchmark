"""Publish compact, non-participant-level Phase 3 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "phase3_fewshot")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "phase3_fewshot")
    parser.add_argument("--report", type=Path, default=Path("reports") / "phase3_fewshot.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.output_dir / "prediction_summary.csv")
    incremental = pd.read_csv(args.output_dir / "incremental_value.csv")
    negative = pd.read_csv(args.output_dir / "negative_control_summary.csv")
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text())
    validation = json.loads((args.output_dir / "validation_checks.json").read_text())
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("prediction_summary.csv", summary), ("incremental_value.csv", incremental), ("negative_control_summary.csv", negative)):
        frame.to_csv(args.results_dir / name, index=False)
    for name, value in (("run_metadata.json", metadata), ("validation_checks.json", validation)):
        (args.results_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    p2 = incremental[incremental.candidate == "P2"].set_index("calibration_size")
    p1 = incremental[incremental.candidate == "P1"].set_index("calibration_size")
    rows = []
    for size in (4, 8, 16, 24):
        null_max = negative[negative.calibration_size == size].mean_log_loss_improvement.max()
        rows.append(
            f"| {size} | {p2.loc[size, 'mean_log_loss_improvement']:.6f} | "
            f"[{p2.loc[size, 'log_loss_ci95_lower']:.6f}, {p2.loc[size, 'log_loss_ci95_upper']:.6f}] | "
            f"{p2.loc[size, 'mean_brier_improvement']:.6f} | {p2.loc[size, 'participants_improved_rate']:.1%} | {null_max:.6f} |"
        )
    report = f"""# Phase 3: Few-Shot Cross-Session Personalization

## Decision

**Classification: passed.** The minimum feasible calibration burden is **4 Session-1 trials per participant**—one trial per odor. This result supports a narrowly scoped claim: a pooled odor-response prior can be improved slightly and reproducibly by updating it with a participant's own labeled trials.

## Design

- 32 participants; calibration uses Session 1 only and prediction evaluates all 48 trials in Sessions 2–3.
- Calibration sizes: 0, 4, 8, 16, and 24; sizes 4–16 use 200 deterministic label-blind balanced subsets.
- Primary P2 model: leave-target-participant-out odor-specific Beta prior, updated by target Session-1 labels.
- P1 direct individual frequency and P4 fixed Phase-2 M5 EEG predictions are secondary comparators. P3 penalized odor logistic regression is implemented but not part of this completed primary run.
- Uncertainty: paired hierarchical bootstrap (participants → calibration subsets → held-out trials), 10,000 replicates.
- Negative control: 20 within-participant Session-1 label permutations across odors. No permuted run passed the statistical gate, and observed P2 improvement exceeded the corresponding permutation maximum at every nonzero size.

## Primary P2 results

| Calibration trials | Log-loss improvement over P0 ↑ | 95% CI | Brier improvement ↑ | Participants improved | Maximum permuted improvement |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

At 4 trials, the effect is statistically stable but small: Δlog-loss {p2.loc[4, 'mean_log_loss_improvement']:.6f}, 95% CI [{p2.loc[4, 'log_loss_ci95_lower']:.6f}, {p2.loc[4, 'log_loss_ci95_upper']:.6f}], with {p2.loc[4, 'participants_improved_rate']:.1%} of participants improved. The effect grows monotonically through 24 trials, but the effect size remains modest.

## Comparator findings

Direct individual frequencies were harmful at every calibration size (at 4 trials, Δlog-loss {p1.loc[4, 'mean_log_loss_improvement']:.4f}), demonstrating why pooled shrinkage is required. The fixed Phase-2 EEG comparator P4 was worse than the odor prior at all sizes (Δlog-loss {incremental[(incremental.candidate == 'P4') & (incremental.calibration_size == 4)].iloc[0].mean_log_loss_improvement:.4f}); the current EEG representation should not be used for this personalization task.

## Audit status and boundaries

- Independent recomputation maximum discrepancy: {validation['maximum_metric_recompute_difference']:.2e}.
- {validation['negative_control_permutations']} negative-control permutations completed; none passed the statistical gate.
- This does not establish relaxation, clinical benefit, product efficacy, or an EEG-driven treatment effect. `y_emotion` remains a public binary subjective-valence proxy.
- The result is limited to SEED-OLF's odor categories, sessions, labels, and the declared cross-session split. External replication is required before a collection protocol is finalized.

## Implication for future collection

For the first proprietary study, retain a randomized, odor-balanced Session-1 calibration block of at least four labeled trials per participant. Use the P2 posterior as the transparent operational baseline. Add EEG + PPG/HRV + EDA only as prospectively evaluated incremental modalities; do not assume they will improve this baseline.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Published compact summaries to {args.results_dir} and {args.report}")


if __name__ == "__main__":
    main()
