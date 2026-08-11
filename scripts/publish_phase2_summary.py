"""Publish compact, non-participant-level Phase 2 research summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts") / "phase2_channel"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results") / "phase2_channel"
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports") / "phase2_channel.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_summary = pd.read_csv(args.output_dir / "prediction_summary.csv")
    incremental = pd.read_csv(args.output_dir / "incremental_value.csv")
    negative = pd.read_csv(args.output_dir / "negative_control_summary.csv")
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text())
    validation = json.loads((args.output_dir / "validation_checks.json").read_text())
    quality = json.loads((args.output_dir / "data_quality_profile.json").read_text())

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary.to_csv(args.results_dir / "prediction_summary.csv", index=False)
    incremental.to_csv(args.results_dir / "incremental_value.csv", index=False)
    negative.to_csv(args.results_dir / "negative_control_summary.csv", index=False)
    (args.results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.results_dir / "validation_checks.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    public_quality = {
        key: quality[key]
        for key in (
            "n_trials",
            "n_subjects",
            "n_sessions",
            "unique_trial_keys",
            "phase_pair_coverage",
            "feature_count",
            "finite_feature_rate",
            "zero_variance_features",
            "signal_qc_by_phase",
        )
    }
    (args.results_dir / "data_quality_summary.json").write_text(
        json.dumps(public_quality, indent=2), encoding="utf-8"
    )

    loso = prediction_summary[prediction_summary.protocol == "loso"].set_index("model")
    losession = prediction_summary[
        prediction_summary.protocol == "losession"
    ].set_index("model")
    m2 = incremental[(incremental.protocol == "loso") & (incremental.candidate == "M2")].iloc[0]
    m5 = incremental[(incremental.protocol == "loso") & (incremental.candidate == "M5")].iloc[0]
    report = f"""# Phase 2: Odor-Controlled Channel Features

## Decision

**Classification: {metadata['classification']}.** Channel-resolved EEG features did not demonstrate reliable incremental probabilistic value beyond odor identity for the public SEED-OLF binary subjective-valence proxy. They must not be described as evidence of relaxation, clinical benefit, or a validated personalized-response model.

## Design

- 2,304 participant-session-trials from 32 participants across three sessions.
- Primary model M2: odor indicators plus training-fold odor-residualized stimulation log band power (62 channels × 5 bands).
- Comparators: M0 odor prior, M1 odor-only logistic regression, M3 differential entropy, M4 stimulation-minus-recovery log power, and M5 robust M2.
- Primary protocol: leave-one-subject-out (LOSO); secondary protocol: within-person leave-one-session-out.
- Every fitted transformation, hyperparameter choice, and odor residualizer was restricted to its training fold.
- 20 negative controls permuted labels within `(subject, odor)` in each outer training fold.

## Data-quality and audit status

- Feature cache: {public_quality['feature_count']:,} features, finite rate {public_quality['finite_feature_rate']:.1%}, zero-variance features {public_quality['zero_variance_features']}.
- Differential-entropy floor activation: 0.0% in stimulation and recovery after calculating variance in microvolt squared.
- Independent metric recomputation difference: {validation['maximum_metric_recompute_difference']:.2e}.
- Negative-control gate: passed; 0/{len(negative)} permuted runs passed all statistical gates, and median LOSO M2 improvement was {negative.loso_log_loss_improvement.median():.4f}.

## Main results

| Protocol | Model | Log loss ↓ | Brier ↓ | Balanced accuracy ↑ | AUROC ↑ |
|---|---|---:|---:|---:|---:|
| LOSO | M0 odor prior | {loso.loc['M0', 'log_loss']:.4f} | {loso.loc['M0', 'brier']:.4f} | {loso.loc['M0', 'balanced_accuracy']:.4f} | {loso.loc['M0', 'roc_auc']:.4f} |
| LOSO | M2 primary log-power | {loso.loc['M2', 'log_loss']:.4f} | {loso.loc['M2', 'brier']:.4f} | {loso.loc['M2', 'balanced_accuracy']:.4f} | {loso.loc['M2', 'roc_auc']:.4f} |
| LOSO | M5 robust log-power | {loso.loc['M5', 'log_loss']:.4f} | {loso.loc['M5', 'brier']:.4f} | {loso.loc['M5', 'balanced_accuracy']:.4f} | {loso.loc['M5', 'roc_auc']:.4f} |
| Leave-session-out | M0 odor prior | {losession.loc['M0', 'log_loss']:.4f} | {losession.loc['M0', 'brier']:.4f} | {losession.loc['M0', 'balanced_accuracy']:.4f} | {losession.loc['M0', 'roc_auc']:.4f} |
| Leave-session-out | M5 robust log-power | {losession.loc['M5', 'log_loss']:.4f} | {losession.loc['M5', 'brier']:.4f} | {losession.loc['M5', 'balanced_accuracy']:.4f} | {losession.loc['M5', 'roc_auc']:.4f} |

M2's LOSO mean paired log-loss improvement over M0 was **{m2.mean_log_loss_improvement:.4f}** (95% CI **{m2.log_loss_ci95_lower:.4f} to {m2.log_loss_ci95_upper:.4f}**); positive favors EEG. M5 was less harmful but still negative at **{m5.mean_log_loss_improvement:.4f}** (95% CI **{m5.log_loss_ci95_lower:.4f} to {m5.log_loss_ci95_upper:.4f}**). All four delete-one-odor checks were negative for M2 and M5.

## Interpretation

The channel models increased LOSO AUROC but worsened log loss and Brier score. This is a ranking-versus-calibration conflict: the models sometimes rank trials better, but their probabilities are less trustworthy than odor-only predictions. The cross-session collapse is stronger evidence against using this representation for immediate few-shot personalization.

The next justified experiment is the preregistered Phase 3 calibration-curve baseline, retaining the odor baseline and failed EEG representations as transparent comparators. EEGNet and TSception do not enter until that experiment identifies an error mode that shallow models leave unresolved.

## Boundaries

- 200 Hz remains an engineering assumption inferred from sample count and stage duration.
- Channels are indexed `ch00`–`ch61`; montage, locations, and reference are not verified.
- `clean_breathing` is post-stimulus recovery, not a true prestimulus baseline.
- `y_emotion` is a public binary subjective-valence proxy, not relaxation, stress, or a clinical endpoint.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Published compact summaries to {args.results_dir} and {args.report}")


if __name__ == "__main__":
    main()
