"""Independently validate Phase 3 few-shot output integrity and headline metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seed_olf_benchmark.phase3 import CALIBRATION_SIZES, verify_phase3_prediction_coverage  # noqa: E402
from seed_olf_benchmark.phase3_experiment import evaluate_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "phase3_fewshot")
    parser.add_argument("--phase2-dir", type=Path, default=Path("artifacts") / "phase2_channel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.phase2_dir / "manifest.csv")
    predictions = pd.read_csv(args.output_dir / "trial_predictions.csv")
    assignments = pd.read_csv(args.output_dir / "calibration_assignments.csv")
    summary = pd.read_csv(args.output_dir / "prediction_summary.csv")
    incremental = pd.read_csv(args.output_dir / "incremental_value.csv")
    negative = pd.read_csv(args.output_dir / "negative_control_summary.csv")
    metadata = json.loads((args.output_dir / "run_metadata.json").read_text())

    models = sorted(predictions.model.unique())
    assert models == ["P0", "P1", "P2", "P4"]
    for subject in sorted(manifest.subject.unique()):
        expected = manifest[(manifest.subject == subject) & manifest.session.isin([2, 3])]
        verify_phase3_prediction_coverage(
            predictions[predictions.subject == subject], expected, list(CALIBRATION_SIZES), models
        )
    assignment_counts = assignments.groupby(
        ["subject", "calibration_size", "replicate", "y_odor"]
    ).size()
    for size in CALIBRATION_SIZES[1:]:
        assert assignment_counts.loc[
            assignment_counts.index.get_level_values("calibration_size") == size
        ].eq(size // 4).all()
    assert set(assignments.session) == {1}

    recomputed = []
    for (size, model), part in predictions.groupby(["calibration_size", "model"]):
        y = part.y_emotion.to_numpy(dtype=int)
        p = np.clip(part.probability.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        recomputed.append({
            "calibration_size": size,
            "model": model,
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
            "brier": float(np.mean((y - p) ** 2)),
        })
    check = pd.DataFrame(recomputed).merge(summary, on=["calibration_size", "model"])
    maximum_metric_difference = float(np.max(np.abs(check.log_loss_x - check.log_loss_y)))
    maximum_metric_difference = max(maximum_metric_difference, float(np.max(np.abs(check.brier_x - check.brier_y))))
    assert maximum_metric_difference < 1e-12

    classification, gate = evaluate_gate(incremental, audits_passed=True, negative_controls=negative)
    assert classification == "passed"
    assert gate == metadata["success_gate"]
    assert len(negative) == 100
    assert not negative.all_statistical_gates_pass.any()
    assert all(
        incremental.loc[(incremental.candidate == "P2") & (incremental.calibration_size == size), "mean_log_loss_improvement"].iloc[0]
        > negative.loc[negative.calibration_size == size, "mean_log_loss_improvement"].max()
        for size in CALIBRATION_SIZES[1:]
    )

    checks = {
        "status": "passed",
        "n_subjects": int(manifest.subject.nunique()),
        "n_trial_prediction_rows": int(len(predictions)),
        "models": models,
        "minimum_feasible_size": gate["minimum_feasible_size"],
        "maximum_metric_recompute_difference": maximum_metric_difference,
        "negative_control_permutations": int(negative.permutation.nunique()),
    }
    (args.output_dir / "validation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
