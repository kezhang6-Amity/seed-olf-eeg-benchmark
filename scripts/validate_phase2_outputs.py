"""Independent integrity and headline-metric validation for Phase 2 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from seed_olf_benchmark.phase2_experiment import ALL_MODELS  # noqa: E402
from seed_olf_benchmark.phase2_validation import verify_prediction_coverage  # noqa: E402


TRIAL_KEY = ["subject", "session", "trial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts") / "phase2_channel"
    )
    return parser.parse_args()


def independent_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "brier": float(np.mean((y - p) ** 2)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def independent_improvement(
    predictions: pd.DataFrame, protocol: str, candidate: str
) -> tuple[float, float]:
    part = predictions[
        (predictions.protocol == protocol)
        & predictions.model.isin(["M0", candidate])
    ]
    wide = part.pivot(index=TRIAL_KEY, columns="model", values="probability")
    y = (
        part.drop_duplicates(TRIAL_KEY)
        .set_index(TRIAL_KEY)
        .loc[wide.index, "y_emotion"]
        .to_numpy(dtype=int)
    )
    p0 = np.clip(wide.M0.to_numpy(), 1e-6, 1 - 1e-6)
    candidate_probability = np.clip(
        wide[candidate].to_numpy(), 1e-6, 1 - 1e-6
    )
    loss0 = -(y * np.log(p0) + (1 - y) * np.log1p(-p0))
    loss_candidate = -(
        y * np.log(candidate_probability)
        + (1 - y) * np.log1p(-candidate_probability)
    )
    brier0 = (y - p0) ** 2
    brier_candidate = (y - candidate_probability) ** 2
    return float(np.mean(loss0 - loss_candidate)), float(
        np.mean(brier0 - brier_candidate)
    )


def main() -> None:
    output_dir = parse_args().output_dir
    manifest = pd.read_csv(output_dir / "manifest.csv")
    predictions = pd.read_csv(output_dir / "trial_predictions.csv")
    summary = pd.read_csv(output_dir / "prediction_summary.csv")
    incremental = pd.read_csv(output_dir / "incremental_value.csv")
    profile = json.loads(
        (output_dir / "data_quality_profile.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    cache_metadata = json.loads(
        (output_dir / "channel_feature_cache.json").read_text(encoding="utf-8")
    )

    assert len(manifest) == 2_304
    assert not manifest.duplicated(TRIAL_KEY).any()
    assert manifest.y_emotion.value_counts().to_dict() == {0: 1_306, 1: 998}
    assert manifest.y_odor.value_counts().to_dict() == {1: 576, 2: 576, 3: 576, 4: 576}
    assert profile["phase_pair_coverage"] == 1.0
    assert profile["finite_feature_rate"] == 1.0
    assert profile["zero_variance_features"] == 0
    assert profile["signal_qc_by_phase"]["stim"]["de_floor_value_rate"] == 0.0
    assert profile["signal_qc_by_phase"]["recovery"]["de_floor_value_rate"] == 0.0
    assert cache_metadata["schema_version"] == 2
    assert cache_metadata["feature_count"] == 2_480

    with np.load(output_dir / "channel_features.npz", allow_pickle=False) as cache:
        assert cache["values"].shape == (2_304, 2_480)
        assert np.isfinite(cache["values"]).all()
        assert len(cache["columns"]) == 2_480

    verify_prediction_coverage(
        predictions, manifest, ["loso", "losession"], ALL_MODELS
    )
    loso_expected = predictions.loc[predictions.protocol == "loso", "subject"].map(
        lambda subject: f"subject_{subject:02d}"
    )
    assert np.array_equal(
        predictions.loc[predictions.protocol == "loso", "fold_id"], loso_expected
    )
    losession = predictions[predictions.protocol == "losession"]
    losession_expected = losession.apply(
        lambda row: f"subject_{row.subject:02d}_session_{row.session}", axis=1
    )
    assert np.array_equal(losession.fold_id, losession_expected)

    recomputed_rows = []
    for (protocol, model), part in predictions.groupby(["protocol", "model"]):
        recomputed_rows.append(
            {
                "protocol": protocol,
                "model": model,
                **independent_metrics(
                    part.y_emotion.to_numpy(), part.probability.to_numpy()
                ),
            }
        )
    recomputed = pd.DataFrame(recomputed_rows)
    merged = recomputed.merge(
        summary,
        on=["protocol", "model"],
        suffixes=("_recomputed", "_reported"),
        validate="one_to_one",
    )
    metric_names = ["log_loss", "brier", "balanced_accuracy", "roc_auc", "pr_auc"]
    maximum_metric_difference = max(
        float(
            np.max(
                np.abs(
                    merged[f"{metric}_recomputed"]
                    - merged[f"{metric}_reported"]
                )
            )
        )
        for metric in metric_names
    )
    assert maximum_metric_difference < 1e-12

    maximum_improvement_difference = 0.0
    for row in incremental.itertuples(index=False):
        log_improvement, brier_improvement = independent_improvement(
            predictions, row.protocol, row.candidate
        )
        maximum_improvement_difference = max(
            maximum_improvement_difference,
            abs(log_improvement - row.mean_log_loss_improvement),
            abs(brier_improvement - row.mean_brier_improvement),
        )
        assert row.log_loss_ci95_lower <= row.mean_log_loss_improvement <= row.log_loss_ci95_upper
        assert row.brier_ci95_lower <= row.mean_brier_improvement <= row.brier_ci95_upper
    assert maximum_improvement_difference < 1e-12

    negative_control_passed = False
    negative_path = output_dir / "negative_control_summary.csv"
    if negative_path.exists():
        negative = pd.read_csv(negative_path)
        negative_control_passed = bool(
            len(negative) == 20
            and not negative.all_statistical_gates_pass.any()
            and negative.loso_log_loss_improvement.median() <= 0.0
        )
    assert negative_control_passed
    assert metadata["success_gate"]["audits_passed"]

    checks = {
        "status": "passed",
        "n_trials": len(manifest),
        "n_prediction_rows": len(predictions),
        "feature_shape": [2_304, 2_480],
        "models": ALL_MODELS,
        "protocols": ["loso", "losession"],
        "maximum_metric_recompute_difference": maximum_metric_difference,
        "maximum_improvement_recompute_difference": maximum_improvement_difference,
        "negative_control_passed": negative_control_passed,
        "sampling_rate_metadata_verified": False,
        "channel_montage_verified": False,
        "required_caveat": (
            "200 Hz remains assumed; channels are indices only; y_emotion is a proxy target."
        ),
    }
    (output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
