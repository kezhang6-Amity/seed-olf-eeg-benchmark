"""Independent integrity and metric checks for the baseline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from seed_olf_benchmark.metrics import prediction_metrics


EXPECTED_MODELS = {
    "global_prior",
    "odor_prior",
    "eeg_stim_logit",
    "odor_eeg_stim_logit",
    "odor_eeg_paired_logit",
    "riemann_stim_logit",
    "odor_riemann_stim_logit",
    "odor_riemann_relative_logit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "baseline_v1",
    )
    return parser.parse_args()


def main() -> None:
    output_dir = parse_args().output_dir
    manifest = pd.read_csv(output_dir / "manifest.csv")
    features = pd.read_csv(output_dir / "spectral_features.csv")
    predictions = pd.read_csv(output_dir / "trial_predictions.csv")
    summary = pd.read_csv(output_dir / "prediction_summary.csv")
    effects = pd.read_csv(output_dir / "paired_spectral_effects.csv")
    reliability = pd.read_csv(output_dir / "cross_session_reliability.csv")

    trial_key = ["subject", "session", "trial"]
    assert len(manifest) == 2_304
    assert not manifest.duplicated(trial_key).any()
    assert manifest.y_emotion.value_counts().to_dict() == {0: 1_306, 1: 998}
    assert manifest.y_odor.value_counts().to_dict() == {1: 576, 2: 576, 3: 576, 4: 576}
    assert len(features) == len(manifest)
    assert not features.isna().any().any()
    assert set(predictions.model) == EXPECTED_MODELS
    assert np.isfinite(predictions.probability).all()
    assert predictions.probability.between(0, 1).all()
    assert not predictions.duplicated(["protocol", "model", *trial_key]).any()
    counts = predictions.groupby(["protocol", "model"]).size()
    assert (counts == 2_304).all()
    assert len(effects) == 10
    assert len(reliability) == 40

    loso = predictions[predictions.protocol == "loso"]
    expected_loso_fold = loso.subject.map(lambda value: f"subject_{value:02d}")
    assert (loso.fold_id == expected_loso_fold).all()
    losession = predictions[predictions.protocol == "losession"]
    expected_session_fold = losession.apply(
        lambda row: f"subject_{row.subject:02d}_session_{row.session}",
        axis=1,
    )
    assert (losession.fold_id == expected_session_fold).all()

    recomputed = []
    for (protocol, model), part in predictions.groupby(["protocol", "model"]):
        recomputed.append(
            {
                "protocol": protocol,
                "model": model,
                **prediction_metrics(part.y_emotion, part.probability),
            }
        )
    recomputed_frame = pd.DataFrame(recomputed)
    reported = summary[summary.subset == "all"].drop(columns=["subset"])
    merged = recomputed_frame.merge(
        reported,
        on=["protocol", "model"],
        suffixes=("_recomputed", "_reported"),
        validate="one_to_one",
    )
    metric_columns = ["log_loss", "brier", "balanced_accuracy", "roc_auc", "pr_auc"]
    maximum_difference = max(
        float(np.max(np.abs(merged[f"{name}_recomputed"] - merged[f"{name}_reported"])))
        for name in metric_columns
    )
    assert maximum_difference < 1e-12

    checks = {
        "status": "passed",
        "n_trials": len(manifest),
        "n_prediction_rows": len(predictions),
        "models": sorted(EXPECTED_MODELS),
        "protocols": sorted(predictions.protocol.unique()),
        "maximum_metric_recompute_difference": maximum_difference,
        "sampling_rate_metadata_verified": False,
        "required_caveat": "200 Hz remains inferred; clean breathing is post-stimulus recovery.",
    }
    (output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
