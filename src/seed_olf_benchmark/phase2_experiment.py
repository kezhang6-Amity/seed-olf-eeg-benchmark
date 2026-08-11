"""Execute the preregistered odor-controlled channel-level Phase 2 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from .phase2_features import (
    ASSUMED_SAMPLING_RATE_HZ,
    channel_feature_names,
    load_or_build_channel_features,
)
from .phase2_metrics import (
    comparison_table,
    evaluate_success_gate,
    summarize_phase2_predictions,
)
from .phase2_models import (
    C_GRID,
    RANDOM_STATE,
    prior_predictions_with_target,
    tuned_odor_eeg_predictions,
)
from .phase2_validation import verify_prediction_coverage
from .validation import outer_folds


MODEL_SPECS = {
    "M1": ([], False),
    "M2": (channel_feature_names("stim_logbp"), False),
    "M3": (channel_feature_names("stim_de"), False),
    "M4": (channel_feature_names("delta_logbp"), False),
    "M5": (channel_feature_names("stim_logbp"), True),
}
ALL_MODELS = ["M0", *MODEL_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=os.environ.get("SEED_OLF_DATA_ROOT")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts") / "phase2_channel"
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--skip-negative-controls", action="store_true")
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument(
        "--config", type=Path, default=Path("configs") / "phase2.toml"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    return parser.parse_args()


def validate_declared_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    expected = {
        "random_state": RANDOM_STATE,
        "sampling_rate_hz": ASSUMED_SAMPLING_RATE_HZ,
        "models": ALL_MODELS,
        "c_grid": list(C_GRID),
        "solver": "lbfgs",
        "max_iter": 5000,
        "bootstrap_replicates": 10_000,
        "negative_control_permutations": 20,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase 2 config/code mismatch: {mismatches}")
    return config


def permute_target_within_subject_odor(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    target = frame.y_emotion.to_numpy(dtype=int).copy()
    for positions in frame.groupby(["subject", "y_odor"], sort=True).indices.values():
        target[positions] = rng.permutation(target[positions])
    return target


def prediction_rows_for_fold(
    protocol: str,
    fold_id: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    models: list[str],
    target: np.ndarray | None = None,
) -> list[dict]:
    predictions: dict[str, tuple[np.ndarray, float | None, float | None]] = {
        "M0": (prior_predictions_with_target(train, test, target), None, None)
    }
    for model_name in models:
        columns, robust = MODEL_SPECS[model_name]
        result = tuned_odor_eeg_predictions(
            train,
            test,
            protocol,
            columns,
            robust=robust,
            target=target,
        )
        predictions[model_name] = (
            result.probability,
            result.best_c,
            result.inner_log_loss,
        )
    identity = ["subject", "session", "trial", "y_emotion", "y_odor"]
    rows = []
    for model_name, (probability, best_c, inner_loss) in predictions.items():
        for record, value in zip(
            test[identity].to_dict("records"), probability, strict=True
        ):
            rows.append(
                {
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "model": model_name,
                    **record,
                    "probability": float(value),
                    "best_c": best_c,
                    "inner_log_loss": inner_loss,
                }
            )
    return rows


def run_predictions(
    features: pd.DataFrame,
    models: list[str],
    progress_label: str = "real labels",
    target_permutation_seed: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(target_permutation_seed)
    for protocol in ("loso", "losession"):
        folds = list(outer_folds(features, protocol))
        for index, (fold_id, train, test, _) in enumerate(folds, start=1):
            target = None
            if target_permutation_seed is not None:
                target = permute_target_within_subject_odor(train, rng)
            rows.extend(
                prediction_rows_for_fold(
                    protocol, fold_id, train, test, models, target=target
                )
            )
            if verbose:
                print(
                    f"{progress_label} {protocol}: {index}/{len(folds)} outer folds",
                    flush=True,
                )
    return pd.DataFrame(rows)


def run_negative_controls(
    features: pd.DataFrame,
    permutations: int,
    n_bootstrap: int,
    output_dir: Path,
) -> pd.DataFrame:
    def run_one(permutation: int) -> tuple[pd.DataFrame, dict]:
        seed = RANDOM_STATE + permutation + 1
        predictions = run_predictions(
            features,
            ["M2"],
            progress_label=f"permutation {permutation + 1}/{permutations}",
            target_permutation_seed=seed,
            verbose=False,
        )
        predictions.insert(0, "permutation", permutation + 1)
        comparisons = comparison_table(
            predictions, ["M2"], n_bootstrap=n_bootstrap
        )
        classification, checks = evaluate_success_gate(comparisons, audits_passed=True)
        loso = comparisons[comparisons.protocol == "loso"].iloc[0]
        summary = {
            "permutation": permutation + 1,
            "seed": seed,
            "classification": classification,
            "all_statistical_gates_pass": all(
                value for key, value in checks.items() if key != "audits_passed"
            ),
            "loso_log_loss_improvement": loso.mean_log_loss_improvement,
            "loso_log_loss_ci95_lower": loso.log_loss_ci95_lower,
        }
        return predictions, summary

    workers = min(4, permutations)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        completed = list(executor.map(run_one, range(permutations)))
    prediction_parts = [item[0] for item in completed]
    summaries = [item[1] for item in completed]
    print(f"Completed {permutations} negative-control permutations", flush=True)
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        output_dir / "negative_control_predictions.csv", index=False
    )
    return pd.DataFrame(summaries)


def feature_equivalence_summary(features: pd.DataFrame) -> pd.DataFrame:
    log_columns = channel_feature_names("stim_logbp")
    de_columns = channel_feature_names("stim_de")
    rows = []
    for subject in sorted(features.subject.unique()):
        training = features[features.subject != subject]
        x = training[log_columns].to_numpy(dtype=np.float64)
        y = training[de_columns].to_numpy(dtype=np.float64)
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
        correlations = np.divide(
            np.sum(x * y, axis=0),
            denominator,
            out=np.full(x.shape[1], np.nan),
            where=denominator > 0,
        )
        rows.append(
            {
                "outer_test_subject": subject,
                "median_correlation": float(np.nanmedian(correlations)),
                "minimum_correlation": float(np.nanmin(correlations)),
                "maximum_correlation": float(np.nanmax(correlations)),
                "proportion_abs_correlation_ge_0_95": float(
                    np.nanmean(np.abs(correlations) >= 0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_run_metadata(output_dir: Path, classification: str, checks: dict) -> None:
    metadata = {
        "experiment": "phase2_channel",
        "classification": classification,
        "success_gate": checks,
        "sampling_rate_hz": ASSUMED_SAMPLING_RATE_HZ,
        "sampling_rate_status": "assumed_from_3000_samples_over_reported_15_seconds",
        "channel_identity_status": "array_indices_only_ch00_to_ch61",
        "random_state": RANDOM_STATE,
        "c_grid": list(C_GRID),
        "solver": "lbfgs",
        "max_iter": 5000,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = validate_declared_config(args.config)
    if args.bootstrap != config["bootstrap_replicates"]:
        raise ValueError("Declared run requires 10,000 bootstrap replicates")
    if not args.skip_negative_controls and args.permutations != config["negative_control_permutations"]:
        raise ValueError("Declared run requires 20 negative-control permutations")
    if args.data_root is None:
        raise SystemExit("Pass --data-root or set SEED_OLF_DATA_ROOT")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, features, profile = load_or_build_channel_features(
        args.data_root.expanduser().resolve(),
        args.output_dir,
        force=args.force_features,
    )
    if profile["finite_feature_rate"] != 1.0 or profile["unique_trial_keys"] != len(manifest):
        raise RuntimeError("Data-quality gate failed before modeling")

    expected_models = ["M0", *args.models]
    prediction_path = args.output_dir / "trial_predictions.csv"
    if args.reuse_predictions:
        predictions = pd.read_csv(prediction_path)
    else:
        predictions = run_predictions(features, args.models)
    verify_prediction_coverage(
        predictions, manifest, ["loso", "losession"], expected_models
    )
    predictions.to_csv(prediction_path, index=False)
    summarize_phase2_predictions(predictions).to_csv(
        args.output_dir / "prediction_summary.csv", index=False
    )
    comparisons = comparison_table(
        predictions, args.models, n_bootstrap=args.bootstrap
    )
    comparisons.to_csv(args.output_dir / "incremental_value.csv", index=False)
    feature_equivalence_summary(features).to_csv(
        args.output_dir / "logbp_de_equivalence.csv", index=False
    )

    audits_passed = False
    negative_summary = None
    if not args.skip_negative_controls:
        negative_summary = run_negative_controls(
            features, args.permutations, args.bootstrap, args.output_dir
        )
        negative_summary.to_csv(
            args.output_dir / "negative_control_summary.csv", index=False
        )
        audits_passed = bool(
            not negative_summary.all_statistical_gates_pass.any()
            and negative_summary.loso_log_loss_improvement.median() <= 0.0
        )
    classification, checks = evaluate_success_gate(comparisons, audits_passed)
    write_run_metadata(args.output_dir, classification, checks)
    print(f"Phase 2 classification: {classification}")


if __name__ == "__main__":
    main()
