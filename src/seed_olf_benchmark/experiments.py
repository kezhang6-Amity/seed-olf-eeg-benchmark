"""Run the first SEED-OLF public-data effect and prediction benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import ttest_1samp, wilcoxon

from .data import build_manifest
from .features import ASSUMED_SAMPLING_RATE_HZ, BANDS_HZ, extract_feature_table
from .metrics import benjamini_hochberg, bootstrap_icc, prediction_metrics
from .models import prior_predictions, tuned_logistic_predictions
from .validation import outer_folds


RANDOM_STATE = 20260810


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("SEED_OLF_DATA_ROOT"),
        help="Dataset root; defaults to SEED_OLF_DATA_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "artifacts" / "baseline_v1",
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args()


def load_or_build_tables(
    data_root: Path,
    output_dir: Path,
    force_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    feature_path = output_dir / "spectral_features.csv"

    if manifest_path.exists() and not force_features:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_manifest(data_root)
        manifest.to_csv(manifest_path, index=False)

    if feature_path.exists() and not force_features:
        features = pd.read_csv(feature_path)
    else:
        features = extract_feature_table(manifest)
        features.to_csv(feature_path, index=False)
    return manifest, features


def subject_bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 5_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    samples = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def paired_spectral_effects(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ("logbp", "relbp"):
        for band in BANDS_HZ:
            column = f"delta_{family}_{band}_mean"
            subject_values = features.groupby("subject")[column].mean().to_numpy()
            t_result = ttest_1samp(subject_values, popmean=0.0)
            w_result = wilcoxon(subject_values, alternative="two-sided", zero_method="wilcox")
            ci_lower, ci_upper = subject_bootstrap_mean_ci(subject_values)
            rows.append(
                {
                    "feature_family": family,
                    "band": band,
                    "n_subjects": len(subject_values),
                    "mean_stim_minus_recovery": float(subject_values.mean()),
                    "ci95_lower": ci_lower,
                    "ci95_upper": ci_upper,
                    "cohen_dz": float(subject_values.mean() / subject_values.std(ddof=1)),
                    "t_statistic": float(t_result.statistic),
                    "t_p_value": float(t_result.pvalue),
                    "wilcoxon_statistic": float(w_result.statistic),
                    "wilcoxon_p_value": float(w_result.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    result["wilcoxon_q_value"] = benjamini_hochberg(result.wilcoxon_p_value)
    return result


def reliability_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for odor in sorted(features.y_odor.unique()):
        for family in ("logbp", "relbp"):
            for band in BANDS_HZ:
                column = f"delta_{family}_{band}_mean"
                aggregated = (
                    features[features.y_odor == odor]
                    .groupby(["subject", "session"], as_index=False)[column]
                    .mean()
                )
                matrix = (
                    aggregated.pivot(index="subject", columns="session", values=column)
                    .sort_index(axis=0)
                    .sort_index(axis=1)
                    .to_numpy()
                )
                estimate, lower, upper = bootstrap_icc(matrix)
                rows.append(
                    {
                        "odor": int(odor),
                        "feature_family": family,
                        "band": band,
                        "n_subjects": matrix.shape[0],
                        "n_sessions": matrix.shape[1],
                        "icc2_1": estimate,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def model_feature_sets(features: pd.DataFrame) -> dict[str, tuple[list[str], bool]]:
    stim_columns = sorted(
        column
        for column in features
        if column.startswith("stim_") and ("logbp_" in column or "relbp_" in column)
    )
    delta_columns = sorted(
        column
        for column in features
        if column.startswith("delta_") and ("logbp_" in column or "relbp_" in column)
    )
    return {
        "eeg_stim_logit": (stim_columns, False),
        "odor_eeg_stim_logit": (stim_columns, True),
        "odor_eeg_paired_logit": (stim_columns + delta_columns, True),
    }


def prediction_rows_for_fold(
    protocol: str,
    fold_id: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    inner_groups: np.ndarray,
    model_specs: dict[str, tuple[list[str], bool]],
) -> list[dict]:
    predictions: dict[str, tuple[np.ndarray, float | None]] = {
        "global_prior": (prior_predictions(train, test, by_odor=False), None),
        "odor_prior": (prior_predictions(train, test, by_odor=True), None),
    }
    for model_name, (columns, include_odor) in model_specs.items():
        probability, best_c = tuned_logistic_predictions(
            train,
            test,
            columns,
            include_odor,
            inner_groups,
            random_state=RANDOM_STATE,
        )
        predictions[model_name] = (probability, best_c)

    odor_probability = predictions["odor_prior"][0]
    odor_disagreement = test.y_emotion.to_numpy() != (odor_probability >= 0.5)
    identity_columns = ["subject", "session", "trial", "y_emotion", "y_odor"]
    rows = []
    for model_name, (probability, best_c) in predictions.items():
        for record, prediction, disagreement in zip(
            test[identity_columns].to_dict("records"),
            probability,
            odor_disagreement,
            strict=True,
        ):
            rows.append(
                {
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "model": model_name,
                    **record,
                    "probability": float(prediction),
                    "best_c": best_c,
                    "odor_prior_disagreement": bool(disagreement),
                }
            )
    return rows


def run_prediction_benchmarks(features: pd.DataFrame) -> pd.DataFrame:
    model_specs = model_feature_sets(features)
    rows = []
    for protocol in ("loso", "losession"):
        folds = list(outer_folds(features, protocol))
        for index, (fold_id, train, test, groups) in enumerate(folds, start=1):
            rows.extend(
                prediction_rows_for_fold(
                    protocol,
                    fold_id,
                    train,
                    test,
                    groups,
                    model_specs,
                )
            )
            print(f"{protocol}: completed {index}/{len(folds)} outer folds", flush=True)
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, model), part in predictions.groupby(["protocol", "model"]):
        for subset_name, subset in (
            ("all", part),
            ("odor_prior_disagreement", part[part.odor_prior_disagreement]),
        ):
            if subset.empty:
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "subset": subset_name,
                    **prediction_metrics(subset.y_emotion, subset.probability),
                }
            )
    return pd.DataFrame(rows)


def incremental_value_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare each EEG model with odor prior using participant-block bootstrap."""

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for protocol, protocol_data in predictions.groupby("protocol"):
        for subset_name, subset in (
            ("all", protocol_data),
            (
                "odor_prior_disagreement",
                protocol_data[protocol_data.odor_prior_disagreement],
            ),
        ):
            key_columns = ["subject", "session", "trial"]
            wide = subset.pivot(index=key_columns, columns="model", values="probability")
            labels = (
                subset.drop_duplicates(key_columns)
                .set_index(key_columns)
                .loc[wide.index, "y_emotion"]
                .to_numpy(dtype=int)
            )
            baseline_probability = np.clip(wide["odor_prior"].to_numpy(), 1e-6, 1 - 1e-6)
            baseline_loss = -(
                labels * np.log(baseline_probability)
                + (1 - labels) * np.log1p(-baseline_probability)
            )
            subjects = wide.index.get_level_values("subject").to_numpy()
            unique_subjects = np.unique(subjects)
            for model in sorted(column for column in wide if column not in {"odor_prior", "global_prior"}):
                model_probability = np.clip(wide[model].to_numpy(), 1e-6, 1 - 1e-6)
                model_loss = -(
                    labels * np.log(model_probability)
                    + (1 - labels) * np.log1p(-model_probability)
                )
                trial_improvement = baseline_loss - model_loss
                subject_improvement = np.array(
                    [trial_improvement[subjects == subject].mean() for subject in unique_subjects]
                )
                bootstrap = subject_improvement[
                    rng.integers(
                        0,
                        len(subject_improvement),
                        size=(10_000, len(subject_improvement)),
                    )
                ].mean(axis=1)
                lower, upper = np.quantile(bootstrap, [0.025, 0.975])
                rows.append(
                    {
                        "protocol": protocol,
                        "subset": subset_name,
                        "model": model,
                        "baseline": "odor_prior",
                        "n_subjects": len(unique_subjects),
                        "n_trials": len(wide),
                        "mean_log_loss_improvement": float(subject_improvement.mean()),
                        "ci95_lower": float(lower),
                        "ci95_upper": float(upper),
                        "subjects_improved_rate": float(np.mean(subject_improvement > 0)),
                    }
                )
    return pd.DataFrame(rows)


def write_run_metadata(
    output_dir: Path,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    metadata = {
        "sampling_rate_hz": ASSUMED_SAMPLING_RATE_HZ,
        "sampling_rate_status": "assumed_from_3000_samples_over_15_seconds",
        "random_state": RANDOM_STATE,
        "n_trials": len(manifest),
        "n_subjects": int(manifest.subject.nunique()),
        "n_sessions_per_subject": int(manifest.groupby("subject").session.nunique().min()),
        "feature_columns": int(len(features.columns)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.data_root is None:
        raise SystemExit(
            "Dataset path required: pass --data-root or set SEED_OLF_DATA_ROOT."
        )
    manifest, features = load_or_build_tables(
        Path(args.data_root).expanduser().resolve(),
        args.output_dir,
        args.force_features,
    )
    effects = paired_spectral_effects(features)
    reliability = reliability_table(features)
    effects.to_csv(args.output_dir / "paired_spectral_effects.csv", index=False)
    reliability.to_csv(args.output_dir / "cross_session_reliability.csv", index=False)
    write_run_metadata(args.output_dir, manifest, features)

    if not args.skip_models:
        predictions = run_prediction_benchmarks(features)
        predictions.to_csv(args.output_dir / "trial_predictions.csv", index=False)
        summarize_predictions(predictions).to_csv(
            args.output_dir / "prediction_summary.csv",
            index=False,
        )
        incremental_value_table(predictions).to_csv(
            args.output_dir / "incremental_value.csv",
            index=False,
        )
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
