"""Execute Phase 3 few-shot cross-session personalization validation."""

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

from .phase2_metrics import phase2_prediction_metrics
from .phase2_validation import compute_source_fingerprint
from .phase3 import (
    CALIBRATION_SIZES,
    MODELS,
    RANDOM_STATE,
    balanced_calibration_subsets,
    direct_individual_probabilities,
    empirical_bayes_probabilities,
    fit_pooled_logistic,
    phase3_hierarchical_bootstrap,
    pooled_odor_prior,
    pooled_plus_target_logistic_probabilities,
    verify_phase3_prediction_coverage,
)


N_SAMPLES = 200
N_PERMUTATIONS = 20
N_BOOTSTRAP = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=os.environ.get("SEED_OLF_DATA_ROOT")
    )
    parser.add_argument(
        "--phase2-dir", type=Path, default=Path("artifacts") / "phase2_channel"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts") / "phase3_fewshot"
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--skip-negative-controls", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--reuse-negative-controls", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs") / "phase3.toml"
    )
    return parser.parse_args()


def validate_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    expected = {
        "random_state": RANDOM_STATE,
        "calibration_sizes": list(CALIBRATION_SIZES),
        "balanced_samples": N_SAMPLES,
        "bootstrap_replicates": N_BOOTSTRAP,
        "negative_control_permutations": N_PERMUTATIONS,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise ValueError(f"Phase 3 config/code mismatch: {mismatch}")
    return config


def _seed(subject: int, size: int, permutation: int = 0) -> int:
    return RANDOM_STATE + subject * 10_000 + size * 100 + permutation


def _calibration_frame(
    session_one: pd.DataFrame,
    subset: tuple[int, ...],
    permuted_labels: np.ndarray | None,
) -> pd.DataFrame:
    calibration = session_one.iloc[list(subset)].copy()
    if permuted_labels is not None:
        calibration["y_emotion"] = permuted_labels[list(subset)]
    return calibration


def _permute_session_one_labels(session_one: pd.DataFrame, seed: int) -> np.ndarray:
    """Break odor-label correspondence while retaining each subject's label total."""

    rng = np.random.default_rng(seed)
    labels = session_one.y_emotion.to_numpy(dtype=int).copy()
    return rng.permutation(labels)


def _m5_predictions(phase2_dir: Path, subject: int, target_test: pd.DataFrame) -> np.ndarray:
    predictions = pd.read_csv(phase2_dir / "trial_predictions.csv", usecols=[
        "protocol", "model", "subject", "session", "trial", "probability"
    ])
    m5 = predictions[
        (predictions.protocol == "loso")
        & (predictions.model == "M5")
        & (predictions.subject == subject)
    ].set_index(["subject", "session", "trial"])
    keys = pd.MultiIndex.from_frame(target_test[["subject", "session", "trial"]])
    return m5.loc[keys, "probability"].to_numpy(dtype=float)


def run_subject(
    manifest: pd.DataFrame,
    phase2_dir: Path,
    subject: int,
    permutation: int = 0,
    models: tuple[str, ...] = MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase2_dir = Path(phase2_dir)
    target = manifest[manifest.subject == subject].copy()
    session_one = target[target.session == 1].reset_index(drop=True)
    target_test = target[target.session.isin([2, 3])].sort_values(["session", "trial"]).reset_index(drop=True)
    pooled = manifest[manifest.subject != subject].copy()
    pooled_model = fit_pooled_logistic(pooled) if "P3" in models else None
    m5 = _m5_predictions(phase2_dir, subject, target_test) if "P4" in models else None
    permuted_labels = (
        _permute_session_one_labels(session_one, _seed(subject, 0, permutation))
        if permutation
        else None
    )
    blocks: list[pd.DataFrame] = []
    assignment_blocks: list[pd.DataFrame] = []
    for size in CALIBRATION_SIZES:
        subsets = balanced_calibration_subsets(
            session_one,
            size,
            _seed(subject, size),
            n_samples=N_SAMPLES,
        )
        for replicate, subset in enumerate(subsets):
            calibration = _calibration_frame(session_one, subset, permuted_labels)
            probabilities = {"P0": pooled_odor_prior(pooled, target_test)}
            if "P1" in models:
                probabilities["P1"] = direct_individual_probabilities(pooled, calibration, target_test)
            if "P2" in models:
                probabilities["P2"] = empirical_bayes_probabilities(pooled, calibration, target_test)
            p3_c = None
            if "P3" in models:
                probabilities["P3"], p3_c = pooled_plus_target_logistic_probabilities(
                    pooled, calibration, target_test, pooled_model=pooled_model
                )
            if "P4" in models:
                probabilities["P4"] = m5
            for model, probability in probabilities.items():
                block = target_test[["subject", "session", "trial", "y_emotion", "y_odor"]].copy()
                block["calibration_size"] = size
                block["replicate"] = replicate
                block["model"] = model
                block["probability"] = probability
                block["p3_c"] = p3_c if model == "P3" else np.nan
                blocks.append(block)
            if subset:
                assignment = session_one.iloc[list(subset)][["subject", "session", "trial", "y_odor"]].copy()
                assignment["calibration_size"] = size
                assignment["replicate"] = replicate
                assignment_blocks.append(assignment)
    return (
        pd.concat(blocks, ignore_index=True),
        pd.concat(assignment_blocks, ignore_index=True)
        if assignment_blocks
        else pd.DataFrame(columns=["subject", "session", "trial", "y_odor", "calibration_size", "replicate"]),
    )


def run_predictions(
    manifest: pd.DataFrame,
    phase2_dir: Path,
    permutation: int = 0,
    models: tuple[str, ...] = MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def run_one(subject: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        return run_subject(manifest, phase2_dir, subject, permutation, models)

    with ThreadPoolExecutor(max_workers=4) as executor:
        completed = list(executor.map(run_one, sorted(manifest.subject.unique())))
    return (
        pd.concat([rows for rows, _ in completed], ignore_index=True),
        pd.concat([assignments for _, assignments in completed], ignore_index=True),
    )


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size, model), part in predictions.groupby(["calibration_size", "model"], sort=True):
        rows.append(
            {
                "calibration_size": size,
                "model": model,
                **phase2_prediction_metrics(part.y_emotion.to_numpy(), part.probability.to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def comparison_table(predictions: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    key = ["subject", "session", "trial", "calibration_size", "replicate"]
    for size in CALIBRATION_SIZES:
        size_data = predictions[predictions.calibration_size == size]
        wide = size_data.pivot(index=key, columns="model", values="probability")
        labels = (
            size_data.drop_duplicates(key).set_index(key).loc[wide.index, "y_emotion"].to_numpy(dtype=int)
        )
        for candidate in sorted(model for model in wide.columns if model != "P0"):
            p0 = np.clip(wide.P0.to_numpy(), 1e-6, 1 - 1e-6)
            p = np.clip(wide[candidate].to_numpy(), 1e-6, 1 - 1e-6)
            log_diff = -(
                labels * np.log(p0) + (1 - labels) * np.log1p(-p0)
            ) + (labels * np.log(p) + (1 - labels) * np.log1p(-p))
            brier_diff = (labels - p0) ** 2 - (labels - p) ** 2
            differences = pd.DataFrame(
                {
                    "subject": wide.index.get_level_values("subject"),
                    "replicate": wide.index.get_level_values("replicate"),
                    "session": wide.index.get_level_values("session"),
                    "log_difference": log_diff,
                    "brier_difference": brier_diff,
                }
            )
            log_estimate, log_lower, log_upper = phase3_hierarchical_bootstrap(
                differences.rename(columns={"log_difference": "difference"})[["subject", "replicate", "difference"]],
                n_bootstrap=n_bootstrap,
                seed=RANDOM_STATE + size,
            )
            brier_estimate, brier_lower, brier_upper = phase3_hierarchical_bootstrap(
                differences.rename(columns={"brier_difference": "difference"})[["subject", "replicate", "difference"]],
                n_bootstrap=n_bootstrap,
                seed=RANDOM_STATE + size + 1,
            )
            participant_means = differences.groupby(["subject", "replicate"]).log_difference.mean().groupby("subject").mean()
            session_means = differences.groupby("session").log_difference.mean()
            rows.append(
                {
                    "calibration_size": size,
                    "candidate": candidate,
                    "baseline": "P0",
                    "n_subjects": int(differences.subject.nunique()),
                    "n_replicates_per_subject": int(differences.groupby("subject").replicate.nunique().min()),
                    "mean_log_loss_improvement": log_estimate,
                    "log_loss_ci95_lower": log_lower,
                    "log_loss_ci95_upper": log_upper,
                    "mean_brier_improvement": brier_estimate,
                    "brier_ci95_lower": brier_lower,
                    "brier_ci95_upper": brier_upper,
                    "participants_improved_rate": float(np.mean(participant_means > 0.0)),
                    "session_2_mean_log_loss_improvement": float(session_means.loc[2]),
                    "session_3_mean_log_loss_improvement": float(session_means.loc[3]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_gate(
    comparisons: pd.DataFrame,
    audits_passed: bool,
    negative_controls: pd.DataFrame | None = None,
) -> tuple[str, dict[str, object]]:
    p2 = comparisons[comparisons.candidate == "P2"].set_index("calibration_size")
    checks = {}
    passing_sizes = []
    for size in CALIBRATION_SIZES[1:]:
        row = p2.loc[size]
        size_checks = {
            "log_loss_ci_lower_above_zero": bool(row.log_loss_ci95_lower > 0.0),
            "brier_point_not_worse": bool(row.mean_brier_improvement >= 0.0),
            "brier_ci_within_margin": bool(row.brier_ci95_lower >= -0.005),
            "majority_of_participants_improve": bool(row.participants_improved_rate > 0.5),
            "session_2_nonnegative": bool(row.session_2_mean_log_loss_improvement >= 0.0),
            "session_3_nonnegative": bool(row.session_3_mean_log_loss_improvement >= 0.0),
            "audits_passed": audits_passed,
        }
        if negative_controls is not None:
            null_maximum = negative_controls.loc[
                negative_controls.calibration_size == size,
                "mean_log_loss_improvement",
            ].max()
            size_checks["exceeds_permuted_null_maximum"] = bool(
                row.mean_log_loss_improvement > null_maximum
            )
        checks[str(size)] = size_checks
        if all(size_checks.values()):
            passing_sizes.append(size)
    classification = "passed" if passing_sizes else "negative"
    return classification, {"by_size": checks, "minimum_feasible_size": min(passing_sizes) if passing_sizes else None}


def run_negative_controls(manifest: pd.DataFrame, phase2_dir: Path, n_bootstrap: int, permutations: int) -> pd.DataFrame:
    summaries = []
    for permutation in range(1, permutations + 1):
        comparisons = negative_control_comparisons(manifest, permutation, n_bootstrap)
        classification, gate = evaluate_gate(comparisons, audits_passed=True)
        p2 = comparisons[comparisons.candidate == "P2"]
        for row in p2.itertuples(index=False):
            summaries.append(
                {
                    "permutation": permutation,
                    "calibration_size": row.calibration_size,
                    "classification": classification,
                    "all_statistical_gates_pass": bool(gate["by_size"].get(str(row.calibration_size), {}) and all(gate["by_size"][str(row.calibration_size)].values())),
                    "mean_log_loss_improvement": row.mean_log_loss_improvement,
                }
            )
        print(f"Completed Phase 3 permutation {permutation}/{permutations}", flush=True)
    return pd.DataFrame(summaries)


def negative_control_comparisons(
    manifest: pd.DataFrame, permutation: int, n_bootstrap: int
) -> pd.DataFrame:
    """Evaluate permuted P2 directly, avoiding materializing redundant P0/P2 tables."""

    rows = []
    for size in CALIBRATION_SIZES:
        subject_differences = []
        for subject in sorted(manifest.subject.unique()):
            target = manifest[manifest.subject == subject]
            session_one = target[target.session == 1].reset_index(drop=True)
            target_test = target[target.session.isin([2, 3])].sort_values(
                ["session", "trial"]
            ).reset_index(drop=True)
            pooled = manifest[manifest.subject != subject]
            p0 = pooled_odor_prior(pooled, target_test)
            alpha = np.array(
                [
                    1.0 + pooled.loc[pooled.y_odor == odor, "y_emotion"].sum()
                    for odor in range(1, 5)
                ]
            )
            beta = np.array(
                [
                    1.0
                    + (pooled.y_odor == odor).sum()
                    - pooled.loc[pooled.y_odor == odor, "y_emotion"].sum()
                    for odor in range(1, 5)
                ]
            )
            labels = _permute_session_one_labels(
                session_one, _seed(subject, 0, permutation)
            )
            subsets = balanced_calibration_subsets(
                session_one, size, _seed(subject, size), n_samples=N_SAMPLES
            )
            subset_array = np.asarray(subsets, dtype=int)
            n_replicates = len(subsets)
            positive_counts = np.zeros((n_replicates, 4), dtype=float)
            selected_counts = np.zeros((n_replicates, 4), dtype=float)
            for odor_index, odor in enumerate(range(1, 5)):
                selected = session_one.y_odor.to_numpy()[subset_array] == odor
                selected_counts[:, odor_index] = selected.sum(axis=1)
                positive_counts[:, odor_index] = (
                    labels[subset_array] * selected
                ).sum(axis=1)
            odor_probabilities = (alpha + positive_counts) / (
                alpha + beta + selected_counts
            )
            test_odor_indices = target_test.y_odor.to_numpy(dtype=int) - 1
            p2 = odor_probabilities[:, test_odor_indices]
            y = target_test.y_emotion.to_numpy(dtype=int)
            p0_safe = np.clip(p0, 1e-6, 1 - 1e-6)
            p2_safe = np.clip(p2, 1e-6, 1 - 1e-6)
            log_difference = -(
                y * np.log(p0_safe) + (1 - y) * np.log1p(-p0_safe)
            ) + (y * np.log(p2_safe) + (1 - y) * np.log1p(-p2_safe))
            brier_difference = (y - p0) ** 2 - (y - p2) ** 2
            subject_differences.append(
                pd.DataFrame(
                    {
                        "subject": subject,
                        "replicate": np.repeat(np.arange(n_replicates), len(target_test)),
                        "session": np.tile(target_test.session.to_numpy(), n_replicates),
                        "log_difference": log_difference.ravel(),
                        "brier_difference": brier_difference.ravel(),
                    }
                )
            )
        differences = pd.concat(subject_differences, ignore_index=True)
        log_estimate, log_lower, log_upper = phase3_hierarchical_bootstrap(
            differences.rename(columns={"log_difference": "difference"})[
                ["subject", "replicate", "difference"]
            ],
            n_bootstrap=n_bootstrap,
            seed=RANDOM_STATE + size,
        )
        brier_estimate, brier_lower, brier_upper = phase3_hierarchical_bootstrap(
            differences.rename(columns={"brier_difference": "difference"})[
                ["subject", "replicate", "difference"]
            ],
            n_bootstrap=n_bootstrap,
            seed=RANDOM_STATE + size + 1,
        )
        participant_means = (
            differences.groupby(["subject", "replicate"]).log_difference.mean()
            .groupby("subject")
            .mean()
        )
        session_means = differences.groupby("session").log_difference.mean()
        rows.append(
            {
                "calibration_size": size,
                "candidate": "P2",
                "baseline": "P0",
                "n_subjects": int(differences.subject.nunique()),
                "n_replicates_per_subject": int(
                    differences.groupby("subject").replicate.nunique().min()
                ),
                "mean_log_loss_improvement": log_estimate,
                "log_loss_ci95_lower": log_lower,
                "log_loss_ci95_upper": log_upper,
                "mean_brier_improvement": brier_estimate,
                "brier_ci95_lower": brier_lower,
                "brier_ci95_upper": brier_upper,
                "participants_improved_rate": float(np.mean(participant_means > 0.0)),
                "session_2_mean_log_loss_improvement": float(session_means.loc[2]),
                "session_3_mean_log_loss_improvement": float(session_means.loc[3]),
            }
        )
    return pd.DataFrame(rows)


def write_metadata(output_dir: Path, classification: str, gate: dict, phase2_dir: Path) -> None:
    metadata = {
        "experiment": "phase3_fewshot",
        "classification": classification,
        "success_gate": gate,
        "random_state": RANDOM_STATE,
        "calibration_sizes": list(CALIBRATION_SIZES),
        "balanced_samples": N_SAMPLES,
        "bootstrap_replicates": N_BOOTSTRAP,
        "negative_control_permutations": N_PERMUTATIONS,
        "phase2_m5_source": str(phase2_dir / "trial_predictions.csv"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = validate_config(args.config)
    if args.bootstrap != config["bootstrap_replicates"]:
        raise ValueError("Declared run requires 10,000 bootstrap replicates")
    if not args.skip_negative_controls and args.permutations != config["negative_control_permutations"]:
        raise ValueError("Declared run requires 20 negative-control permutations")
    if args.reuse_negative_controls and args.skip_negative_controls:
        raise ValueError("--reuse-negative-controls requires negative controls")
    if args.data_root is None:
        raise SystemExit("Pass --data-root or set SEED_OLF_DATA_ROOT")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.phase2_dir / "manifest.csv")
    source_paths = [Path(path) for path in manifest[["stimulation_path", "recovery_path"]].to_numpy().reshape(-1)]
    phase2_cache = json.loads((args.phase2_dir / "channel_feature_cache.json").read_text())
    if compute_source_fingerprint(source_paths, args.data_root).combined_sha256 != phase2_cache["source_combined_sha256"]:
        raise RuntimeError("Phase 2 source fingerprint mismatch")

    models = tuple(args.models)
    if "P0" not in models:
        raise ValueError("P0 is required for every Phase 3 comparison")
    prediction_path = args.output_dir / "trial_predictions.csv"
    assignment_path = args.output_dir / "calibration_assignments.csv"
    if args.reuse_predictions:
        if not prediction_path.exists() or not assignment_path.exists():
            raise ValueError("--reuse-predictions requires existing prediction and assignment files")
        predictions = pd.read_csv(prediction_path)
        assignments = pd.read_csv(assignment_path)
    else:
        predictions, assignments = run_predictions(manifest, args.phase2_dir, models=models)
    target_test = manifest[manifest.session.isin([2, 3])]
    for subject in sorted(manifest.subject.unique()):
        verify_phase3_prediction_coverage(predictions[predictions.subject == subject], target_test[target_test.subject == subject], list(CALIBRATION_SIZES), list(models))
    if not args.reuse_predictions:
        predictions.to_csv(prediction_path, index=False)
        assignments.to_csv(assignment_path, index=False)
    summarize_predictions(predictions).to_csv(args.output_dir / "prediction_summary.csv", index=False)
    comparisons = comparison_table(predictions, args.bootstrap)
    comparisons.to_csv(args.output_dir / "incremental_value.csv", index=False)
    audits_passed = False
    negative = None
    if not args.skip_negative_controls:
        if "P2" not in models:
            raise ValueError("P2 is required for Phase 3 negative controls")
        negative_path = args.output_dir / "negative_control_summary.csv"
        if args.reuse_negative_controls:
            if not negative_path.exists():
                raise ValueError("--reuse-negative-controls requires existing negative-control results")
            negative = pd.read_csv(negative_path)
        else:
            negative = run_negative_controls(manifest, args.phase2_dir, args.bootstrap, args.permutations)
            negative.to_csv(negative_path, index=False)
        audits_passed = bool(
            not negative.all_statistical_gates_pass.any()
        )
    classification, gate = evaluate_gate(comparisons, audits_passed, negative)
    write_metadata(args.output_dir, classification, gate, args.phase2_dir)
    print(f"Phase 3 classification: {classification}")


if __name__ == "__main__":
    main()
