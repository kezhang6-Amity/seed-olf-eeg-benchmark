"""Add a spatial-covariance baseline using professional Riemannian tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.geometry.base import logm
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.compose import ColumnTransformer
from sklearn.covariance import oas
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiments import (
    RANDOM_STATE,
    incremental_value_table,
    outer_folds,
    summarize_predictions,
)
from .data import load_payload
from .features import ASSUMED_SAMPLING_RATE_HZ, average_reference


MODEL_NAMES = {
    "riemann_stim_logit",
    "odor_riemann_stim_logit",
    "odor_riemann_relative_logit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "artifacts" / "baseline_v1",
    )
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def relative_covariance(stimulation: np.ndarray, recovery: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(recovery)
    floor = max(float(eigenvalues.max()) * 1e-9, np.finfo(float).tiny)
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
    relative = inverse_sqrt @ stimulation @ inverse_sqrt
    return (relative + relative.T) / 2.0


def logeuclidean_vector(covariance: np.ndarray) -> np.ndarray:
    matrix_log = logm(covariance)
    row, column = np.triu_indices(covariance.shape[0])
    vector = matrix_log[row, column]
    vector[row != column] *= np.sqrt(2.0)
    return vector


def covariance_features(manifest: pd.DataFrame, cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}

    bandpass = butter(
        4,
        [1.0, 45.0],
        btype="bandpass",
        fs=ASSUMED_SAMPLING_RATE_HZ,
        output="sos",
    )
    stimulation_vectors = []
    relative_vectors = []
    for index, trial in enumerate(manifest.itertuples(index=False), start=1):
        stimulation = average_reference(
            load_payload(Path(trial.stimulation_path))["X_raw"]
        )
        recovery = average_reference(load_payload(Path(trial.recovery_path))["X_raw"])
        stimulation = sosfiltfilt(bandpass, stimulation, axis=-1)
        recovery = sosfiltfilt(bandpass, recovery, axis=-1)
        stimulation_covariance = oas(stimulation.T, assume_centered=True)[0]
        recovery_covariance = oas(recovery.T, assume_centered=True)[0]
        relative = relative_covariance(stimulation_covariance, recovery_covariance)
        stimulation_vectors.append(logeuclidean_vector(stimulation_covariance))
        relative_vectors.append(logeuclidean_vector(relative))
        if index % 100 == 0:
            print(f"Covariance features: {index}/{len(manifest)}", flush=True)

    result = {
        "stimulation": np.asarray(stimulation_vectors, dtype=np.float32),
        "relative": np.asarray(relative_vectors, dtype=np.float32),
    }
    np.savez_compressed(cache_path, **result)
    return result


def append_odor(features: np.ndarray, odor: np.ndarray) -> np.ndarray:
    odor_one_hot = np.eye(4, dtype=np.float32)[odor.astype(int) - 1]
    return np.column_stack([features, odor_one_hot])


def tuned_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_test: np.ndarray,
    n_eeg_features: int,
) -> tuple[np.ndarray, float]:
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    eeg_pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("select", SelectPercentile(f_classif, percentile=10)),
            ("scale", StandardScaler()),
        ]
    )
    transformers = [("eeg", eeg_pipeline, list(range(n_eeg_features)))]
    if x_train.shape[1] > n_eeg_features:
        transformers.append(
            ("odor", "passthrough", list(range(n_eeg_features, x_train.shape[1])))
        )
    pipeline = Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", LogisticRegression(max_iter=3_000, solver="lbfgs")),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="neg_log_loss",
        cv=splitter,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train, groups=groups)
    return search.predict_proba(x_test)[:, 1], float(search.best_params_["model__C"])


def run_models(
    manifest: pd.DataFrame,
    covariance: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = manifest.copy().reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))
    rows = []
    for protocol in ("loso", "losession"):
        folds = list(outer_folds(frame, protocol))
        for fold_index, (fold_id, train, test, groups) in enumerate(folds, start=1):
            train_indices = train.row_id.to_numpy()
            test_indices = test.row_id.to_numpy()
            y_train = train.y_emotion.to_numpy()
            specifications = {
                "riemann_stim_logit": (
                    covariance["stimulation"][train_indices],
                    covariance["stimulation"][test_indices],
                ),
                "odor_riemann_stim_logit": (
                    append_odor(covariance["stimulation"][train_indices], train.y_odor.to_numpy()),
                    append_odor(covariance["stimulation"][test_indices], test.y_odor.to_numpy()),
                ),
                "odor_riemann_relative_logit": (
                    append_odor(covariance["relative"][train_indices], train.y_odor.to_numpy()),
                    append_odor(covariance["relative"][test_indices], test.y_odor.to_numpy()),
                ),
            }
            for model_name, (x_train, x_test) in specifications.items():
                probability, best_c = tuned_predictions(
                    x_train,
                    y_train,
                    np.asarray(groups),
                    x_test,
                    covariance["stimulation"].shape[1],
                )
                for trial, prediction in zip(test.itertuples(index=False), probability, strict=True):
                    rows.append(
                        {
                            "protocol": protocol,
                            "fold_id": fold_id,
                            "model": model_name,
                            "subject": trial.subject,
                            "session": trial.session,
                            "trial": trial.trial,
                            "y_emotion": trial.y_emotion,
                            "y_odor": trial.y_odor,
                            "probability": float(prediction),
                            "best_c": best_c,
                        }
                    )
            print(
                f"Riemannian {protocol}: {fold_index}/{len(folds)} outer folds",
                flush=True,
            )
    return pd.DataFrame(rows)


def add_disagreement_flag(
    new_predictions: pd.DataFrame,
    existing_predictions: pd.DataFrame,
) -> pd.DataFrame:
    key = ["protocol", "subject", "session", "trial"]
    flag = (
        existing_predictions[existing_predictions.model == "odor_prior"]
        [key + ["odor_prior_disagreement"]]
        .drop_duplicates(key)
    )
    return new_predictions.merge(flag, on=key, how="left", validate="many_to_one")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.output_dir / "manifest.csv")
    cache_path = args.output_dir / "logeuclidean_covariance_features.npz"
    if args.force_features and cache_path.exists():
        cache_path.unlink()
    covariance = covariance_features(manifest, cache_path)
    new_predictions = run_models(manifest, covariance)

    prediction_path = args.output_dir / "trial_predictions.csv"
    existing = pd.read_csv(prediction_path)
    existing = existing[~existing.model.isin(MODEL_NAMES)]
    new_predictions = add_disagreement_flag(new_predictions, existing)
    combined = pd.concat([existing, new_predictions], ignore_index=True)
    combined.to_csv(prediction_path, index=False)
    summarize_predictions(combined).to_csv(
        args.output_dir / "prediction_summary.csv",
        index=False,
    )
    incremental_value_table(combined).to_csv(
        args.output_dir / "incremental_value.csv",
        index=False,
    )
    print(f"Riemannian results added to {args.output_dir}")


if __name__ == "__main__":
    main()
