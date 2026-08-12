"""Leakage-safe utilities for Phase 3 few-shot personalization."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


CALIBRATION_SIZES = (0, 4, 8, 16, 24)
ODORS = (1, 2, 3, 4)
TRIAL_KEY = ["subject", "session", "trial"]
RANDOM_STATE = 20260812
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
MODELS = ("P0", "P1", "P2", "P3", "P4")


def balanced_calibration_subsets(
    session_one: pd.DataFrame,
    calibration_size: int,
    seed: int,
    n_samples: int = 200,
) -> list[tuple[int, ...]]:
    """Sample deterministic, label-blind, odor-balanced index subsets."""

    if calibration_size not in CALIBRATION_SIZES:
        raise ValueError(f"Unsupported calibration size: {calibration_size}")
    if set(session_one.session.unique()) != {1}:
        raise ValueError("Calibration candidates must be session 1 only")
    counts = session_one.y_odor.value_counts().to_dict()
    if counts != {odor: 6 for odor in ODORS}:
        raise ValueError(f"Expected six trials per odor, found {counts}")
    if calibration_size == 0:
        return [()]
    per_odor = calibration_size // len(ODORS)
    odor_positions = {
        odor: tuple(session_one.index[session_one.y_odor == odor]) for odor in ODORS
    }
    combinations = [
        list(itertools.combinations(odor_positions[odor], per_odor)) for odor in ODORS
    ]
    total_combinations = int(np.prod([len(values) for values in combinations]))
    if calibration_size == 24:
        return [tuple(range(len(session_one)))]
    if n_samples > total_combinations:
        n_samples = total_combinations
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total_combinations, size=n_samples, replace=False)
    subsets = []
    dimensions = tuple(len(values) for values in combinations)
    for flat_index in chosen:
        indices = np.unravel_index(flat_index, dimensions)
        subset = tuple(
            sorted(
                trial_index
                for odor_index, combination_index in enumerate(indices)
                for trial_index in combinations[odor_index][combination_index]
            )
        )
        subsets.append(subset)
    return subsets


def _posterior_counts(pooled: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    alpha = {}
    beta = {}
    for odor in ODORS:
        labels = pooled.loc[pooled.y_odor == odor, "y_emotion"]
        alpha[odor] = 1.0 + float(labels.sum())
        beta[odor] = 1.0 + float(len(labels) - labels.sum())
    return alpha, beta


def pooled_odor_prior(pooled: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Return Beta(1,1)-smoothed leave-target-participant-out odor probabilities."""

    alpha, beta = _posterior_counts(pooled)
    return test.y_odor.map(
        {odor: alpha[odor] / (alpha[odor] + beta[odor]) for odor in ODORS}
    ).to_numpy(dtype=np.float64)


def empirical_bayes_probabilities(
    pooled: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """Update the pooled Beta posterior with target calibration labels."""

    alpha, beta = _posterior_counts(pooled)
    probabilities = {}
    for odor in ODORS:
        labels = calibration.loc[calibration.y_odor == odor, "y_emotion"]
        positive = float(labels.sum())
        negative = float(len(labels) - labels.sum())
        probabilities[odor] = (alpha[odor] + positive) / (
            alpha[odor] + beta[odor] + positive + negative
        )
    return test.y_odor.map(probabilities).to_numpy(dtype=np.float64)


def direct_individual_probabilities(
    pooled: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """Use direct target frequencies with Beta(1,1) smoothing; P0 at size zero."""

    if calibration.empty:
        return pooled_odor_prior(pooled, test)
    probabilities = {}
    for odor in ODORS:
        labels = calibration.loc[calibration.y_odor == odor, "y_emotion"]
        probabilities[odor] = (1.0 + float(labels.sum())) / (2.0 + len(labels))
    return test.y_odor.map(probabilities).to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class PooledLogisticModel:
    encoder_categories: tuple[int, ...]
    coefficient: np.ndarray
    intercept: float
    c_value: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        design = np.column_stack(
            [(frame.y_odor.to_numpy() == odor).astype(float) for odor in self.encoder_categories]
        )
        linear = self.intercept + design @ self.coefficient
        return 1.0 / (1.0 + np.exp(-linear))


def _odor_design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [(frame.y_odor.to_numpy() == odor).astype(float) for odor in ODORS]
    )


def fit_pooled_logistic(pooled: pd.DataFrame) -> PooledLogisticModel:
    """Tune P3 on other participants only, then fit it once for one target."""

    groups = pooled.subject.to_numpy()
    splitter = GroupKFold(n_splits=5)
    x = _odor_design(pooled)
    y = pooled.y_emotion.to_numpy(dtype=int)
    losses = []
    for c_value in C_GRID:
        fold_losses = []
        for fit, validation in splitter.split(x, y, groups=groups):
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=5000,
                class_weight=None,
                random_state=RANDOM_STATE,
            ).fit(x[fit], y[fit])
            probability = np.clip(model.predict_proba(x[validation])[:, 1], 1e-6, 1 - 1e-6)
            fold_losses.append(
                -np.mean(
                    y[validation] * np.log(probability)
                    + (1 - y[validation]) * np.log1p(-probability)
                )
            )
        losses.append(float(np.mean(fold_losses)))
    best_c = C_GRID[int(np.argmin(losses))]
    fitted = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=RANDOM_STATE,
    ).fit(x, y)
    return PooledLogisticModel(
        encoder_categories=ODORS,
        coefficient=fitted.coef_[0].copy(),
        intercept=float(fitted.intercept_[0]),
        c_value=best_c,
    )


def pooled_plus_target_logistic_probabilities(
    pooled: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    pooled_model: PooledLogisticModel | None = None,
) -> tuple[np.ndarray, float]:
    """Fit P3 on pooled plus selected target labels after pooled-only C selection."""

    pooled_model = pooled_model or fit_pooled_logistic(pooled)
    if calibration.empty:
        return pooled_model.predict_proba(test), pooled_model.c_value
    training = pd.concat([pooled, calibration], ignore_index=True)
    model = LogisticRegression(
        C=pooled_model.c_value,
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=RANDOM_STATE,
    ).fit(_odor_design(training), training.y_emotion.to_numpy(dtype=int))
    probability = model.predict_proba(_odor_design(test))[:, 1]
    return probability, pooled_model.c_value


def phase3_hierarchical_bootstrap(
    differences: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Resample participants, calibration replicates, then held-out trials."""

    required = {"subject", "replicate", "difference"}
    if not required.issubset(differences):
        raise ValueError(f"Missing bootstrap columns: {sorted(required - set(differences))}")
    grouped = [
        [
            replicate.difference.to_numpy(dtype=np.float64)
            for _, replicate in part.groupby("replicate", sort=True)
        ]
        for _, part in differences.groupby("subject", sort=True)
    ]
    replicate_counts = {len(replicates) for replicates in grouped}
    trial_counts = {
        len(values) for replicates in grouped for values in replicates
    }
    if len(replicate_counts) != 1 or len(trial_counts) != 1:
        raise ValueError("Phase 3 bootstrap requires balanced replicate and trial counts")
    values = np.asarray(grouped, dtype=np.float64)
    n_subjects, n_replicates, n_trials = values.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    batch_size = 256
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        batch = stop - start
        selected_subjects = rng.integers(0, n_subjects, size=(batch, n_subjects))
        selected_replicates = rng.integers(0, n_replicates, size=(batch, n_subjects))
        selected_trials = rng.integers(
            0, n_trials, size=(batch, n_subjects, n_trials)
        )
        samples[start:stop] = values[
            selected_subjects[:, :, None],
            selected_replicates[:, :, None],
            selected_trials,
        ].mean(axis=(1, 2))
    estimate = float(
        differences.groupby(["subject", "replicate"]).difference.mean().groupby("subject").mean().mean()
    )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def verify_phase3_prediction_coverage(
    predictions: pd.DataFrame,
    target_test: pd.DataFrame,
    calibration_sizes: list[int],
    models: list[str],
    sampled_replicates: int = 200,
) -> None:
    """Require every target test trial once for each size/model/replicate."""

    expected_trials = len(target_test)
    problems = []
    counts = predictions.groupby(
        ["calibration_size", "replicate", "model"], sort=True
    ).size()
    observed_sizes = set(counts.index.get_level_values("calibration_size"))
    for size in calibration_sizes:
        subset = counts.loc[size] if size in observed_sizes else pd.Series(dtype=int)
        replicates = set(subset.index.get_level_values("replicate")) if not subset.empty else set()
        expected_replicate_count = 1 if size in (0, 24) else sampled_replicates
        if len(replicates) != expected_replicate_count:
            problems.append(f"size={size}/replicate_count={len(replicates)}")
        for replicate in replicates:
            present_models = set(subset.loc[replicate].index)
            if present_models != set(models):
                problems.append(f"size={size}/replicate={replicate}/models")
                continue
            if any(subset.loc[replicate, model] != expected_trials for model in models):
                problems.append(f"size={size}/replicate={replicate}/trial_count")
    duplicate_trials = predictions.duplicated(
        ["calibration_size", "replicate", "model", *TRIAL_KEY]
    ).any()
    if duplicate_trials:
        problems.append("duplicate_trial_keys")
    valid_trials = target_test[TRIAL_KEY].drop_duplicates()
    invalid_trials = predictions.merge(
        valid_trials, on=TRIAL_KEY, how="left", indicator=True
    )["_merge"].ne("both").any()
    if invalid_trials:
        problems.append("unexpected_trial_keys")
    if problems or not np.isfinite(predictions.probability.to_numpy(dtype=float)).all():
        raise ValueError("Phase 3 prediction coverage failure: " + ", ".join(problems))
