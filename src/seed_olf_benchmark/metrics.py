"""Prediction, reliability, and multiple-testing metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def prediction_metrics(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(probability), dtype=float), 1e-6, 1 - 1e-6)
    result = {
        "n_trials": int(len(y)),
        "prevalence": float(np.mean(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
    }
    result["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan
    result["pr_auc"] = (
        float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan
    )
    return result


def icc_absolute_agreement(matrix: np.ndarray) -> float:
    """Two-way random-effects absolute-agreement ICC(2,1)."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return math.nan
    n_targets, n_raters = values.shape
    grand_mean = values.mean()
    target_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)
    ms_targets = n_raters * np.sum((target_means - grand_mean) ** 2) / (n_targets - 1)
    ms_raters = n_targets * np.sum((rater_means - grand_mean) ** 2) / (n_raters - 1)
    residual = values - target_means[:, None] - rater_means[None, :] + grand_mean
    ms_error = np.sum(residual**2) / ((n_targets - 1) * (n_raters - 1))
    denominator = (
        ms_targets
        + (n_raters - 1) * ms_error
        + n_raters * (ms_raters - ms_error) / n_targets
    )
    return float((ms_targets - ms_error) / denominator) if denominator else math.nan


def bootstrap_icc(
    matrix: np.ndarray,
    n_bootstrap: int = 1_000,
    random_state: int = 20260810,
) -> tuple[float, float, float]:
    values = np.asarray(matrix, dtype=float)
    estimate = icc_absolute_agreement(values)
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, values.shape[0], values.shape[0])
        value = icc_absolute_agreement(values[indices])
        if np.isfinite(value):
            samples.append(value)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted
