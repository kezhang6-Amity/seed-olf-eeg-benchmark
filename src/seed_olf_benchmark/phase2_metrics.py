"""Phase 2 predictive metrics, paired uncertainty, and decision gates."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

from .metrics import benjamini_hochberg, prediction_metrics
from .phase2_validation import hierarchical_paired_bootstrap


TRIAL_KEY = ["subject", "session", "trial"]


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    order = np.argsort(probability, kind="stable")
    bins = np.array_split(order, n_bins)
    return float(
        sum(
            len(indices)
            / len(order)
            * abs(float(np.mean(y_true[indices])) - float(np.mean(probability[indices])))
            for indices in bins
            if len(indices)
        )
    )


def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    logits = np.log(p / (1.0 - p))

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logits
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    fit = minimize(negative_log_likelihood, np.array([0.0, 1.0]), method="BFGS")
    if not np.isfinite(fit.x).all():
        return math.nan, math.nan
    return float(fit.x[0]), float(fit.x[1])


def phase2_prediction_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    result = prediction_metrics(y_true, probability)
    intercept, slope = calibration_intercept_slope(y_true, probability)
    result.update(
        {
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "ece_10_equal_frequency": expected_calibration_error(
                np.asarray(y_true, dtype=int), np.asarray(probability, dtype=np.float64)
            ),
        }
    )
    return result


def summarize_phase2_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, model), part in predictions.groupby(["protocol", "model"], sort=True):
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                **phase2_prediction_metrics(
                    part.y_emotion.to_numpy(), part.probability.to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(
    predictions: pd.DataFrame,
    protocol: str,
    candidate: str,
    baseline: str = "M0",
    n_bootstrap: int = 10_000,
    seed: int = 20260810,
) -> dict[str, float | int | str]:
    part = predictions[
        (predictions.protocol == protocol)
        & predictions.model.isin([baseline, candidate])
    ]
    wide = part.pivot(index=TRIAL_KEY, columns="model", values="probability")
    labels = (
        part.drop_duplicates(TRIAL_KEY)
        .set_index(TRIAL_KEY)
        .loc[wide.index, "y_emotion"]
        .to_numpy(dtype=int)
    )
    odors = (
        part.drop_duplicates(TRIAL_KEY)
        .set_index(TRIAL_KEY)
        .loc[wide.index, "y_odor"]
        .to_numpy(dtype=int)
    )
    baseline_probability = np.clip(wide[baseline].to_numpy(), 1e-6, 1 - 1e-6)
    candidate_probability = np.clip(wide[candidate].to_numpy(), 1e-6, 1 - 1e-6)
    baseline_loss = -(
        labels * np.log(baseline_probability)
        + (1 - labels) * np.log1p(-baseline_probability)
    )
    candidate_loss = -(
        labels * np.log(candidate_probability)
        + (1 - labels) * np.log1p(-candidate_probability)
    )
    log_differences = pd.DataFrame(
        {
            "subject": wide.index.get_level_values("subject"),
            "difference": baseline_loss - candidate_loss,
        }
    )
    brier_differences = pd.DataFrame(
        {
            "subject": wide.index.get_level_values("subject"),
            "difference": (labels - baseline_probability) ** 2
            - (labels - candidate_probability) ** 2,
        }
    )
    log_interval = hierarchical_paired_bootstrap(
        log_differences, n_bootstrap=n_bootstrap, seed=seed
    )
    brier_interval = hierarchical_paired_bootstrap(
        brier_differences, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    participant_means = log_differences.groupby("subject").difference.mean()
    result: dict[str, float | int | str] = {
        "protocol": protocol,
        "candidate": candidate,
        "baseline": baseline,
        "n_subjects": int(log_differences.subject.nunique()),
        "n_trials": len(log_differences),
        "mean_log_loss_improvement": log_interval.estimate,
        "log_loss_ci95_lower": log_interval.lower,
        "log_loss_ci95_upper": log_interval.upper,
        "mean_brier_improvement": brier_interval.estimate,
        "brier_ci95_lower": brier_interval.lower,
        "brier_ci95_upper": brier_interval.upper,
        "participants_improved_rate": float(np.mean(participant_means > 0.0)),
    }
    if protocol == "loso":
        for odor in sorted(np.unique(odors)):
            result[f"delete_odor_{odor}_log_loss_improvement"] = float(
                np.mean((baseline_loss - candidate_loss)[odors != odor])
            )
    try:
        result["participant_wilcoxon_p_value"] = float(
            wilcoxon(participant_means, zero_method="wilcox").pvalue
        )
    except ValueError:
        result["participant_wilcoxon_p_value"] = math.nan
    return result


def comparison_table(
    predictions: pd.DataFrame,
    candidates: list[str],
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    rows = [
        paired_comparison(
            predictions,
            protocol,
            candidate,
            n_bootstrap=n_bootstrap,
        )
        for protocol in ("loso", "losession")
        for candidate in candidates
    ]
    result = pd.DataFrame(rows)
    secondary_mask = result.candidate.isin(["M3", "M4", "M5"])
    result["participant_wilcoxon_q_value"] = math.nan
    if secondary_mask.any():
        result.loc[secondary_mask, "participant_wilcoxon_q_value"] = benjamini_hochberg(
            result.loc[secondary_mask, "participant_wilcoxon_p_value"].to_numpy()
        )
    return result


def evaluate_success_gate(
    comparisons: pd.DataFrame,
    audits_passed: bool,
) -> tuple[str, dict[str, bool]]:
    loso = comparisons[
        (comparisons.protocol == "loso") & (comparisons.candidate == "M2")
    ].iloc[0]
    losession = comparisons[
        (comparisons.protocol == "losession") & (comparisons.candidate == "M2")
    ].iloc[0]
    odor_columns = [
        f"delete_odor_{odor}_log_loss_improvement" for odor in (1, 2, 3, 4)
    ]
    checks = {
        "loso_log_loss_ci_lower_above_zero": bool(loso.log_loss_ci95_lower > 0.0),
        "loso_brier_point_not_worse": bool(loso.mean_brier_improvement >= 0.0),
        "loso_brier_ci_within_margin": bool(loso.brier_ci95_lower >= -0.005),
        "losession_mean_log_loss_improvement_positive": bool(
            losession.mean_log_loss_improvement > 0.0
        ),
        "majority_of_participants_improve": bool(
            loso.participants_improved_rate > 0.5
        ),
        "delete_one_odor_checks_nonnegative": bool(
            all(loso[column] >= 0.0 for column in odor_columns)
        ),
        "audits_passed": bool(audits_passed),
    }
    return ("passed" if all(checks.values()) else "signal_without_passage"), checks
