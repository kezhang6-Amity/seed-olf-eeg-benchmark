"""Fold-local transformations and models for Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .models import smoothed_prior
from .phase2_validation import inner_splits


RANDOM_STATE = 20260810
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


class OdorResidualizer:
    """Subtract odor means learned only from the supplied training frame."""

    def __init__(self, feature_columns: Sequence[str]):
        self.feature_columns = list(feature_columns)

    def fit(self, frame: pd.DataFrame) -> OdorResidualizer:
        if not self.feature_columns:
            raise ValueError("At least one feature column is required")
        values = frame[self.feature_columns].to_numpy(dtype=np.float64)
        odors = frame["y_odor"].to_numpy()
        self.odor_values_ = np.sort(np.unique(odors))
        self.odor_means_ = np.vstack(
            [values[odors == odor].mean(axis=0) for odor in self.odor_values_]
        )
        self.global_mean_ = values.mean(axis=0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_mean_"):
            raise RuntimeError("OdorResidualizer must be fitted before transform")
        values = frame[self.feature_columns].to_numpy(dtype=np.float64)
        odors = frame["y_odor"].to_numpy()
        result = np.empty_like(values)
        lookup = {
            odor: self.odor_means_[index]
            for index, odor in enumerate(self.odor_values_)
        }
        for odor in np.unique(odors):
            mask = odors == odor
            result[mask] = values[mask] - lookup.get(odor, self.global_mean_)
        return result

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


@dataclass
class FittedOdorEEGModel:
    feature_columns: list[str]
    robust: bool
    encoder: OneHotEncoder
    classifier: LogisticRegression
    residualizer: OdorResidualizer | None = None
    scaler: StandardScaler | RobustScaler | None = None
    clip_lower: np.ndarray | None = None
    clip_upper: np.ndarray | None = None

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        odor = self.encoder.transform(frame[["y_odor"]])
        if not self.feature_columns:
            return odor
        if self.residualizer is None or self.scaler is None:
            raise RuntimeError("EEG model is missing fitted transformations")
        eeg = self.residualizer.transform(frame)
        if self.robust:
            eeg = np.clip(eeg, self.clip_lower, self.clip_upper)
        return np.hstack([odor, self.scaler.transform(eeg)])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self._design(frame))[:, 1]


@dataclass(frozen=True)
class TunedPrediction:
    probability: np.ndarray
    best_c: float
    inner_log_loss: float


def prior_predictions_with_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: Sequence[int] | None = None,
) -> np.ndarray:
    """Add-one odor prior using the explicitly supplied training target."""

    y = train.y_emotion.to_numpy(dtype=int) if target is None else np.asarray(target, dtype=int)
    fallback = smoothed_prior(int(y.sum()), len(y))
    priors = {
        odor: smoothed_prior(int(y[train.y_odor.to_numpy() == odor].sum()), int(np.sum(train.y_odor.to_numpy() == odor)))
        for odor in np.unique(train.y_odor)
    }
    return test.y_odor.map(priors).fillna(fallback).to_numpy(dtype=np.float64)


def fit_odor_eeg_model(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    c_value: float,
    robust: bool = False,
    target: Sequence[int] | None = None,
) -> FittedOdorEEGModel:
    """Fit all transformations on one training partition and nothing else."""

    columns = list(feature_columns)
    y = train.y_emotion.to_numpy(dtype=int) if target is None else np.asarray(target, dtype=int)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    odor = encoder.fit_transform(train[["y_odor"]])
    residualizer = None
    scaler = None
    clip_lower = None
    clip_upper = None
    design = odor
    if columns:
        residualizer = OdorResidualizer(columns)
        eeg = residualizer.fit_transform(train)
        if robust:
            clip_lower = np.quantile(eeg, 0.01, axis=0)
            clip_upper = np.quantile(eeg, 0.99, axis=0)
            eeg = np.clip(eeg, clip_lower, clip_upper)
            scaler = RobustScaler().fit(eeg)
        else:
            scaler = StandardScaler().fit(eeg)
        design = np.hstack([odor, scaler.transform(eeg)])
    classifier = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=RANDOM_STATE,
    ).fit(design, y)
    return FittedOdorEEGModel(
        feature_columns=columns,
        robust=robust,
        encoder=encoder,
        classifier=classifier,
        residualizer=residualizer,
        scaler=scaler,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


def tuned_odor_eeg_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    protocol: str,
    feature_columns: Sequence[str],
    robust: bool = False,
    target: Sequence[int] | None = None,
) -> TunedPrediction:
    """Select C by declared grouped inner CV, then refit on the outer training fold."""

    y = train.y_emotion.to_numpy(dtype=int) if target is None else np.asarray(target, dtype=int)
    splits = inner_splits(train, protocol)
    mean_losses = []
    for c_value in C_GRID:
        fold_losses = []
        for fit_indices, validation_indices in splits:
            fitted = fit_odor_eeg_model(
                train.iloc[fit_indices],
                feature_columns,
                c_value,
                robust=robust,
                target=y[fit_indices],
            )
            probability = fitted.predict_proba(train.iloc[validation_indices])
            fold_losses.append(
                log_loss(y[validation_indices], probability, labels=[0, 1])
            )
        mean_losses.append(float(np.mean(fold_losses)))
    best_index = int(np.argmin(mean_losses))
    best_c = C_GRID[best_index]
    fitted = fit_odor_eeg_model(
        train,
        feature_columns,
        best_c,
        robust=robust,
        target=y,
    )
    return TunedPrediction(
        probability=fitted.predict_proba(test),
        best_c=best_c,
        inner_log_loss=mean_losses[best_index],
    )
