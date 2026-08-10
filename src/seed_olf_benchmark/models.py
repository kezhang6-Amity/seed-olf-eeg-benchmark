"""Train-fold-only probabilistic baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def smoothed_prior(successes: int, count: int, alpha: float = 1.0) -> float:
    return (successes + alpha) / (count + 2.0 * alpha)


def prior_predictions(train: pd.DataFrame, test: pd.DataFrame, by_odor: bool) -> np.ndarray:
    """Generate training-fold-only priors with Laplace smoothing."""

    fallback = smoothed_prior(int(train.y_emotion.sum()), len(train))
    if not by_odor:
        return np.full(len(test), fallback)
    odor_priors = {
        int(odor): smoothed_prior(int(part.y_emotion.sum()), len(part))
        for odor, part in train.groupby("y_odor")
    }
    return test.y_odor.map(odor_priors).fillna(fallback).to_numpy(dtype=float)


def _model_pipeline(feature_columns: Sequence[str], include_odor: bool) -> Pipeline:
    transformers = []
    if include_odor:
        transformers.append(("odor", OneHotEncoder(handle_unknown="ignore"), ["y_odor"]))
    if feature_columns:
        transformers.append(("eeg", StandardScaler(), list(feature_columns)))
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", LogisticRegression(max_iter=3_000, solver="lbfgs")),
        ]
    )


def tuned_logistic_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    include_odor: bool,
    inner_groups: Sequence,
    random_state: int = 20260810,
) -> tuple[np.ndarray, float]:
    """Tune a compact logistic model with grouped inner cross-validation."""

    groups = np.asarray(inner_groups)
    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two inner groups are required")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        _model_pipeline(feature_columns, include_odor),
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="neg_log_loss",
        cv=splitter,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(train, train.y_emotion.to_numpy(), groups=groups)
    return search.predict_proba(test)[:, 1], float(search.best_params_["model__C"])
