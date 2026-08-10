"""Leakage-safe outer-fold definitions."""

from __future__ import annotations

import pandas as pd


def outer_folds(features: pd.DataFrame, protocol: str):
    if protocol == "loso":
        for subject in sorted(features.subject.unique()):
            test_mask = features.subject == subject
            yield (
                f"subject_{subject:02d}",
                features.loc[~test_mask].copy(),
                features.loc[test_mask].copy(),
                features.loc[~test_mask, "subject"].to_numpy(),
            )
        return
    if protocol == "losession":
        for subject in sorted(features.subject.unique()):
            subject_data = features[features.subject == subject]
            for session in sorted(subject_data.session.unique()):
                test_mask = subject_data.session == session
                train = subject_data.loc[~test_mask].copy()
                test = subject_data.loc[test_mask].copy()
                yield (
                    f"subject_{subject:02d}_session_{session}",
                    train,
                    test,
                    train.session_fold.to_numpy(),
                )
        return
    raise ValueError(f"Unknown protocol: {protocol}")
