from __future__ import annotations

import datetime as dt
import pickle

import numpy as np
import pandas as pd
import pytest

from seed_olf_benchmark.data import load_payload, parse_trial_key
from seed_olf_benchmark.features import spectral_summary
from seed_olf_benchmark.metrics import icc_absolute_agreement
from seed_olf_benchmark.models import prior_predictions
from seed_olf_benchmark.riemannian import (
    logeuclidean_vector,
    relative_covariance,
)
from seed_olf_benchmark.validation import outer_folds


def test_parse_trial_key() -> None:
    assert parse_trial_key("01_3_24.pkl") == (1, 3, 24)


def test_restricted_loader_blocks_unexpected_globals(tmp_path) -> None:
    path = tmp_path / "unexpected.pkl"
    path.write_bytes(pickle.dumps(dt.datetime(2026, 8, 10)))
    with pytest.raises(pickle.UnpicklingError, match="Blocked pickle global"):
        load_payload(path)


def test_alpha_sine_has_dominant_alpha_power() -> None:
    sampling_rate = 200
    times = np.arange(3000) / sampling_rate
    rng = np.random.default_rng(7)
    eeg = rng.normal(scale=1e-7, size=(62, 3000))
    eeg[0] += 10e-6 * np.sin(2 * np.pi * 10 * times)
    features = spectral_summary(eeg, sampling_rate)
    assert features["relbp_alpha_mean"] > features["relbp_theta_mean"]
    assert features["relbp_alpha_mean"] > features["relbp_beta_mean"]


def test_odor_prior_uses_training_labels_only() -> None:
    train = pd.DataFrame(
        {
            "y_odor": [1, 1, 2, 2],
            "y_emotion": [0, 0, 1, 0],
        }
    )
    test = pd.DataFrame({"y_odor": [1, 2, 3], "y_emotion": [1, 1, 1]})
    probabilities = prior_predictions(train, test, by_odor=True)
    assert np.allclose(probabilities, [0.25, 0.5, 1 / 3])


def test_icc_is_one_for_identical_sessions() -> None:
    subject_effect = np.linspace(-2, 2, 32)
    matrix = np.column_stack([subject_effect, subject_effect, subject_effect])
    assert np.isclose(icc_absolute_agreement(matrix), 1.0)


def test_relative_covariance_of_identical_inputs_is_identity() -> None:
    covariance = np.diag(np.linspace(1.0, 3.0, 62))
    relative = relative_covariance(covariance, covariance)
    assert np.allclose(relative, np.eye(62), atol=1e-10)
    assert np.allclose(logeuclidean_vector(relative), 0.0, atol=1e-10)


def test_outer_splits_prevent_subject_and_session_leakage() -> None:
    rows = []
    for subject in (1, 2, 3):
        for session in (1, 2, 3):
            for trial in range(1, 9):
                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "trial": trial,
                        "session_fold": f"{session}_{(trial - 1) // 4 + 1}",
                        "y_emotion": trial % 2,
                        "y_odor": (trial - 1) % 4 + 1,
                    }
                )
    frame = pd.DataFrame(rows)

    for _, train, test, _ in outer_folds(frame, "loso"):
        assert set(train.subject).isdisjoint(set(test.subject))

    for _, train, test, _ in outer_folds(frame, "losession"):
        assert train.subject.nunique() == 1
        assert test.subject.nunique() == 1
        assert train.subject.iloc[0] == test.subject.iloc[0]
        assert set(train.session).isdisjoint(set(test.session))
