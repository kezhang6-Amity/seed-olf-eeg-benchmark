from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_olf_benchmark.phase3 import (
    CALIBRATION_SIZES,
    balanced_calibration_subsets,
    empirical_bayes_probabilities,
    pooled_odor_prior,
    phase3_hierarchical_bootstrap,
    verify_phase3_prediction_coverage,
)
from seed_olf_benchmark.phase3_experiment import negative_control_comparisons


def _session_one_frame() -> pd.DataFrame:
    rows = []
    for odor in range(1, 5):
        for trial in range(1, 7):
            rows.append(
                {
                    "subject": 1,
                    "session": 1,
                    "trial": (odor - 1) * 6 + trial,
                    "y_odor": odor,
                    "y_emotion": trial % 2,
                }
            )
    return pd.DataFrame(rows)


def test_balanced_subsets_have_declared_odor_counts_and_are_deterministic() -> None:
    session_one = _session_one_frame()
    first = balanced_calibration_subsets(session_one, 8, seed=23, n_samples=10)
    second = balanced_calibration_subsets(session_one, 8, seed=23, n_samples=10)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    for subset in first:
        sampled = session_one.iloc[list(subset)]
        assert sampled.y_odor.value_counts().to_dict() == {1: 2, 2: 2, 3: 2, 4: 2}


def test_size_zero_and_24_sampling_rules() -> None:
    session_one = _session_one_frame()
    assert balanced_calibration_subsets(session_one, 0, seed=7) == [()]
    full = balanced_calibration_subsets(session_one, 24, seed=7)
    assert full == [tuple(range(24))]


def test_empirical_bayes_matches_declared_beta_update_and_size_zero_prior() -> None:
    pooled = pd.DataFrame(
        {
            "y_odor": [1, 1, 1, 2, 2],
            "y_emotion": [1, 1, 0, 0, 0],
        }
    )
    calibration = pd.DataFrame({"y_odor": [1, 1, 2], "y_emotion": [0, 1, 1]})
    test = pd.DataFrame({"y_odor": [1, 2, 3]})
    p0 = pooled_odor_prior(pooled, test)
    p2_zero = empirical_bayes_probabilities(pooled, calibration.iloc[:0], test)
    p2 = empirical_bayes_probabilities(pooled, calibration, test)
    assert np.allclose(p0, p2_zero)
    assert np.allclose(p2[:2], [4 / 7, 2 / 5])
    assert np.isclose(p2[2], 0.5)


def test_calibration_never_contains_target_test_sessions() -> None:
    session_one = _session_one_frame()
    target_test = pd.DataFrame(
        {"subject": [1, 1], "session": [2, 3], "trial": [1, 1]}
    )
    for subset in balanced_calibration_subsets(session_one, 16, seed=31, n_samples=10):
        sampled = session_one.iloc[list(subset)]
        assert set(sampled.session) == {1}
        assert set(map(tuple, sampled[["subject", "session", "trial"]].to_numpy())).isdisjoint(
            set(map(tuple, target_test.to_numpy()))
        )


def test_phase3_coverage_rejects_duplicate_or_missing_predictions() -> None:
    expected = pd.DataFrame(
        {
            "subject": [1, 1],
            "session": [2, 3],
            "trial": [1, 1],
        }
    )
    predictions = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1],
            "session": [2, 3, 2, 3],
            "trial": [1, 1, 1, 1],
            "calibration_size": [4] * 4,
            "replicate": [0] * 4,
            "model": ["P0", "P0", "P2", "P2"],
            "probability": [0.4, 0.4, 0.5, 0.5],
        }
    )
    verify_phase3_prediction_coverage(
        predictions, expected, [4], ["P0", "P2"], sampled_replicates=1
    )
    with pytest.raises(ValueError, match="coverage"):
        verify_phase3_prediction_coverage(
            predictions.iloc[:-1], expected, [4], ["P0", "P2"], sampled_replicates=1
        )


def test_declared_sizes_are_evenly_divisible_by_four() -> None:
    assert CALIBRATION_SIZES == (0, 4, 8, 16, 24)


def test_phase3_bootstrap_is_reproducible() -> None:
    differences = pd.DataFrame(
        {
            "subject": [1, 1, 1, 1, 2, 2, 2, 2],
            "replicate": [0, 0, 1, 1, 0, 0, 1, 1],
            "difference": [0.1, 0.2, 0.3, 0.4, -0.1, 0.0, 0.2, 0.3],
        }
    )
    first = phase3_hierarchical_bootstrap(differences, n_bootstrap=200, seed=13)
    second = phase3_hierarchical_bootstrap(differences, n_bootstrap=200, seed=13)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_negative_control_uses_permuted_session_one_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = []
    for subject in (1, 2):
        for session in (1, 2, 3):
            for odor in range(1, 5):
                for repetition in range(1, 7):
                    rows.append(
                        {
                            "subject": subject,
                            "session": session,
                            "trial": (odor - 1) * 6 + repetition,
                            "y_odor": odor,
                            "y_emotion": (subject + session + odor + repetition) % 2,
                        }
                    )
    manifest = pd.DataFrame(rows)
    monkeypatch.setattr(
        "seed_olf_benchmark.phase3_experiment.CALIBRATION_SIZES", (0,)
    )
    result = negative_control_comparisons(manifest, permutation=1, n_bootstrap=20)
    assert result.candidate.tolist() == ["P2"]
    assert result.mean_log_loss_improvement.iloc[0] == 0.0


def test_negative_control_permutation_breaks_odor_label_counts() -> None:
    session_one = _session_one_frame()
    session_one["y_emotion"] = [1] * 6 + [0] * 18
    from seed_olf_benchmark.phase3_experiment import _permute_session_one_labels

    permuted = _permute_session_one_labels(session_one, seed=4)
    assert permuted.sum() == session_one.y_emotion.sum()
    assert not np.array_equal(permuted, session_one.y_emotion.to_numpy())
    assert np.any(permuted[session_one.y_odor.to_numpy() != 1] == 1)
