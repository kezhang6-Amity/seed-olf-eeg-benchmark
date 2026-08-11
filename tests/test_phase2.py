from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_olf_benchmark.phase2_features import (
    channel_feature_names,
    channel_differential_entropy,
    channel_log_band_power,
    differential_entropy_from_variance,
)
from seed_olf_benchmark.phase2_models import OdorResidualizer
from seed_olf_benchmark.phase2_experiment import permute_target_within_subject_odor
from seed_olf_benchmark.phase2_metrics import evaluate_success_gate
from seed_olf_benchmark.phase2_models import tuned_odor_eeg_predictions
from seed_olf_benchmark.phase2_validation import (
    compute_source_fingerprint,
    hierarchical_paired_bootstrap,
    inner_splits,
    verify_prediction_coverage,
)


def test_channel_alpha_sine_peaks_in_alpha_for_injected_channel() -> None:
    sampling_rate = 200.0
    times = np.arange(3000) / sampling_rate
    rng = np.random.default_rng(7)
    eeg = rng.normal(scale=1e-7, size=(62, 3000))
    eeg[0] += 10e-6 * np.sin(2 * np.pi * 10 * times)

    log_power = channel_log_band_power(eeg, sampling_rate)

    assert log_power.shape == (62, 5)
    assert int(np.argmax(log_power[0])) == 2
    assert log_power[0, 2] > log_power[1, 2] + 20.0


def test_differential_entropy_matches_gaussian_variance_formula() -> None:
    variances = np.array([1.0, 4.0])
    expected = 0.5 * np.log(2.0 * np.pi * np.e * variances)
    assert np.allclose(differential_entropy_from_variance(variances), expected)


def test_differential_entropy_respects_signal_scale() -> None:
    rng = np.random.default_rng(13)
    eeg = rng.normal(scale=10e-6, size=(62, 3000))
    original = channel_differential_entropy(eeg)
    doubled = channel_differential_entropy(2.0 * eeg)
    assert np.allclose(doubled - original, np.log(2.0), atol=1e-8)


def test_channel_feature_names_are_deterministic() -> None:
    names = channel_feature_names("stim_logbp")
    assert len(names) == 310
    assert names[:5] == [
        "stim_logbp_ch00_delta",
        "stim_logbp_ch00_theta",
        "stim_logbp_ch00_alpha",
        "stim_logbp_ch00_beta",
        "stim_logbp_ch00_gamma",
    ]
    assert names[-1] == "stim_logbp_ch61_gamma"


def test_odor_residualizer_uses_training_statistics_and_unseen_fallback() -> None:
    train = pd.DataFrame(
        {
            "y_odor": [1, 1, 2, 2],
            "f1": [1.0, 3.0, 10.0, 14.0],
            "f2": [2.0, 4.0, 20.0, 24.0],
        }
    )
    test = pd.DataFrame(
        {
            "y_odor": [1, 3],
            "f1": [1_000_000.0, 8.0],
            "f2": [2_000_000.0, 12.5],
        }
    )
    residualizer = OdorResidualizer(["f1", "f2"]).fit(train)
    odor_means_before = residualizer.odor_means_.copy()
    global_mean_before = residualizer.global_mean_.copy()

    transformed = residualizer.transform(test)

    assert np.allclose(transformed[0], [999_998.0, 1_999_997.0])
    assert np.allclose(transformed[1], [1.0, 0.0])
    assert np.array_equal(residualizer.odor_means_, odor_means_before)
    assert np.array_equal(residualizer.global_mean_, global_mean_before)


def test_inner_splits_preserve_declared_groups() -> None:
    loso = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(1, 11), 4),
            "session": np.tile([1, 1, 2, 2], 10),
            "y_emotion": np.tile([0, 1, 0, 1], 10),
        }
    )
    loso_splits = inner_splits(loso, "loso")
    assert len(loso_splits) == 5
    for fit_indices, validation_indices in loso_splits:
        assert set(loso.iloc[fit_indices].subject).isdisjoint(
            set(loso.iloc[validation_indices].subject)
        )

    losession = pd.DataFrame(
        {
            "subject": [1] * 8,
            "session": [1] * 4 + [2] * 4,
            "y_emotion": [0, 1] * 4,
        }
    )
    session_splits = inner_splits(losession, "losession")
    assert len(session_splits) == 2
    for fit_indices, validation_indices in session_splits:
        assert set(losession.iloc[fit_indices].session).isdisjoint(
            set(losession.iloc[validation_indices].session)
        )


def test_source_fingerprint_changes_with_bytes_not_mtime(tmp_path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")

    initial = compute_source_fingerprint([first, second], tmp_path)
    first.touch()
    touched = compute_source_fingerprint([second, first], tmp_path)
    assert touched.combined_sha256 == initial.combined_sha256

    first.write_bytes(b"ALPHA")
    changed = compute_source_fingerprint([first, second], tmp_path)
    assert changed.combined_sha256 != initial.combined_sha256


def test_hierarchical_bootstrap_is_paired_and_reproducible() -> None:
    differences = pd.DataFrame(
        {
            "subject": [1, 1, 2, 2],
            "difference": [0.1, 0.2, -0.1, 0.4],
        }
    )
    first = hierarchical_paired_bootstrap(differences, n_bootstrap=200, seed=17)
    second = hierarchical_paired_bootstrap(differences, n_bootstrap=200, seed=17)
    assert first == second
    assert np.isclose(first.estimate, 0.15)
    assert first.lower <= first.estimate <= first.upper


def test_prediction_coverage_rejects_duplicate_or_missing_trials() -> None:
    manifest = pd.DataFrame(
        {
            "subject": [1, 1],
            "session": [1, 1],
            "trial": [1, 2],
        }
    )
    valid = pd.DataFrame(
        {
            "protocol": ["loso"] * 4,
            "model": ["M0", "M0", "M2", "M2"],
            "subject": [1, 1, 1, 1],
            "session": [1, 1, 1, 1],
            "trial": [1, 2, 1, 2],
            "probability": [0.4, 0.6, 0.3, 0.7],
        }
    )
    verify_prediction_coverage(valid, manifest, ["loso"], ["M0", "M2"])

    with pytest.raises(ValueError, match="coverage"):
        verify_prediction_coverage(
            pd.concat([valid, valid.iloc[[0]]], ignore_index=True),
            manifest,
            ["loso"],
            ["M0", "M2"],
        )


def test_permutation_preserves_subject_odor_label_counts() -> None:
    frame = pd.DataFrame(
        {
            "subject": np.repeat([1, 2], 8),
            "y_odor": np.tile(np.repeat([1, 2], 4), 2),
            "y_emotion": np.tile([0, 0, 1, 1, 0, 1, 1, 1], 2),
        }
    )
    permuted = permute_target_within_subject_odor(frame, np.random.default_rng(11))
    before = frame.assign(target=frame.y_emotion).groupby(["subject", "y_odor"]).target.sum()
    after = frame.assign(target=permuted).groupby(["subject", "y_odor"]).target.sum()
    assert before.equals(after)


def test_tuned_model_returns_finite_grouped_predictions() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for subject in range(1, 11):
        for trial in range(4):
            label = trial % 2
            rows.append(
                {
                    "subject": subject,
                    "session": 1,
                    "trial": trial + 1,
                    "y_odor": trial % 2 + 1,
                    "y_emotion": label,
                    "f1": label + rng.normal(scale=0.1),
                    "f2": rng.normal(),
                }
            )
    frame = pd.DataFrame(rows)
    train = frame[frame.subject != 10].reset_index(drop=True)
    test = frame[frame.subject == 10].reset_index(drop=True)
    result = tuned_odor_eeg_predictions(train, test, "loso", ["f1", "f2"])
    assert result.best_c in (0.001, 0.01, 0.1, 1.0, 10.0)
    assert np.isfinite(result.probability).all()
    assert np.all((result.probability >= 0.0) & (result.probability <= 1.0))


def test_primary_success_gate_requires_every_condition() -> None:
    common = {
        "candidate": "M2",
        "mean_log_loss_improvement": 0.02,
        "log_loss_ci95_lower": 0.001,
        "mean_brier_improvement": 0.01,
        "brier_ci95_lower": -0.004,
        "participants_improved_rate": 0.6,
        "delete_odor_1_log_loss_improvement": 0.01,
        "delete_odor_2_log_loss_improvement": 0.01,
        "delete_odor_3_log_loss_improvement": 0.01,
        "delete_odor_4_log_loss_improvement": 0.01,
    }
    comparisons = pd.DataFrame(
        [
            {"protocol": "loso", **common},
            {"protocol": "losession", **common},
        ]
    )
    classification, checks = evaluate_success_gate(comparisons, audits_passed=True)
    assert classification == "passed"
    assert all(checks.values())

    comparisons.loc[comparisons.protocol == "loso", "log_loss_ci95_lower"] = 0.0
    classification, checks = evaluate_success_gate(comparisons, audits_passed=True)
    assert classification == "signal_without_passage"
    assert not checks["loso_log_loss_ci_lower_above_zero"]
