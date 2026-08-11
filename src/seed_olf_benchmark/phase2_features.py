"""Channel-resolved features for the preregistered Phase 2 analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from .data import build_manifest, load_payload_with_digest
from .features import ASSUMED_SAMPLING_RATE_HZ, BANDS_HZ, average_reference
from .phase2_validation import (
    SourceFingerprint,
    compute_source_fingerprint,
    source_fingerprint_from_records,
)


CHANNEL_COUNT = 62
BAND_NAMES = tuple(BANDS_HZ)
DE_VARIANCE_FLOOR = 1e-12
FEATURE_SCHEMA_VERSION = 2
IDENTITY_COLUMNS = [
    "subject",
    "session",
    "trial",
    "experimental_fold",
    "session_fold",
    "y_emotion",
    "y_odor",
    "y_sniff",
]
FEATURE_PREFIXES = (
    "stim_logbp",
    "recovery_logbp",
    "delta_logbp",
    "stim_relbp",
    "recovery_relbp",
    "stim_de",
    "recovery_de",
    "delta_de",
)


def channel_feature_names(prefix: str) -> list[str]:
    """Return deterministic channel-major, band-minor feature names."""

    return [
        f"{prefix}_ch{channel:02d}_{band}"
        for channel in range(CHANNEL_COUNT)
        for band in BAND_NAMES
    ]


def _validate_eeg(eeg: np.ndarray) -> np.ndarray:
    values = np.asarray(eeg, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"Expected ({CHANNEL_COUNT}, samples) EEG, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEG contains non-finite values")
    return values


def channel_spectral_power(
    eeg: np.ndarray,
    sampling_rate_hz: float = ASSUMED_SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Return channel-level log dB and relative Welch power by declared band."""

    referenced = average_reference(_validate_eeg(eeg))
    frequencies, psd = welch(
        referenced,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=int(2 * sampling_rate_hz),
        noverlap=int(sampling_rate_hz),
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask], frequencies[total_mask], axis=-1)
    tiny = np.finfo(np.float64).tiny
    log_power = np.empty((CHANNEL_COUNT, len(BAND_NAMES)), dtype=np.float64)
    relative_power = np.empty_like(log_power)
    for band_index, (band, (low_hz, high_hz)) in enumerate(BANDS_HZ.items()):
        mask = (frequencies >= low_hz) & (
            frequencies <= high_hz if band == "gamma" else frequencies < high_hz
        )
        power = np.trapezoid(psd[:, mask], frequencies[mask], axis=-1)
        log_power[:, band_index] = 10.0 * np.log10(np.maximum(power, tiny))
        relative_power[:, band_index] = power / np.maximum(total_power, tiny)
    return log_power, relative_power


def channel_log_band_power(
    eeg: np.ndarray,
    sampling_rate_hz: float = ASSUMED_SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Return the 62 by 5 matrix used by the primary M2 representation."""

    return channel_spectral_power(eeg, sampling_rate_hz)[0]


def differential_entropy_from_variance(variance: np.ndarray) -> np.ndarray:
    """Gaussian differential entropy in nats with the declared variance floor."""

    values = np.maximum(np.asarray(variance, dtype=np.float64), DE_VARIANCE_FLOOR)
    return 0.5 * np.log(2.0 * np.pi * np.e * values)


def channel_differential_entropy(
    eeg: np.ndarray,
    sampling_rate_hz: float = ASSUMED_SAMPLING_RATE_HZ,
) -> np.ndarray:
    """Return fourth-order zero-phase band-limited Gaussian DE per channel."""

    referenced = average_reference(_validate_eeg(eeg))
    trim_samples = int(sampling_rate_hz)
    if referenced.shape[1] <= 2 * trim_samples + 1:
        raise ValueError("EEG is too short for one-second edge trimming")
    entropy = np.empty((CHANNEL_COUNT, len(BAND_NAMES)), dtype=np.float64)
    nyquist = sampling_rate_hz / 2.0
    for band_index, (low_hz, high_hz) in enumerate(BANDS_HZ.values()):
        sos = butter(
            4,
            [low_hz / nyquist, high_hz / nyquist],
            btype="bandpass",
            output="sos",
        )
        filtered_microvolts = sosfiltfilt(sos, referenced, axis=-1) * 1e6
        variance = np.var(
            filtered_microvolts[:, trim_samples:-trim_samples], axis=-1, ddof=1
        )
        entropy[:, band_index] = differential_entropy_from_variance(variance)
    return entropy


def _phase_features(
    eeg: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_power, relative_power = channel_spectral_power(eeg, sampling_rate_hz)
    entropy = channel_differential_entropy(eeg, sampling_rate_hz)
    return log_power, relative_power, entropy


def extract_channel_feature_table(
    manifest: pd.DataFrame,
    data_root: Path,
    sampling_rate_hz: float = ASSUMED_SAMPLING_RATE_HZ,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, SourceFingerprint, dict]:
    """Stream paired recordings once and create the complete Phase 2 cache."""

    n_trials = len(manifest)
    width = CHANNEL_COUNT * len(BAND_NAMES)
    matrices = {
        prefix: np.empty((n_trials, width), dtype=np.float32)
        for prefix in FEATURE_PREFIXES
    }
    source_records: list[tuple[str, int, str]] = []
    signal_qc = []
    root = data_root.resolve()

    for row_index, trial in enumerate(manifest.itertuples(index=False)):
        phase_values = {}
        for phase, path_value in (
            ("stim", trial.stimulation_path),
            ("recovery", trial.recovery_path),
        ):
            path = Path(path_value).resolve()
            payload, digest, byte_size = load_payload_with_digest(path)
            eeg = np.asarray(payload["X_raw"], dtype=np.float64)
            log_power, relative_power, entropy = _phase_features(eeg, sampling_rate_hz)
            phase_values[phase] = (log_power, relative_power, entropy)
            channel_std = np.std(average_reference(eeg), axis=1)
            entropy_floor = float(differential_entropy_from_variance(np.array([0.0]))[0])
            signal_qc.append(
                {
                    "phase": phase,
                    "peak_abs": float(np.max(np.abs(eeg))),
                    "median_channel_std": float(np.median(channel_std)),
                    "flat_channels": int(np.sum(channel_std == 0.0)),
                    "de_floor_values": int(np.sum(entropy == entropy_floor)),
                }
            )
            source_records.append(
                (path.relative_to(root).as_posix(), byte_size, digest)
            )

        stim_log, stim_rel, stim_de = phase_values["stim"]
        recovery_log, recovery_rel, recovery_de = phase_values["recovery"]
        values_by_prefix = {
            "stim_logbp": stim_log,
            "recovery_logbp": recovery_log,
            "delta_logbp": stim_log - recovery_log,
            "stim_relbp": stim_rel,
            "recovery_relbp": recovery_rel,
            "stim_de": stim_de,
            "recovery_de": recovery_de,
            "delta_de": stim_de - recovery_de,
        }
        for prefix, values in values_by_prefix.items():
            matrices[prefix][row_index] = values.reshape(-1)
        if progress_every and (row_index + 1) % progress_every == 0:
            print(f"Phase 2 features: {row_index + 1}/{n_trials} trials", flush=True)

    feature_blocks = [manifest[IDENTITY_COLUMNS].reset_index(drop=True)]
    for prefix in FEATURE_PREFIXES:
        feature_blocks.append(
            pd.DataFrame(matrices[prefix], columns=channel_feature_names(prefix))
        )
    features = pd.concat(feature_blocks, axis=1)
    feature_values = features.drop(columns=IDENTITY_COLUMNS).to_numpy(dtype=np.float64)
    qc_frame = pd.DataFrame(signal_qc)
    feature_profile_by_prefix = {}
    for prefix, values in matrices.items():
        numeric = values.astype(np.float64, copy=False)
        feature_profile_by_prefix[prefix] = {
            "finite_rate": float(np.isfinite(numeric).mean()),
            "zero_variance_features": int(np.sum(np.var(numeric, axis=0) == 0.0)),
            "quantiles": {
                str(quantile): float(np.quantile(numeric, quantile))
                for quantile in (0.001, 0.01, 0.5, 0.99, 0.999)
            },
        }
    profile = {
        "n_trials": n_trials,
        "n_subjects": int(manifest.subject.nunique()),
        "n_sessions": int(manifest[["subject", "session"]].drop_duplicates().shape[0]),
        "unique_trial_keys": int(
            manifest[["subject", "session", "trial"]].drop_duplicates().shape[0]
        ),
        "phase_pair_coverage": 1.0,
        "feature_count": int(feature_values.shape[1]),
        "finite_feature_rate": float(np.isfinite(feature_values).mean()),
        "zero_variance_features": int(np.sum(np.var(feature_values, axis=0) == 0.0)),
        "feature_quantiles": {
            str(quantile): float(np.quantile(feature_values, quantile))
            for quantile in (0.001, 0.01, 0.5, 0.99, 0.999)
        },
        "feature_profile_by_prefix": feature_profile_by_prefix,
        "signal_qc_by_phase": {
            phase: {
                "records": int(len(part)),
                "flat_channel_records": int((part.flat_channels > 0).sum()),
                "maximum_flat_channels": int(part.flat_channels.max()),
                "de_floor_value_rate": float(
                    part.de_floor_values.sum()
                    / (len(part) * CHANNEL_COUNT * len(BAND_NAMES))
                ),
                "peak_abs_quantile_99": float(part.peak_abs.quantile(0.99)),
                "median_channel_std_quantile_01": float(
                    part.median_channel_std.quantile(0.01)
                ),
            }
            for phase, part in qc_frame.groupby("phase")
        },
        "label_counts": {
            str(key): int(value) for key, value in manifest.y_emotion.value_counts().items()
        },
        "odor_counts": {
            str(key): int(value) for key, value in manifest.y_odor.value_counts().items()
        },
    }
    return features, source_fingerprint_from_records(source_records), profile


def _feature_columns() -> list[str]:
    return [name for prefix in FEATURE_PREFIXES for name in channel_feature_names(prefix)]


def write_channel_feature_cache(
    output_dir: Path,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    fingerprint: SourceFingerprint,
    profile: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    columns = _feature_columns()
    np.savez_compressed(
        output_dir / "channel_features.npz",
        values=features[columns].to_numpy(dtype=np.float32),
        columns=np.asarray(columns),
    )
    cache_metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "sampling_rate_hz": ASSUMED_SAMPLING_RATE_HZ,
        "de_variance_unit": "microvolt_squared",
        "source_combined_sha256": fingerprint.combined_sha256,
        "source_file_count": len(fingerprint.files),
        "feature_count": len(columns),
        "trial_count": len(manifest),
    }
    (output_dir / "channel_feature_cache.json").write_text(
        json.dumps(cache_metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "data_quality_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        fingerprint.files,
        columns=["relative_path", "byte_size", "sha256"],
    ).to_csv(output_dir / "source_fingerprints.csv", index=False)


def load_channel_feature_cache(
    output_dir: Path,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a cache only after exact schema, configuration, and source checks."""

    manifest = pd.read_csv(output_dir / "manifest.csv")
    metadata = json.loads(
        (output_dir / "channel_feature_cache.json").read_text(encoding="utf-8")
    )
    if metadata.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Channel feature cache schema mismatch")
    if metadata.get("sampling_rate_hz") != ASSUMED_SAMPLING_RATE_HZ:
        raise ValueError("Channel feature cache sampling-rate mismatch")
    paths = [
        Path(path)
        for path in manifest[["stimulation_path", "recovery_path"]].to_numpy().reshape(-1)
    ]
    fingerprint = compute_source_fingerprint(paths, data_root)
    if fingerprint.combined_sha256 != metadata.get("source_combined_sha256"):
        raise ValueError("Channel feature cache source fingerprint mismatch")
    with np.load(output_dir / "channel_features.npz", allow_pickle=False) as cache:
        values = cache["values"]
        columns = cache["columns"].astype(str).tolist()
    expected_columns = _feature_columns()
    if columns != expected_columns or values.shape != (len(manifest), len(expected_columns)):
        raise ValueError("Channel feature cache column or shape mismatch")
    features = pd.concat(
        [
            manifest[IDENTITY_COLUMNS].reset_index(drop=True),
            pd.DataFrame(values, columns=columns),
        ],
        axis=1,
    )
    profile = json.loads(
        (output_dir / "data_quality_profile.json").read_text(encoding="utf-8")
    )
    return manifest, features, profile


def load_or_build_channel_features(
    data_root: Path,
    output_dir: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cache_files = (
        output_dir / "manifest.csv",
        output_dir / "channel_features.npz",
        output_dir / "channel_feature_cache.json",
        output_dir / "data_quality_profile.json",
    )
    if not force and all(path.exists() for path in cache_files):
        return load_channel_feature_cache(output_dir, data_root)
    manifest = build_manifest(data_root)
    features, fingerprint, profile = extract_channel_feature_table(manifest, data_root)
    write_channel_feature_cache(output_dir, manifest, features, fingerprint, profile)
    return manifest, features, profile
