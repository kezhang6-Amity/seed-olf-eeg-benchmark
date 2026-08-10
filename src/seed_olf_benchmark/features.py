"""Interpretable EEG feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from .data import load_payload


ASSUMED_SAMPLING_RATE_HZ = 200.0
BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def average_reference(eeg: np.ndarray) -> np.ndarray:
    """Apply common-average reference without modifying the input."""

    return eeg - eeg.mean(axis=0, keepdims=True)


def spectral_summary(
    eeg: np.ndarray,
    sampling_rate_hz: float = ASSUMED_SAMPLING_RATE_HZ,
) -> dict[str, float]:
    """Compute channel-aggregated Welch band-power features."""

    referenced = average_reference(np.asarray(eeg, dtype=np.float64))
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

    features: dict[str, float] = {
        "peak_uv": float(np.max(np.abs(referenced)) * 1e6),
        "median_channel_std_uv": float(np.median(np.std(referenced, axis=1)) * 1e6),
    }
    for band, (low_hz, high_hz) in BANDS_HZ.items():
        mask = (frequencies >= low_hz) & (
            frequencies <= high_hz if band == "gamma" else frequencies < high_hz
        )
        band_power = np.trapezoid(psd[:, mask], frequencies[mask], axis=-1)
        log_power_db = 10.0 * np.log10(np.maximum(band_power, tiny))
        relative_power = band_power / np.maximum(total_power, tiny)
        features[f"logbp_{band}_mean"] = float(np.mean(log_power_db))
        features[f"logbp_{band}_std"] = float(np.std(log_power_db))
        features[f"relbp_{band}_mean"] = float(np.mean(relative_power))
        features[f"relbp_{band}_std"] = float(np.std(relative_power))
    return features


def extract_feature_table(
    manifest: pd.DataFrame,
    sampling_rate_hz: float = ASSUMED_SAMPLING_RATE_HZ,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Stream paired files and return one feature row per trial."""

    rows = []
    for index, trial in enumerate(manifest.itertuples(index=False), start=1):
        stimulation = load_payload(Path(trial.stimulation_path))["X_raw"]
        recovery = load_payload(Path(trial.recovery_path))["X_raw"]
        stim_features = spectral_summary(stimulation, sampling_rate_hz)
        recovery_features = spectral_summary(recovery, sampling_rate_hz)
        row = {
            "subject": trial.subject,
            "session": trial.session,
            "trial": trial.trial,
            "experimental_fold": trial.experimental_fold,
            "session_fold": trial.session_fold,
            "y_emotion": trial.y_emotion,
            "y_odor": trial.y_odor,
            "y_sniff": trial.y_sniff,
        }
        row.update({f"stim_{name}": value for name, value in stim_features.items()})
        row.update({f"recovery_{name}": value for name, value in recovery_features.items()})
        row.update(
            {
                f"delta_{name}": stim_features[name] - recovery_features[name]
                for name in stim_features
            }
        )
        rows.append(row)
        if progress_every and index % progress_every == 0:
            print(f"Extracted {index}/{len(manifest)} paired trials", flush=True)
    return pd.DataFrame(rows)
