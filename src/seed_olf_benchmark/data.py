"""Safe SEED-OLF loading and trial-manifest construction."""

from __future__ import annotations

import hashlib
import io
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


FILE_PATTERN = re.compile(r"^(\d{2})_(\d)_(\d{1,2})\.pkl$")
PHASES = ("clean_breathing", "stimulation")
EXPECTED_KEYS = {"X_raw", "y_emotion", "y_odor", "y_sniff"}
EXPECTED_SIGNAL_SHAPE = (62, 3000)
EXPECTED_TRIALS = 32 * 3 * 24


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only NumPy constructors observed in the audited public files."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):  # noqa: ANN201
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")
        return super().find_class(module, name)


def _validate_payload(payload: object, path: Path) -> dict:
    if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected payload schema in {path}")
    eeg = np.asarray(payload["X_raw"])
    if eeg.shape != EXPECTED_SIGNAL_SHAPE or eeg.dtype != np.float64:
        raise ValueError(f"Unexpected EEG array in {path}: {eeg.shape}, {eeg.dtype}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"Non-finite EEG value in {path}")
    return payload


def load_payload(path: Path) -> dict:
    """Load one audited payload and fail closed on schema changes."""

    with path.open("rb") as handle:
        payload = RestrictedNumpyUnpickler(handle).load()
    return _validate_payload(payload, path)


def load_payload_with_digest(path: Path) -> tuple[dict, str, int]:
    """Load and SHA-256 one payload from the same bytes read from disk."""

    raw = path.read_bytes()
    payload = RestrictedNumpyUnpickler(io.BytesIO(raw)).load()
    return _validate_payload(payload, path), hashlib.sha256(raw).hexdigest(), len(raw)


def parse_trial_key(filename: str) -> tuple[int, int, int]:
    match = FILE_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Invalid SEED-OLF filename: {filename}")
    return tuple(map(int, match.groups()))


def build_manifest(data_root: Path) -> pd.DataFrame:
    """Build one row per trial and require complete phase pairing."""

    phase_paths: dict[str, dict[tuple[int, int, int], Path]] = {}
    for phase in PHASES:
        phase_dir = data_root / phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {phase_dir}. See docs/data-access.md for the required layout."
            )
        paths: dict[tuple[int, int, int], Path] = {}
        for path in sorted(phase_dir.glob("*.pkl")):
            key = parse_trial_key(path.name)
            if key in paths:
                raise ValueError(f"Duplicate {phase} trial key: {key}")
            paths[key] = path
        phase_paths[phase] = paths

    clean_keys = set(phase_paths["clean_breathing"])
    stim_keys = set(phase_paths["stimulation"])
    if clean_keys != stim_keys:
        raise ValueError(
            f"Unpaired trials: clean-only={len(clean_keys - stim_keys)}, "
            f"stimulation-only={len(stim_keys - clean_keys)}"
        )

    rows = []
    for subject, session, trial in sorted(stim_keys):
        stim_path = phase_paths["stimulation"][(subject, session, trial)]
        payload = load_payload(stim_path)
        rows.append(
            {
                "subject": subject,
                "session": session,
                "trial": trial,
                "experimental_fold": (trial - 1) // 4 + 1,
                "session_fold": f"{session}_{(trial - 1) // 4 + 1}",
                "stimulation_path": str(stim_path),
                "recovery_path": str(
                    phase_paths["clean_breathing"][(subject, session, trial)]
                ),
                "y_emotion": int(payload["y_emotion"]),
                "y_odor": int(payload["y_odor"]),
                "y_sniff": int(payload["y_sniff"]),
            }
        )
    manifest = pd.DataFrame(rows)
    if len(manifest) != EXPECTED_TRIALS:
        raise ValueError(f"Expected {EXPECTED_TRIALS} trials, found {len(manifest)}")
    return manifest
