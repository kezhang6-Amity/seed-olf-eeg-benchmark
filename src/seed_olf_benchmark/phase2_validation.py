"""Independent split, fingerprint, coverage, and uncertainty checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class SourceFingerprint:
    combined_sha256: str
    files: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float


def source_fingerprint_from_records(
    records: list[tuple[str, int, str]],
) -> SourceFingerprint:
    ordered = tuple(sorted(records))
    encoded = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    return SourceFingerprint(hashlib.sha256(encoded).hexdigest(), ordered)


def compute_source_fingerprint(
    paths: list[Path],
    data_root: Path,
) -> SourceFingerprint:
    """Hash file identity and bytes; mtimes deliberately do not participate."""

    root = data_root.resolve()
    records = []
    for path in paths:
        resolved = path.resolve()
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        records.append((resolved.relative_to(root).as_posix(), resolved.stat().st_size, digest))
    return source_fingerprint_from_records(records)


def inner_splits(frame: pd.DataFrame, protocol: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create the exact deterministic inner splits declared in the Phase 2 spec."""

    positions = np.arange(len(frame))
    if protocol == "loso":
        groups = frame["subject"].to_numpy()
        if len(np.unique(groups)) < 5:
            raise ValueError("LOSO inner tuning requires at least five participants")
        return list(GroupKFold(n_splits=5).split(positions, groups=groups))
    if protocol == "losession":
        sessions = np.sort(frame["session"].unique())
        if len(sessions) != 2:
            raise ValueError("Leave-session-out inner tuning requires two training sessions")
        return [
            (positions[frame.session.to_numpy() == fit], positions[frame.session.to_numpy() == valid])
            for fit, valid in ((sessions[0], sessions[1]), (sessions[1], sessions[0]))
        ]
    raise ValueError(f"Unknown protocol: {protocol}")


def hierarchical_paired_bootstrap(
    differences: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 20260810,
) -> BootstrapInterval:
    """Resample participant clusters, then trials within sampled clusters."""

    required = {"subject", "difference"}
    if not required.issubset(differences):
        raise ValueError(f"Missing bootstrap columns: {sorted(required - set(differences))}")
    grouped = [
        part["difference"].to_numpy(dtype=np.float64)
        for _, part in differences.groupby("subject", sort=True)
    ]
    if not grouped or any(len(values) == 0 for values in grouped):
        raise ValueError("Bootstrap requires at least one observation per participant")
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=np.float64)
    group_sizes = {len(values) for values in grouped}
    if len(group_sizes) == 1:
        values = np.stack(grouped)
        trials_per_group = values.shape[1]
        batch_size = 512
        for start in range(0, n_bootstrap, batch_size):
            stop = min(start + batch_size, n_bootstrap)
            size = stop - start
            sampled_groups = rng.integers(
                0, len(grouped), size=(size, len(grouped))
            )
            sampled_trials = rng.integers(
                0,
                trials_per_group,
                size=(size, len(grouped), trials_per_group),
            )
            replicates[start:stop] = values[
                sampled_groups[:, :, None], sampled_trials
            ].mean(axis=(1, 2))
    else:
        for replicate in range(n_bootstrap):
            sampled_groups = rng.integers(0, len(grouped), size=len(grouped))
            sampled_values = [
                values[rng.integers(0, len(values), size=len(values))]
                for values in (grouped[index] for index in sampled_groups)
            ]
            replicates[replicate] = np.concatenate(sampled_values).mean()
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return BootstrapInterval(
        estimate=float(differences["difference"].mean()),
        lower=float(lower),
        upper=float(upper),
    )


def verify_prediction_coverage(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    protocols: list[str],
    models: list[str],
) -> None:
    """Require every protocol/model to predict every manifest trial exactly once."""

    key = ["subject", "session", "trial"]
    expected_keys = set(map(tuple, manifest[key].to_numpy()))
    problems = []
    for protocol in protocols:
        for model in models:
            part = predictions[
                (predictions.protocol == protocol) & (predictions.model == model)
            ]
            observed = list(map(tuple, part[key].to_numpy()))
            if len(observed) != len(expected_keys) or set(observed) != expected_keys:
                problems.append(f"{protocol}/{model}")
    unexpected = predictions[
        ~predictions.protocol.isin(protocols) | ~predictions.model.isin(models)
    ]
    invalid_probability = (
        not np.isfinite(predictions.probability.to_numpy(dtype=float)).all()
        or not predictions.probability.between(0.0, 1.0).all()
    )
    if problems or not unexpected.empty or invalid_probability:
        raise ValueError(
            "Prediction coverage failure: "
            + ", ".join(
                problems
                or [
                    "invalid probabilities"
                    if invalid_probability
                    else "unexpected protocol/model rows"
                ]
            )
        )
