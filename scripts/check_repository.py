"""Fail CI if public-boundary or compact-result invariants are violated."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".npz", ".pem", ".key"}
FORBIDDEN_NAMES = {
    "manifest.csv",
    "spectral_features.csv",
    "trial_predictions.csv",
}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".cff"}
USER_PATH_MARKERS = ("/" + "Users/", "C:" + "\\Users\\")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    )
    return [ROOT / line for line in output.splitlines() if line]


def check_public_boundary(paths: list[Path]) -> None:
    violations = []
    for path in paths:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES:
            violations.append(str(path.relative_to(ROOT)))
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in USER_PATH_MARKERS):
                violations.append(f"absolute user path in {path.relative_to(ROOT)}")
    if violations:
        raise SystemExit("Public-boundary check failed:\n- " + "\n- ".join(violations))


def check_result_summaries() -> None:
    result_dir = ROOT / "results" / "baseline_v1"
    effects = pd.read_csv(result_dir / "paired_spectral_effects.csv")
    reliability = pd.read_csv(result_dir / "cross_session_reliability.csv")
    summary = pd.read_csv(result_dir / "prediction_summary.csv")
    incremental = pd.read_csv(result_dir / "incremental_value.csv")
    checks = json.loads((result_dir / "validation_checks.json").read_text())

    assert len(effects) == 10
    assert len(reliability) == 40
    assert set(summary.protocol) == {"loso", "losession"}
    assert {"odor_prior", "odor_eeg_stim_logit"}.issubset(summary.model)
    assert set(incremental.baseline) == {"odor_prior"}
    assert checks["status"] == "passed"
    assert checks["n_trials"] == 2304


def main() -> None:
    paths = tracked_files()
    check_public_boundary(paths)
    check_result_summaries()
    print(f"Repository checks passed for {len(paths)} tracked files.")


if __name__ == "__main__":
    main()
