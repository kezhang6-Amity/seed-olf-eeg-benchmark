# 0002 — Primary evaluation gate

**Status:** Accepted

**Date:** 2026-08-10

## Decision

Use held-out log loss versus the strongest eligible train-fold non-EEG baseline as the primary model comparison. Estimate uncertainty at the participant level. Treat Brier score and calibration as guardrails; balanced accuracy, AUROC, and PR-AUC are secondary.

## Rationale

Odor identity agrees with subjective valence on roughly 80% of public trials. Overall accuracy or AUROC can therefore over-credit models that reconstruct odor identity or produce poorly calibrated rankings.
