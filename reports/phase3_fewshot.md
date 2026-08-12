# Phase 3: Few-Shot Cross-Session Personalization

## Decision

**Classification: passed.** The minimum feasible calibration burden is **4 Session-1 trials per participant**—one trial per odor. This result supports a narrowly scoped claim: a pooled odor-response prior can be improved slightly and reproducibly by updating it with a participant's own labeled trials.

## Design

- 32 participants; calibration uses Session 1 only and prediction evaluates all 48 trials in Sessions 2–3.
- Calibration sizes: 0, 4, 8, 16, and 24; sizes 4–16 use 200 deterministic label-blind balanced subsets.
- Primary P2 model: leave-target-participant-out odor-specific Beta prior, updated by target Session-1 labels.
- P1 direct individual frequency and P4 fixed Phase-2 M5 EEG predictions are secondary comparators. P3 penalized odor logistic regression is implemented but not part of this completed primary run.
- Uncertainty: paired hierarchical bootstrap (participants → calibration subsets → held-out trials), 10,000 replicates.
- Negative control: 20 within-participant Session-1 label permutations across odors. No permuted run passed the statistical gate, and observed P2 improvement exceeded the corresponding permutation maximum at every nonzero size.

## Primary P2 results

| Calibration trials | Log-loss improvement over P0 ↑ | 95% CI | Brier improvement ↑ | Participants improved | Maximum permuted improvement |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.000299 | [0.000052, 0.000635] | 0.000073 | 78.1% | 0.000158 |
| 8 | 0.000542 | [0.000148, 0.001021] | 0.000129 | 78.1% | 0.000309 |
| 16 | 0.001118 | [0.000404, 0.001976] | 0.000269 | 78.1% | 0.000575 |
| 24 | 0.001637 | [0.000668, 0.002847] | 0.000395 | 78.1% | 0.000830 |

At 4 trials, the effect is statistically stable but small: Δlog-loss 0.000299, 95% CI [0.000052, 0.000635], with 78.1% of participants improved. The effect grows monotonically through 24 trials, but the effect size remains modest.

## Comparator findings

Direct individual frequencies were harmful at every calibration size (at 4 trials, Δlog-loss -0.1301), demonstrating why pooled shrinkage is required. The fixed Phase-2 EEG comparator P4 was worse than the odor prior at all sizes (Δlog-loss -0.0221); the current EEG representation should not be used for this personalization task.

## Audit status and boundaries

- Independent recomputation maximum discrepancy: 1.67e-16.
- 20 negative-control permutations completed; none passed the statistical gate.
- This does not establish relaxation, clinical benefit, product efficacy, or an EEG-driven treatment effect. `y_emotion` remains a public binary subjective-valence proxy.
- The result is limited to SEED-OLF's odor categories, sessions, labels, and the declared cross-session split. External replication is required before a collection protocol is finalized.

## Implication for future collection

For the first proprietary study, retain a randomized, odor-balanced Session-1 calibration block of at least four labeled trials per participant. Use the P2 posterior as the transparent operational baseline. Add EEG + PPG/HRV + EDA only as prospectively evaluated incremental modalities; do not assume they will improve this baseline.
