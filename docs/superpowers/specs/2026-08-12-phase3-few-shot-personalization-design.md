# Phase 3 Few-Shot Cross-Session Personalization Design

**Date:** 2026-08-12

**Status:** Approved design; implementation pending written-spec review

## 1. Purpose and claim boundary

Test how many labeled session-1 trials are needed before a participant-specific, odor-conditioned probability model improves prediction of that participant's session-2 and session-3 public SEED-OLF binary subjective-valence labels.

The target `y_emotion` remains a methodological proxy. This experiment does not measure relaxation, clinical benefit, product efficacy, or an objective emotional state. It estimates a public-data calibration burden under a fixed four-odor paradigm.

## 2. Population, unit, and separation

- Population: all 32 public participants.
- Calibration source: only session 1 of the target participant.
- Test source: every trial in that participant's sessions 2 and 3; these 48 trials are never used for fitting, feature processing, hyperparameter choice, or sampling decisions.
- Unit: participant-session-trial.
- Trial key: `(subject, session, trial)`.
- Calibration sizes: 0, 4, 8, 16, and 24 trials.

The primary analysis is within-participant, cross-session prediction. It is not a claim of unseen-participant generalization.

## 3. Balanced calibration sampling

Each participant has six session-1 trials for each of four odors. For calibration size `k`, select exactly `k / 4` trials from each odor:

- 0: none;
- 4: one per odor;
- 8: two per odor;
- 16: four per odor;
- 24: all six per odor.

For sizes 4, 8, and 16, enumerate every possible odor-balanced calibration subset: respectively `6^4 = 1,296`, `15^4 = 50,625`, and `15^4 = 50,625` subsets per participant. Exact enumeration is computationally unnecessary and would over-weight this public realization, so use 200 deterministic stratified samples without replacement for each eligible participant-size pair. At size 24, use the single available subset. Seeds are derived deterministically from the global seed, participant, and calibration size.

The same sampled calibration subset is evaluated for every model in that replicate. Sampling uses trial identity and odor only, never labels, test features, or test outcomes.

## 4. Information allowed at prediction time

The public-data deployment analogue permits the following information:

1. data from the other 31 participants, including their labels;
2. target participant session-1 labels selected by the calibration policy;
3. target participant session-1 trial odor labels;
4. target participant session-2/3 odor labels at prediction time.

Target participant sessions 2–3 labels and EEG features do not affect calibration fitting. Session-2/3 odor labels are allowed because odor is an experimental input known before a response is recorded.

## 5. Model ladder

- **P0:** pooled leave-target-participant-out odor prior. For each odor, estimate a Beta(1,1)-smoothed positive-label probability from the other 31 participants. This is the size-0 non-personalized baseline.
- **P1:** direct individual odor frequency. For each odor, estimate the target participant's session-1 positive-label frequency with Beta(1,1) smoothing. At size 0, P1 equals P0.
- **P2 primary:** hierarchical empirical-Bayes odor calibration. For each odor, start from P0's pooled count-derived Beta posterior and update it with the target participant's selected session-1 labels. This is the primary few-shot model.
- **P3 sensitivity:** pooled-plus-target logistic calibration using odor one-hot indicators and L2 regularization. The regularization parameter is selected by five-fold participant-grouped cross-validation using only the other 31 participants, with `C` in `[0.001, 0.01, 0.1, 1, 10]`, `lbfgs`, `max_iter=5000`, no class weighting, and seed `20260812`; it is then held fixed for every target participant and calibration size.
- **P4 exploratory EEG comparator:** the Phase 2 M5 leave-target-participant-out probability, reused without target calibration or refitting. It is a fixed failed-EEG comparator, not a candidate personalization model.

P2 does not use target EEG. Its purpose is to establish the minimum label burden required for a stable individualized odor-response curve before assigning future value to EEG features. P4 receives the exact same sessions-2/3 held-out trials but does not receive session-1 labels, so it cannot gain an advantage from calibration sampling.

## 6. Empirical-Bayes definition

For odor `o`, the other-participant pooled labels define:

`alpha_o = 1 + positive_count_o`

`beta_o = 1 + negative_count_o`.

For a target calibration subset with `positive_target_o` and `negative_target_o`, P2 predicts:

`(alpha_o + positive_target_o) / (alpha_o + beta_o + positive_target_o + negative_target_o)`.

At size 0, P2 equals P0 exactly. At size 24, P2 uses every session-1 label but still retains the prespecified pooled prior rather than replacing it with an unstable individual frequency.

## 7. Evaluation and uncertainty

For each participant, calibration size, replicate, model, and held-out trial, store one probability. Aggregate the 48 sessions-2/3 held-out trials within participant first, then summarize across participants.

Primary endpoint: participant-mean held-out log-loss improvement of P2 over P0 at each calibration size. Positive values favor P2.

Secondary endpoints: Brier improvement, calibration intercept/slope, ten-bin equal-frequency ECE, balanced accuracy, AUROC, PR-AUC, per-session performance, and replicate variability.

Use a paired hierarchical percentile bootstrap with 10,000 replicates and seed `20260812`: resample participants, then calibration replicates within participant, then held-out trials within participant-replicate. Report 95% intervals.

## 8. Primary decision rule

For each nonzero calibration size, P2 is considered to show reliable improvement only when all conditions hold:

1. the 95% lower interval for log-loss improvement over P0 is greater than zero;
2. P2 Brier point improvement is non-negative;
3. P2 Brier 95% lower interval is at least -0.005;
4. more than half of participants have a positive mean log-loss improvement;
5. the session-2 and session-3 mean log-loss improvements are both non-negative;
6. all leakage, sampling, coverage, metric, and cache audits pass.

The minimum feasible calibration burden is the smallest size that passes every condition. If no size passes, the result is a negative feasibility finding rather than a recommendation to increase model complexity.

P1, P3, P4, and comparisons among sizes are secondary. Their p-values, if reported, are controlled by Benjamini-Hochberg within the declared secondary family. They cannot substitute for P2's primary gate.

## 9. Negative controls and audits

Run 20 deterministic calibration-label permutation controls. Within each target participant and odor, permute only session-1 labels before sampling and fitting; retain the sessions-2/3 test labels unchanged. No permuted P2 run may pass the primary gate, and its median improvement at every size must be less than or equal to zero.

Automated checks include:

- each calibration subset contains exactly `k / 4` trials per odor;
- sessions 2–3 trial keys never occur in calibration subsets;
- calibration sampling is deterministic from its seed and independent of labels;
- P2 equals P0 exactly at size 0;
- P2's posterior mean follows the declared Beta update on a toy example;
- all models receive identical calibration subsets within a replicate;
- P0 excludes every target participant label;
- each expected held-out trial is predicted exactly once per model, participant, size, and replicate;
- an independent validator recomputes headline metrics and gate status;
- feature-cache fingerprints match before P4 uses EEG;
- the reader-facing notebook executes from top to bottom.

## 10. Outputs and interpretation

Local ignored outputs include full calibration assignments, trial-level probabilities, permutation predictions, and caches. Public outputs contain only compact calibration curves, gate tables, run metadata, validation status, an executed notebook, and a report without participant-level records.

Classify the outcome as:

- **Passed:** at least one P2 calibration size passes every primary condition;
- **Signal without passage:** P2 or a secondary model improves a partial metric but fails a primary condition;
- **Negative:** no P2 calibration size has stable incremental value;
- **Invalid:** an audit fails.

Phase 4 remains blocked unless Phase 3 identifies a specific residual error mode that a compact deep model can plausibly address beyond P0, P2, and the failed EEG comparator.
