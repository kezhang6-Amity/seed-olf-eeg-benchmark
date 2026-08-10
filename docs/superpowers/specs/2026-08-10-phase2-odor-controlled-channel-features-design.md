# Phase 2 Odor-Controlled Channel Features Design

**Date:** 2026-08-10

**Status:** Approved design; implementation pending written-spec review

## 1. Purpose

Test whether channel-resolved EEG contains held-out information about the public SEED-OLF binary subjective-valence label beyond odor identity.

`y_emotion` is a methodological proxy target. It must not be described as relaxation, stress reduction, clinical benefit, or an objective emotional state. The future MindScents continuous relaxation target is outside this public-data claim.

## 2. Scope and sequence

Phase 2 covers the metadata gate, channel-resolved features, fold-local odor control, grouped predictive validation, negative controls, and an executed report notebook.

Few-shot personalization is Phase 3 and begins only after the Phase 2 representation and model interface are frozen. EEGNet and TSception remain conditional Phase 4 models. DSEN remains deferred until authoritative code or a sufficiently complete specification is available.

## 3. Metadata gate

The [original paper](https://doi.org/10.1109/TAFFC.2026.3662364) and [author institution's public overview](https://news.sjtu.edu.cn/jdzh/20260301/219968.html) establish the dataset and experimental paradigm. The files available in this workspace do not embed sampling rate, channel order, montage, reference, line frequency, or prior-filter metadata.

Until an author-level source resolves those fields:

- sampling rate remains an explicit 200 Hz engineering assumption inferred from 3,000 samples over the reported 15-second stage;
- channels are named `ch00` through `ch61` according to array index only;
- no electrode, region, laterality, or cortical-mechanism claim is permitted;
- channel features may support predictive engineering validation but not scalp-topography interpretation;
- `clean_breathing` is treated as post-stimulus recovery, not a pre-stimulus baseline.

Metadata uncertainty does not block the declared predictive smoke test, but it blocks physiological promotion of channel-level results.

## 4. Analytical unit and population

- Population: all 32 public participants.
- Sessions: three per participant.
- Trials: 24 per session and 2,304 stimulation/recovery pairs overall.
- Primary unit: participant-session-trial.
- Trial key: `(subject, session, trial)`.
- All derived features from one trial remain within its assigned fold.
- Primary analysis includes every schema-valid trial; no trial is deleted based on the outcome or the full-dataset feature distribution.

## 5. Channel feature definitions

The primary preprocessing variant applies common-average reference to each phase independently. It does not add an unverified notch or claim a verified acquisition reference.

For each phase, channel, and declared frequency band, compute:

1. Welch log band power using the existing two-second Hann segments and one-second overlap;
2. relative band power for quality and sensitivity checks;
3. differential entropy under the declared Gaussian band-limited formulation.

For differential entropy, band-limit each channel with a fourth-order zero-phase Butterworth filter under the assumed 200 Hz sampling rate, trim one second from each edge after filtering, and calculate the sample variance of the remaining samples. Report

`DE = 0.5 × ln(2πe × max(sample_variance, 1e-12))`

in nats. The same edge trimming and estimator are used in every fold and phase.

The declared bands remain delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and gamma 30–45 Hz under the assumed sampling rate.

Primary feature set:

- stimulation channel log band power: 62 channels × 5 bands = 310 features.

Secondary feature sets:

- stimulation channel differential entropy;
- stimulation minus recovery channel log band power;
- stimulation minus recovery channel differential entropy;
- robustly transformed versions fit only inside training folds.

Differential entropy and log power are not concatenated in the primary model. Their fold-wise correlation and numerical equivalence are quantified first because band-limited Gaussian differential entropy is closely related to log variance/power. Any apparent benefit from concatenation would require a separate declared experiment.

## 6. Data-quality profile

The channel feature build records:

- input coverage and unique trial keys;
- phase-pair coverage;
- signal shape, dtype, finite values, and flat channels;
- per-channel peak amplitude and standard deviation;
- feature finite values, zero variance, quantiles, and extreme-value rates;
- distributions by subject, session, odor, phase, and label;
- feature count and expected schema;
- cache configuration and source-file fingerprint metadata.

The source fingerprint is a combined SHA-256 digest over the sorted tuples `(relative path, byte size, file SHA-256)` for every consumed raw file. Feature-cache reuse requires an exact source fingerprint, feature configuration, and schema-version match.

The primary analysis performs no artifact-based trial exclusion. A sensitivity model may use training-fold quantile clipping and robust scaling. Test observations never determine clipping limits, centers, scales, exclusions, or feature availability.

## 7. Fold-local odor control

For each outer fold, fit an odor residualizer using only outer-training observations:

1. estimate the mean of every EEG feature within each odor in training data;
2. subtract the matching training-derived odor mean from training and test observations;
3. use the training global feature mean for an odor absent from training;
4. fit scaling and the classifier after residualization;
5. repeat all fitted steps independently inside every inner-validation split.

This estimand asks whether EEG deviations within odor categories improve subjective-valence prediction. It does not claim that odor is causally removed, because odor identity can interact with participant, session, respiration, and measurement noise.

## 8. Model ladder

- **M0:** odor-specific positive-label frequency with add-one Laplace smoothing, estimated inside each training fold; an unseen odor uses the add-one-smoothed global training prevalence.
- **M1:** one-hot odor-only logistic regression.
- **M2:** odor plus odor-residualized stimulation channel log band power. This is the single primary EEG model.
- **M3:** odor plus odor-residualized stimulation differential entropy.
- **M4:** odor plus odor-residualized stimulation/recovery paired features.
- **M5 sensitivity:** robustly transformed version of M2 using training-fold clipping and robust scaling.

Ridge logistic regression is the primary classifier. M1 uses the odor indicators directly. After odor residualization, M2–M4 use a training-fitted standard scaler followed by L2 logistic regression. All logistic models search `C` in `[0.001, 0.01, 0.1, 1, 10]` with the `lbfgs` solver, `max_iter=5000`, no class weighting, and fixed seed `20260810`. M5 clips each feature to its outer-training 1st and 99th percentiles, applies a training-fitted robust scaler, and otherwise uses the same classifier grid. No test-fold model or feature selection is permitted.

An observed-label odor-disagreement subset may be reported only as exploratory because subgroup membership depends on the outcome.

## 9. Validation protocols

### 9.1 Primary: leave one subject out

- Outer test fold: all three sessions of one participant.
- Outer training fold: the other 31 participants.
- Inner tuning: deterministic five-fold `GroupKFold` by participant.
- Primary comparison: M0 versus M2 on all held-out trials.

### 9.2 Secondary: within-participant leave one session out

- Outer test fold: one complete session.
- Outer training fold: the participant's other two sessions.
- Inner tuning: two folds, each using one complete training session for fitting and the other for validation.
- Purpose: cross-day direction and stability, not the primary efficacy claim.

### 9.3 No permitted random full-dataset split

Participant, session, trial, phase, and derived-window grouping constraints are invariant and tested automatically.

## 10. Endpoints and uncertainty

Primary endpoint: held-out log-loss improvement, defined so positive values favor M2 over M0.

Estimate per-trial paired loss differences and calculate a paired hierarchical percentile-bootstrap interval by resampling 32 participant clusters with replacement and then resampling trials with replacement within each sampled cluster. Use the mean of the resampled trial differences as the replicate statistic, 10,000 replicates, seed `20260810`, and the 2.5th and 97.5th percentiles for the interval.

Guardrails and secondary endpoints:

- Brier score and paired uncertainty;
- calibration intercept and slope;
- ten-bin equal-frequency expected calibration error;
- balanced accuracy;
- AUROC and PR-AUC;
- participant improvement rate;
- performance by odor and session.

The main conclusion is not promoted from an AUROC increase alone.

## 11. Primary success gate

Phase 2 demonstrates reliable incremental EEG value only if every condition holds:

1. the 95% confidence interval lower bound for LOSO log-loss improvement is greater than zero;
2. the M2 Brier point estimate does not worsen relative to M0;
3. the 95% confidence interval lower bound for Brier improvement is not below −0.005;
4. mean leave-one-session-out log-loss improvement is positive;
5. more than half of participants have positive LOSO log-loss improvement;
6. delete-one-odor influence checks remain non-negative for all four exclusions;
7. leakage, coverage, schema, cache, and independent metric audits pass.

Failure of any gate prevents a positive incremental-value claim. A partial metric improvement is reported as signal without gate passage.

The −0.005 Brier threshold is a preregistered pragmatic engineering non-inferiority margin, not a clinical threshold. Report sensitivity results at margins of 0 and −0.0025 so that the conclusion is not hidden behind the selected tolerance.

The delete-one-odor check does not retrain models. For each odor in turn, remove that odor's already held-out LOSO predictions and recompute the M2-versus-M0 mean log-loss improvement. This diagnoses whether one odor alone determines the pooled conclusion.

## 12. Secondary multiplicity

M3, M4, M5, secondary protocols, channel-wise effects, and subgroup results are not co-primary. Related secondary comparisons use Benjamini–Hochberg false-discovery-rate control within their declared family. The single predeclared M2 primary comparison is not selected from secondary results.

## 13. Negative controls

Within each outer-training fold, permute `y_emotion` within `(subject, odor)` strata in the outer-training data. This preserves subject-odor label prevalence and leaves the held-out observations and M0 predictions unchanged. Refit the complete residualization, scaling, tuning, and classifier pipeline for 20 fixed permutations.

No permuted run may pass the primary success gate, and the median permuted log-loss improvement must be less than or equal to zero. Any violation triggers a leakage/cache investigation and blocks interpretation of the real-label model.

Additional software negative controls use synthetic signals and deliberately shifted test distributions.

## 14. Automated tests

Required unit and integration checks include:

- a channel-specific sinusoid peaks in the expected band;
- differential entropy matches a known Gaussian-variance calculation;
- channel order and feature naming are deterministic;
- feature shape is exactly the declared channel-band schema;
- an extreme test value cannot change training residual means, clipping bounds, or scaling parameters;
- unseen odors use the training global feature mean;
- stimulation and recovery from the same trial cannot cross folds;
- LOSO has no participant overlap;
- leave-one-session-out has no session overlap;
- complete predictions contain every expected trial exactly once per protocol and model;
- an independent validator recomputes all headline metrics from trial predictions;
- cached feature metadata matches configuration and source fingerprints;
- the reader-facing notebook executes top to bottom without error.

## 15. Outputs

Local ignored artifacts:

- full channel feature cache;
- data-quality profile;
- deterministic fold assignments;
- full trial predictions;
- fitted hyperparameter and negative-control logs.

Versioned public artifacts:

- compact prediction summary;
- primary incremental-value table;
- calibration and robustness summaries;
- channel-feature reliability summary without participant-level records;
- run metadata and independent validation status;
- executed notebook and answer-first report.

## 16. Result classification

- **Passed:** every primary success gate passes.
- **Signal without passage:** a secondary/ranking/subgroup result improves but at least one primary gate fails.
- **Negative:** no stable incremental value is observed under the primary design.
- **Invalid run:** leakage, schema, coverage, metric, cache, or execution validation fails.

Negative or partial results remain publishable methodological evidence and directly inform future acquisition design.

## 17. Phase 3 handoff

Phase 3 may begin after Phase 2 produces a validated feature interface. The representation used for few-shot personalization is selected by the Phase 2 primary gate and frozen before testing calibration sizes 0, 4, 8, 16, and 24. If M2 fails, Phase 3 still runs with the strongest validated non-EEG and EEG baselines, but it cannot describe the failed Phase 2 representation as established.
