# Research roadmap

Each phase is promoted only after tests, leakage checks, machine-readable outputs, and an answer-first report are complete.

## Phase 1 — Baseline v1: complete

- Safe ingestion and paired manifest.
- Stimulation–recovery spectral effects.
- Cross-session band-power reliability.
- Training-fold odor priors, spectral logistic regression, and Riemannian comparison.
- LOSO and within-participant leave-one-session-out evaluation.

Decision: odor priors remain the primary benchmark; basic EEG does not yet show reliable incremental log-loss value.

## Phase 2 — Odor-controlled channel features: next

- Verify official sampling rate, channel order, reference, and prior preprocessing.
- Add channel-resolved band power and differential entropy.
- Model residual information after training-fold odor effects.
- Report all-trial performance as primary; treat observed-label disagreement subsets as exploratory.

Gate: participant-block 95% CI for log-loss improvement excludes zero, Brier score does not worsen, and direction agrees in a cross-session protocol.

## Phase 3 — Few-shot personalization: next

- Calibrate with 0, 4, 8, 16, or 24 odor-stratified session-1 trials.
- Test only on sessions 2–3.
- Repeat deterministic calibration sampling and report participant heterogeneity.

Output: calibration-size curve that informs future collection burden.

## Phase 4 — Compact deep models: conditional

- EEGNet from the author repository or a paper-checked maintained implementation.
- TSception only after official montage verification.
- At least five fixed seeds and inner-validation early stopping.

Gate: improve a declared remaining error mode over odor and classical EEG baselines. Model size alone is not a reason to proceed.

## Deferred

DSEN is deferred until authoritative code or a complete specification is available. Foundation models are deferred until the target, metadata, and cross-session failure modes are better characterized.
