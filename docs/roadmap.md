# Research roadmap

Each phase is promoted only after tests, leakage checks, machine-readable outputs, and an answer-first report are complete.

## Phase 1 — Baseline v1: complete

- Safe ingestion and paired manifest.
- Stimulation–recovery spectral effects.
- Cross-session band-power reliability.
- Training-fold odor priors, spectral logistic regression, and Riemannian comparison.
- LOSO and within-participant leave-one-session-out evaluation.

Decision: odor priors remain the primary benchmark; basic EEG does not yet show reliable incremental log-loss value.

## Phase 2 — Odor-controlled channel features: complete

- Verify official sampling rate, channel order, reference, and prior preprocessing.
- Add channel-resolved band power and differential entropy.
- Model residual information after training-fold odor effects.
- Report all-trial performance as primary; treat observed-label disagreement subsets as exploratory.

Decision: the primary channel model was reliably worse than the odor prior on log loss. It is retained only as a transparent failed comparator.

## Phase 3 — Few-shot personalization: complete

- Calibrate with 0, 4, 8, 16, or 24 odor-stratified session-1 trials.
- Test only on sessions 2–3.
- Repeat deterministic calibration sampling and report participant heterogeneity.

Decision: a leave-target pooled odor posterior updated with Session-1 labels passes all gates with four trials (one per odor). The incremental gain is small; direct target-only frequencies and the Phase-2 EEG comparator are worse.

## Phase 4 — Compact deep models: conditional

- EEGNet from the author repository or a paper-checked maintained implementation.
- TSception only after official montage verification.
- At least five fixed seeds and inner-validation early stopping.

Gate: improve a declared remaining error mode over the Phase-3 P2 posterior and classical EEG baselines. Model size alone is not a reason to proceed.

## Deferred

DSEN is deferred until authoritative code or a complete specification is available. Foundation models are deferred until the target, metadata, and cross-session failure modes are better characterized.
