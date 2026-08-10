# Methodology

## Target and unit of analysis

The target is the participant's binary subjective valence report. The primary unit is a participant-session-trial. Derived windows remain inside their parent trial's fold.

## Comparators

The benchmark starts with a global class prior and an odor-conditional prior estimated only from training labels. EEG models must demonstrate incremental value beyond the strongest eligible non-EEG comparator.

## Validation

- **LOSO:** all sessions of one participant are held out; inner tuning is participant-grouped.
- **Leave-one-session-out:** one full session is held out within each participant; inner groups preserve experimental folds.
- **Few-shot, planned:** target session 1 supplies 4, 8, 16, or 24 calibration trials; sessions 2–3 remain test-only.

Primary comparison uses held-out log loss and participant-block uncertainty. Brier score, balanced accuracy, AUROC, and PR-AUC are secondary. Ranking improvement alone does not establish useful incremental probability estimates.

## Baseline features

Signals receive common-average reference. Welch power spectral density uses two-second Hann segments with one-second overlap. Trial summaries contain mean and standard deviation of log and relative power across channels for delta, theta, alpha, beta, and gamma bands. Paired representations include stimulation minus post-stimulus recovery summaries.

The sampling rate is assumed to be 200 Hz. This is an explicit engineering assumption, not verified embedded metadata.

## Reliability and effects

Stimulation–recovery effects are first averaged within participant and tested with participant-level paired statistics and BH-FDR correction. Cross-session reliability uses absolute-agreement ICC(2,1) after aggregation by participant, session, and odor.

## Claim boundary

Recovery is not a true pre-stimulus baseline. Stage differences cannot isolate odor causality, relaxation, respiration, carry-over, or time. The public dataset cannot validate PPG, HRV, EDA, clinical benefit, or multimodal fusion.
