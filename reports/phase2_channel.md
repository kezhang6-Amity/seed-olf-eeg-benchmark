# Phase 2: Odor-Controlled Channel Features

## Decision

**Classification: signal_without_passage.** Channel-resolved EEG features did not demonstrate reliable incremental probabilistic value beyond odor identity for the public SEED-OLF binary subjective-valence proxy. They must not be described as evidence of relaxation, clinical benefit, or a validated personalized-response model.

## Design

- 2,304 participant-session-trials from 32 participants across three sessions.
- Primary model M2: odor indicators plus training-fold odor-residualized stimulation log band power (62 channels × 5 bands).
- Comparators: M0 odor prior, M1 odor-only logistic regression, M3 differential entropy, M4 stimulation-minus-recovery log power, and M5 robust M2.
- Primary protocol: leave-one-subject-out (LOSO); secondary protocol: within-person leave-one-session-out.
- Every fitted transformation, hyperparameter choice, and odor residualizer was restricted to its training fold.
- 20 negative controls permuted labels within `(subject, odor)` in each outer training fold.

## Data-quality and audit status

- Feature cache: 2,480 features, finite rate 100.0%, zero-variance features 0.
- Differential-entropy floor activation: 0.0% in stimulation and recovery after calculating variance in microvolt squared.
- Independent metric recomputation difference: 8.33e-17.
- Negative-control gate: passed; 0/20 permuted runs passed all statistical gates, and median LOSO M2 improvement was -0.0356.

## Main results

| Protocol | Model | Log loss ↓ | Brier ↓ | Balanced accuracy ↑ | AUROC ↑ |
|---|---|---:|---:|---:|---:|
| LOSO | M0 odor prior | 0.4411 | 0.1385 | 0.8093 | 0.8142 |
| LOSO | M2 primary log-power | 0.4786 | 0.1524 | 0.7765 | 0.8465 |
| LOSO | M5 robust log-power | 0.4630 | 0.1469 | 0.7870 | 0.8548 |
| Leave-session-out | M0 odor prior | 0.4135 | 0.1299 | 0.8247 | 0.8869 |
| Leave-session-out | M5 robust log-power | 0.6412 | 0.2246 | 0.5999 | 0.6608 |

M2's LOSO mean paired log-loss improvement over M0 was **-0.0376** (95% CI **-0.0661 to -0.0096**); positive favors EEG. M5 was less harmful but still negative at **-0.0220** (95% CI **-0.0442 to -0.0004**). All four delete-one-odor checks were negative for M2 and M5.

## Interpretation

The channel models increased LOSO AUROC but worsened log loss and Brier score. This is a ranking-versus-calibration conflict: the models sometimes rank trials better, but their probabilities are less trustworthy than odor-only predictions. The cross-session collapse is stronger evidence against using this representation for immediate few-shot personalization.

The next justified experiment is the preregistered Phase 3 calibration-curve baseline, retaining the odor baseline and failed EEG representations as transparent comparators. EEGNet and TSception do not enter until that experiment identifies an error mode that shallow models leave unresolved.

## Boundaries

- 200 Hz remains an engineering assumption inferred from sample count and stage duration.
- Channels are indexed `ch00`–`ch61`; montage, locations, and reference are not verified.
- `clean_breathing` is post-stimulus recovery, not a true prestimulus baseline.
- `y_emotion` is a public binary subjective-valence proxy, not relaxation, stress, or a clinical endpoint.
