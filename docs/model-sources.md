# Model and Code Sources

Only primary papers, author-maintained repositories, or established EEG libraries are eligible for the benchmark.

| Method | Approved implementation source | Role in this project |
|---|---|---|
| Welch spectral features | SciPy `signal.welch` | First-stage interpretable features |
| Regularized logistic regression | scikit-learn | Probabilistic baseline and incremental-value test |
| Riemannian covariance/tangent space | [pyRiemann](https://pyriemann.readthedocs.io/) | Strong classical EEG baseline after spectral smoke tests |
| EEGNet | [Author ARL repository](https://github.com/vlawhern/arl-eegmodels) or maintained Braindecode implementation checked against the paper | Compact deep baseline |
| TSception | [Author repository](https://github.com/yi-ding-cs/TSception) or maintained Braindecode implementation checked against the author model | Emotion-specific deep baseline |
| DSEN for SEED-OLF | IEEE paper DOI `10.1109/TAFFC.2026.3662364` | Deferred until authoritative code or a complete specification is available |

## Selection rules

- No unrelated repository with a matching model acronym may be used.
- Train/test grouping and preprocessing from third-party examples are not copied blindly; this project uses participant-, session-, and trial-safe splits.
- A larger or newer model is not preferred unless it improves held-out probabilistic performance over odor/history and classical EEG baselines.
- Deep models must use multiple fixed seeds and inner-validation early stopping.
- Any reimplementation must document deviations from the source paper and pass a shape/parameter smoke test.

## Primary references

- Zhang et al., *SEED-OLF: A Novel EEG Dataset With Olfactory Stimulation for Emotion Recognition*, IEEE Transactions on Affective Computing, 2026.
- Barachant et al., *Multiclass Brain-Computer Interface Classification by Riemannian Geometry*, IEEE TBME, 2012.
- Lawhern et al., *EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces*, Journal of Neural Engineering, 2018.
- Ding et al., *TSception: Capturing Temporal Dynamics and Spatial Asymmetry From EEG for Emotion Recognition*, IEEE Transactions on Affective Computing, 2023.
- Chevallier et al., *The Largest EEG-Based BCI Reproducibility Study for Open Science: the MOABB Benchmark*, Journal of Neural Engineering, 2024.
