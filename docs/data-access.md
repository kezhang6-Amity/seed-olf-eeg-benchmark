# Data access and governance

SEED-OLF is not redistributed by this repository. Obtain the data from the authors' official channel and follow their current terms. The original paper is [Zhang et al., 2026](https://doi.org/10.1109/TAFFC.2026.3662364).

Set an absolute local path:

```bash
export SEED_OLF_DATA_ROOT=/absolute/path/to/SEED-OLF
```

The directory must contain `stimulation/` and `clean_breathing/`, with filenames such as `01_1_1.pkl`. Raw files remain outside Git.

The loader accepts only the audited NumPy pickle constructors and requires keys `X_raw`, `y_emotion`, `y_odor`, and `y_sniff`, a float64 signal shape of 62 × 3,000, finite values, valid filenames, and complete phase pairing. Schema changes fail closed.

Generated manifests, feature tables, full predictions, and covariance caches go to `artifacts/`, which is ignored. Only aggregate tables needed to audit public claims are copied into `results/` after validation.
