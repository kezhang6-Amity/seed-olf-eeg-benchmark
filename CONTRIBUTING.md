# Contributing

## Before opening a change

Open an issue describing the research question, required data, comparator, grouped validation protocol, primary endpoint, and expected artifact. Do not add a model solely because it is larger or newer.

## Development

```bash
python -m pip install -e '.[test,notebook]'
pytest
python scripts/check_repository.py
```

Keep raw EEG and participant-level generated artifacts outside Git. Never commit credentials, institutional documents, or personal data.

## Experiment requirements

- Assign participant/session/trial groups before fitted preprocessing or window generation.
- Fit priors, transforms, feature selection, and tuning on training folds only.
- Compare EEG against the strongest eligible non-EEG baseline.
- Preserve failed runs and report convergence problems.
- Save configuration, seeds, package versions, trial predictions locally, and compact public summaries.
- Add or update synthetic tests for every change to shared analysis logic.

## Pull requests

State the hypothesis, changed files, validation evidence, result interpretation, and any new caveats. A result is not promoted merely because one ranking metric improves.
