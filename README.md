# SEED-OLF EEG Benchmark

[![CI](https://github.com/kezhang6-Amity/seed-olf-eeg-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/kezhang6-Amity/seed-olf-eeg-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An independent, leakage-safe research benchmark for asking whether EEG explains subjective olfactory response beyond odor identity.

This repository contains public-data analysis code, synthetic tests, compact result summaries, and an executable research roadmap. It does **not** redistribute SEED-OLF recordings.

## Current result

The strongest unseen-participant baseline is the training-fold odor prior. Channel-aggregated spectral EEG improves ranking but does not provide reliable incremental probabilistic value in baseline v1.

| LOSO model | Log loss ↓ | Brier ↓ | Balanced accuracy ↑ | AUROC ↑ |
|---|---:|---:|---:|---:|
| Odor prior | **0.4411** | **0.1385** | **0.8093** | 0.8142 |
| Odor + spectral EEG | 0.4464 | 0.1409 | 0.7840 | **0.8579** |
| EEG only | 0.6818 | 0.2443 | 0.5268 | 0.5395 |

The participant-block bootstrap estimate for odor + spectral EEG versus odor prior is **-0.0054 log-loss improvement, 95% CI [-0.0165, 0.0079]**. The interval includes zero. See the [baseline report](reports/baseline_v1.md) for the full interpretation.

Phase 2 tested odor-controlled 62-channel features. Its primary log-power model also failed to add reliable probabilistic value: LOSO mean log-loss improvement was **-0.0376 (95% CI [-0.0661, -0.0096])** relative to the odor prior, despite an AUROC increase. The complete [Phase 2 report](reports/phase2_channel.md) documents the 20 permutation controls, data-quality audit, and interpretation boundary.

## Research questions

1. Does EEG improve held-out subjective-valence prediction beyond train-derived odor priors?
2. Which olfactory EEG responses remain reliable across sessions?
3. How many target-person calibration trials are needed for cross-session personalization?

## Install

```bash
git clone https://github.com/kezhang6-Amity/seed-olf-eeg-benchmark.git
cd seed-olf-eeg-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test,notebook]'
pytest
```

Python 3.11 and 3.12 are supported.

## Data setup

Obtain SEED-OLF through the dataset authors' official access route, accept its terms, and keep the files outside this repository. The expected layout is:

```text
/path/to/SEED-OLF/
├── stimulation/
│   └── 01_1_1.pkl
└── clean_breathing/
    └── 01_1_1.pkl
```

```bash
export SEED_OLF_DATA_ROOT=/path/to/SEED-OLF
seed-olf-baselines --output-dir artifacts/baseline_v1
seed-olf-riemannian --output-dir artifacts/baseline_v1
python scripts/validate_outputs.py --output-dir artifacts/baseline_v1

seed-olf-phase2 --data-root "$SEED_OLF_DATA_ROOT" --output-dir artifacts/phase2_channel
python scripts/validate_phase2_outputs.py --output-dir artifacts/phase2_channel
python scripts/build_phase2_notebook.py --output-dir artifacts/phase2_channel
python scripts/publish_phase2_summary.py --output-dir artifacts/phase2_channel
```

Raw files, participant-level predictions, feature caches, and manifests are ignored by Git. See [data access and governance](docs/data-access.md).

## Repository map

- `src/seed_olf_benchmark/`: safe loading, features, metrics, grouped validation, and models.
- `scripts/`: validation, notebook, and repository checks.
- `configs/`: declared experiment assumptions and model settings.
- `notebooks/`: audit and baseline-effect narratives.
- `results/`: compact, non-participant-level tables supporting reported results.
- `reports/`: reader-facing findings and limitations, including the Phase 2 negative result.
- `docs/`: methods, model provenance, decisions, and roadmap.
- `tests/`: dataset-free synthetic checks used by CI.

## Interpretation boundaries

- The 200 Hz sampling rate is inferred from 3,000 samples over a reported 15-second stage; it is not embedded in the files.
- `clean_breathing` is post-stimulus recovery, not a pre-stimulus baseline.
- Stimulation–recovery differences may include respiration, carry-over, task-stage, and time effects.
- `y_emotion` is a subjective binary report, not a validated relaxation or clinical outcome.
- Disagreement-subset analyses are exploratory because subgroup membership depends on the observed outcome.

## Status and roadmap

Baseline v1 and the channel-resolved odor-controlled Phase 2 benchmark are validated. The next gate is few-shot cross-session personalization using transparent odor and failed-EEG comparators. EEGNet and TSception enter only if that evaluation identifies a remaining error mode under grouped validation. See the [roadmap](docs/roadmap.md).

## Independence and affiliation

This repository is owned and maintained by Ke Zhang in a personal research capacity. It was not created for or on behalf of any company, employer, or commercial entity, and no such entity sponsors, controls, or endorses it.

The research is independent student-led work developed by students at the University of Illinois Urbana-Champaign. It is not an official unit of, sponsored by, or endorsed by the university. It is not affiliated with the SEED-OLF authors. University names and marks are used only to describe contributor context.

MindScents is used only as the name of an internal student research and grant project. It does not identify a company or the owner of this repository. No product brand is used here.

## License and citation

Repository code is released under the [MIT License](LICENSE). The license does not apply to SEED-OLF data or third-party implementations. Cite both this software via [CITATION.cff](CITATION.cff) and the [original SEED-OLF paper](https://doi.org/10.1109/TAFFC.2026.3662364).

## 中文摘要

本仓库是一个独立的 SEED-OLF 公开数据研究基准，重点验证 EEG 是否在气味身份之外解释个体主观反应。仓库只发布代码、合成测试、结果摘要和研究路线，不发布原始 EEG。当前基础结果显示：气味先验仍是最强概率基线，基础频谱 EEG 尚未带来可靠增量。下一阶段优先研究控制气味后的通道级特征与跨日少样本个性化，而不是直接扩大深度模型规模。
