# SEED-OLF EEG Benchmark Repository Design

**Date:** 2026-08-10  
**Repository:** `seed-olf-eeg-benchmark`  
**Status:** Approved for implementation

## Purpose

Create a public, reproducible research repository for leakage-safe SEED-OLF EEG baselines and follow-up work on individual response, cross-session reliability, and few-shot personalization.

The repository is an independent student-led research project. It is not an official University of Illinois Urbana-Champaign project and is not affiliated with the SEED-OLF authors.

## Public boundary

Include source code, synthetic tests, configuration, notebooks, compact result summaries, methodology, and a roadmap. Exclude raw SEED-OLF files, full trial-level predictions, internal grant or pitch materials, resumes, credentials, caches, and machine-specific environments.

MIT covers repository code only. It does not cover SEED-OLF data or third-party model code. MindScents may be named only as the internal research-project context. LemoriX will not appear while its public brand status remains unresolved.

## Architecture

- `src/seed_olf_benchmark/`: safe data access, features, validation, models, and metrics.
- `scripts/`: stable command-line experiment entry points.
- `configs/`: versioned assumptions and experiment parameters.
- `notebooks/`: data audit and executed baseline narrative.
- `results/baseline_v1/`: compact, non-sensitive summary tables and validation metadata.
- `reports/`: answer-first research reports.
- `docs/`: data access, methods, roadmap, model sources, and decision records.
- `tests/`: synthetic-data unit and integration checks.
- `.github/`: CI, issue forms, pull-request template, and dependency updates.

## Reproducible data flow

1. The user downloads SEED-OLF through its official access route.
2. `SEED_OLF_DATA_ROOT` points to a directory containing `stimulation/` and `clean_breathing/`.
3. Safe loading validates pickle globals, schema, trial keys, shapes, and phase pairing.
4. Experiment configuration and deterministic grouped splits are recorded with outputs.
5. All fitted priors, preprocessing, feature selection, and tuning operate on training folds only.
6. Scripts write generated artifacts to a configurable output directory.
7. Only compact summaries required to audit reported conclusions are versioned.

The inferred 200 Hz sampling rate and post-stimulus recovery interpretation remain explicit caveats until authoritative metadata are available.

## Testing and CI

CI runs on supported Python versions without the private dataset. It installs the package, compiles source files, runs synthetic unit tests, checks tracked-result schemas, and verifies that forbidden raw-data patterns are not committed. Dataset-dependent full runs remain local and produce machine-readable validation output.

## Sustainable progression

The roadmap has gated phases: baseline v1, odor-controlled channel-resolved features, few-shot cross-session personalization, compact deep models, and future external validation. Each phase must define its hypothesis, non-EEG comparator, leakage-safe protocol, primary probabilistic endpoint, and promotion criterion.

Research decisions are recorded as short ADR-style files. New experiments use configuration files and stable scripts instead of notebook-only logic. Releases correspond to validated result sets; issues and pull requests must state data requirements, expected outputs, and validation evidence.

## Error handling

Unexpected pickle globals or signal schemas fail closed. Missing phase pairs are reported and excluded only from paired analyses. Metadata uncertainty blocks physiological claims but not labeled engineering smoke tests. Failed convergence or incomplete model runs remain visible in logs and may not be silently dropped.

## Success criteria

- A new contributor understands scope and caveats from the README.
- Installation and synthetic tests pass from a clean environment.
- No raw data or internal material is tracked.
- Baseline headline metrics can be traced to versioned summaries.
- The next two experiments are actionable from the roadmap and issue templates.
- The repository is ready to publish under the authenticated GitHub account as a public MIT project.
