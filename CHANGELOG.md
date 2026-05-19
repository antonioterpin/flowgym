# Changelog

This file tracks user-visible changes that downstream consumers or older
notebooks may need to be aware of.

## Unreleased

### Renamed: `model` → `estimator`

The estimator-facing public API has been renamed from "model" to
"estimator" across docs, CLI, configs, and examples.

Downstream consumers updating from prior versions should rename:

- `model_config` → `estimator_config`
- `model_path` → `estimator_path`
- `model` (local variable / print label) → `estimator`
- YAML filename `model.yaml` → `estimator.yaml`
- CLI flag `--model` → `--estimator`

The Python package is still imported as `flowgym`; only the user-facing
"model" vocabulary changed.
