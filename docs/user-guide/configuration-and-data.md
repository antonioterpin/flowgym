# Configuration and data flow

FlowGym uses configuration files to connect estimators, datasets, and
repository-level workflows.

This is one of the main ways the project keeps evaluation and training runs
repeatable.

## The two main kinds of config

- model configs:
  choose the estimator family, estimate type, and model-specific settings
- dataset configs:
  describe the data source or sampler setup used for evaluation or training

## How they come together

In repository-backed workflows, the orchestration layer reads these configs,
creates the corresponding estimator and dataset objects, and dispatches into
evaluation or training.

The key entrypoints are:

- `src/main.py`
- `src/eval.py`
- `src/train.py`
- `src/train_supervised.py`

## Where configs live

Most repository configs are under `src/flowgym/config/`, with estimator
defaults grouped under `src/flowgym/config/estimators/`.

When you are using the package directly, you may still construct the same
kind of configuration objects in Python rather than loading them from files.

## Related pages

- [Training and evaluation workflows](training-and-evaluation.md)
- [API reference](../api/index.md)
- [Contributing guide](../guides/contributing.md)
