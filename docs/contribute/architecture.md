# Architecture guide

This guide is for readers who want to understand how FlowGym is organized
 before making changes. It favors the current mental model of the repo over
 historical detail.

## The mental model

FlowGym has three layers that matter most for navigation:

- Package layer:
  `src/flowgym/` contains estimators, environments, shared utilities,
  configuration loaders, and training helpers.
- Orchestration layer:
  `src/main.py`, `src/train.py`, `src/train_supervised.py`, `src/eval.py`,
  and `src/compare.py` wire configs, samplers, estimators, and output
  handling together.
- Verification layer:
  `tests/` mirrors the package shape and covers both unit and integration
  workflows.

If you are unsure where a change belongs, start by identifying whether it is
about package behavior, orchestration, or verification.

## Repository layout

```text
.
├── src/
│   ├── flowgym/          # Package code
│   ├── main.py           # Top-level CLI orchestration
│   ├── train.py          # Reinforcement-learning training loop
│   ├── train_supervised.py
│   ├── eval.py
│   └── compare.py
├── tests/                # Unit and integration tests
├── examples/             # Repository-backed walkthrough scripts
├── experiments/          # One-off or method-specific experiments
├── docs/                 # Human-facing docs and contributor docs
└── pyproject.toml        # Dependencies, test config, lint config
```

## Package map

### Estimators and shared interfaces

- `flowgym.common.base`:
  base estimator interfaces and trainable-state types.
- `flowgym.flow`:
  flow-field estimators and algorithm-specific implementations.
- `flowgym.density`:
  density estimators.
- `flowgym.make`:
  estimator construction, compilation, checkpoint loading, and checkpoint
  saving.

This is the most important package area for public API navigation.

### Environment and data flow

- `flowgym.environment.fluid_env`:
  training-oriented wrapper around `synthpix` samplers.
- `flowgym.common.preprocess`, `flowgym.common.filters`,
  `flowgym.flow.postprocess`:
  transformation helpers around raw images and estimates.
- `flowgym.common.evaluation`:
  metric computation and evaluation helpers.

### Training and optimization

- `flowgym.training.optimizer`:
  optimizer construction from config.
- `flowgym.training.schedules`:
  learning-rate schedules.
- `flowgym.training.caching`:
  cache-backed reuse of expensive derived quantities.
- `flowgym.training.replay`, `flowgym.training.losses`,
  `flowgym.training.target_transforms`:
  training support modules used by the orchestration scripts.

### Configuration

- `src/flowgym/config/`:
  YAML configs for datasets, estimators, and experiments.
- `src/flowgym/config/estimators/`:
  estimator-specific defaults.
- top-level dataset configs like `piv_dataset_class1_eval.yaml`:
  ready-made CLI inputs for `src/main.py`.

## Entry points that matter

- `src/main.py`:
  parse CLI arguments, load dataset/estimator configs, create samplers, and
  dispatch into eval or training modes.
- `src/train.py`:
  reinforcement-learning training loop.
- `src/train_supervised.py`:
  supervised training loop.
- `src/eval.py`:
  evaluation helpers and full-dataset evaluation flow.
- `src/compare.py`:
  sampler comparison workflow.

If you are documenting or debugging a user-facing workflow, start with
`src/main.py`.

## Common navigation shortcuts

- "I need the public estimator API":
  start with `flowgym.make` and `flowgym.common.base`.
- "I need to understand an estimator implementation":
  start under `flowgym.flow/` or `flowgym.density/`.
- "I need dataset or experiment defaults":
  start in `src/flowgym/config/`.
- "I need the closest tests":
  look for the matching module name under `tests/`.
- "I need a real workflow example":
  check `examples/` and the docs pages under `docs/examples/`.

## Testing shape

Tests live in `tests/` and are intended to largely follow the package
structure, with a mix of unit and integration coverage. That is the shape
the project is aiming for, but it is not fully realized yet; see
[issue #22](https://github.com/antonioterpin/flowgym/issues/22) for the
current cleanup and restructuring work.

- top-level tests:
  cross-cutting behavior such as preprocessing, filtering, or configuration
  handling
- focused subdirectories:
  `tests/base_estimator/`, `tests/training/`, `tests/caching/`,
  `tests/consensus/`, and `tests/nn/`
- `tests/conftest.py`:
  shared fixtures

When adding code, the nearest existing test file is usually the right place
to start.

## Related docs

- [Getting started](../getting-started/index.md)
- [Example workflows](../examples/index.md)
- [Contributor's guide](contributing.md)
