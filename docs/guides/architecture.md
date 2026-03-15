# Repository guide: architecture and navigation

This guide provides an overview of the repository's architecture, helping contributors and agents navigate the codebase and understand the responsibilities of each component.

## 1. Overview

The **Flow Gym** repository is a JAX-based framework for estimating physical quantities in fluid dynamics, primarily focusing on **flow fields (PIV)** and **seeding densities**. It provides a unified interface for various estimation algorithms (traditional and neural-network based), supports synthetic data generation via `synthpix`, and offers both Reinforcement Learning (RL) and supervised training pipelines.

---

## 2. Repository layout

```text
.
├── src/                                                # Source code
│   ├── flowgym/                                        # Core package: estimators, envs, and logic
│   ├── [compare.py](@src/compare.py)                   # Data comparison entry point
│   ├── [eval.py](@src/eval.py)                         # Evaluation orchestration
│   ├── [main.py](@src/main.py)                         # Unified CLI entry point
│   ├── [train.py](@src/train.py)                       # RL training orchestration
│   └── [train_supervised.py](@src/train_supervised.py) # Supervised training orchestration
├── [tests/](@tests/)                                   # Test suite (mirrors src/flowgym structure)
│   ├── [conftest.py](@tests/conftest.py)               # Shared pytest fixtures
│   └── ...                                             # Unit and integration tests
├── [docs/](@docs/)                                     # Supplementary architectural documentation
├── [examples/](@examples/)                             # Usage examples and integration demos
├── [experiments/](@experiments/)                       # Experiment-specific scripts and artifacts
├── [pyproject.toml](@pyproject.toml)                   # Build and dependency configuration
└── [.agent/rules/rules.md](@.agent/rules/rules.md)     # Development and coding standards
```

---

## 3. Core package: `flowgym`

The `flowgym` package is organized into several functional subpackages:

- **`common`**: Owns base classes and shared utilities.
  - `common.base`: Contains `Estimator` (abstract base for all models) and `TrainableState` (JAX PyTree for model parameters and optimizer state).

  > [!TIP]
  > **TrainableState Architecture**: Two classes exist for different use cases:
  > - `EstimatorTrainableState`: Concrete "empty" state for non-trainable/eval-only estimators. Manually registered as PyTree.
  > - `NNEstimatorTrainableState(TrainState, EstimatorTrainableState)`: For trainable models. Subclasses Flax's `TrainState` and adds `extras` via `struct.field()`. **No custom `__init__` is needed**—Flax's dataclass machinery handles all fields automatically through `.create()`.

  - `common.preprocess` / `common.filters`: Image processing and signal filtering utilities.
  - `common.evaluation`: Metrics calculation (e.g., EPE, density loss).
- **`config`**: Owns all YAML-based configurations for models, datasets, and experiments.
- **`density`**: Owns estimators specifically for seeding density (e.g., `nn.py`, `simple.py`).
- **`environment`**: Owns the `FluidEnv`, a Gym-like interface that wraps `synthpix` samplers for reinforcement learning.
- **`flow`**: Owns flow field (PIV) estimators. Subpackages like `dis`, `raft`, and `open_piv` contain algorithm-specific implementations.
- **`nn`**: Owns neural network architectures.
  > [!NOTE]
  > The repository prioritizes the **JAX ecosystem** (via **Flax**, **Orbax**, **Optax**, etc.) for all new developments. While some legacy or comparative models exist in Torch (`raft_torch_nn/`), future extensions should leverage JAX-compatible libraries.
- **`training`**: Owns optimization logic, learning rate schedules, and training-specific utilities.
  - `training.caching`: Contains the `CacheManager`, a read-through on-disk cache designed to avoid recomputation of expensive derived data (like EPEs or model estimates) for `SynthpixBatch` items. It uses Parquet files for storage and supports multiple warm-start strategies (none, index-only, or full data in RAM).

---

## 4. Entry points & scripts

The repository uses a functional separation for its main scripts in `src/`:

- **`main.py`**: The primary user-facing CLI. Use it to run training, evaluation, or comparison by selecting a `--mode`. It acts as the high-level orchestrator.
- **`train.py`**: Implements the Reinforcement Learning training loop. It is called by `main.py` when `--mode train` is used.
- **`train_supervised.py`**: Implements supervised training loops for models with ground-truth data. Called via `main.py --mode train-supervised`. It supports integration with `CacheManager` to speed up training when specific derived data is required.
- **`eval.py`**: Contains the logic for evaluating models on batches of data and calculating performance statistics. It integrates with `CacheManager` to enable fast evaluation by loading pre-computed results.
- **`compare.py`**: A specialized script for comparing estimator performance across different data distributions (e.g., synthetic vs. real PIV images).

### Caching integration
The caching system is integrated into the `main -> [eval, train_supervised]` flow:
1. `main.py` parses the `caching` configuration from the dataset YAML and initializes the `CacheManager`.
2. The `CacheManager` is passed to `eval_full_dataset` (in `eval.py`) or `train_supervised` (in `train_supervised.py`).
3. During evaluation/training iterations, the `CacheManager.enrich` method is used to look up results by batch keys. If a miss occurs, the model's `compute_cache_miss` hook is called, and the results are written back to the cache.

---

## 5. Testing structure

Tests are located in the `tests/` directory and largely mirror the structure of `src/flowgym/`.

- **Unit Tests**: Test individual functions and classes (e.g., `test_filters.py`, `test_median.py`).
- **Integration Tests**: Test the full estimation or training flow (e.g., `tests/training/test_integrated_checkpointing.py`).
- **`conftest.py`**: Defines shared fixtures, such as mock data samplers or estimator instances, ensuring consistent testing environments.
- **Organization**: Subdirectories like `tests/base_estimator/` or `tests/consensus/` isolate tests for specific architecture components.

---

## 6. Common change patterns

### Adding a new model
1. Create a new subclass of `Estimator` (found in `src/flowgym/common/base/estimator.py`) in `src/flowgym/flow/` or `src/flowgym/density/`. If the model is a flow field estimator, it should inherit from `FlowFieldEstimator` (found in `src/flowgym/flow/base.py`).
2. Implement `_estimate` and (optionally) `create_train_step`.
3. Register the new model in `src/flowgym/__init__.py`.
4. Add a corresponding YAML configuration in `src/flowgym/config/estimators/`.

### Modifying training behavior

> [!IMPORTANT]
> Agents should avoid modifying core training files unless fixing bugs. Any changes should be general enough to support all estimators without breaking existing functionality.

- **Optimizer/Schedules**: Change `src/flowgym/training/optimizer.py` or `src/flowgym/training/schedules.py`.
- **Training Loop**: Modify `src/train.py` (for RL) or `src/train_supervised.py` (for supervised).

### Adding evaluation metrics
- Add the metric calculation logic to `src/flowgym/common/evaluation.py`.
- Update the relevant `eval_*` function in `src/eval.py` to include the new metric.

### Changing configuration
- All default settings reside in `src/flowgym/config/`. When running experiments, pass a customized YAML via the `--model` or `--dataset` flags in `main.py`.

---
