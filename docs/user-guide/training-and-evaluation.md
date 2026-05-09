# Training and evaluation workflows

This page is about the two workflows most users come to Flow Gym for:

- **Benchmarking estimators on a dataset.**
- **Training learning-based estimators.**

Both run through the repository CLI, `src/main.py`, driven by an
estimator YAML and a dataset YAML.

If instead you want to embed an estimator inside your own Python code
(no CLI, no YAMLs), see the
[Estimator API overview](estimator-api.md) and the
[first-estimate example](../examples/first-estimate.md).

## Repository workflows

`src/main.py` is the user-facing entrypoint and dispatches into one of
five modes via `--mode`:

| `--mode`           | Purpose                                                   | Backing script              |
|--------------------|-----------------------------------------------------------|-----------------------------|
| `eval`             | Evaluate an estimator on a dataset                        | `src/eval.py`               |
| `train`            | Reinforcement-learning training loop                      | `src/train.py`              |
| `train-supervised` | Supervised training loop                                  | `src/train_supervised.py`   |
| `compare-samplers` | Run two samplers side by side for comparison              | (in `src/main.py`)          |
| `main`             | Smoke-test loop: build the estimator and run a few steps  | (in `src/main.py`)          |

All modes take the same two configs: `--estimator` (path to an estimator
YAML) and `--dataset` (path to a dataset YAML). See
[Configuration and data flow](configuration-and-data.md) for the YAML
shape.

## Evaluation workflows

Evaluation runs an estimator on a dataset through the repository
orchestration layer:

```bash
uv run python src/main.py \
    --mode eval \
    --estimator src/flowgym/config/estimators/flow/real_dis.yaml \
    --dataset src/flowgym/config/JHTDB.yaml
```

What this does:

- Loads both YAMLs and builds the estimator with `make_estimator`.
- Constructs a sampler from the dataset config (forcing `loop: false` in
  eval modes so the run terminates).
- Iterates batches through `src/eval.py`, computing metrics from
  `flowgym.common.evaluation` against the ground-truth fields.
- If the dataset config includes a `caching:` block, evaluation goes
  through `flowgym.training.caching.CacheManager` so derived quantities
  can be reused across runs (see the
  [caching examples](../examples/caching.md)).

For a runnable end-to-end version that generates its own tiny dataset,
see the [flow evaluation example](../examples/flow-eval.md) and the
[density evaluation example](../examples/density-eval.md).

## Training workflows

Training adds optimization, replay, and (optionally) periodic validation
on top of the same estimator/dataset config split:

```bash
uv run python src/main.py \
    --mode train-supervised \
    --estimator src/flowgym/config/estimators/flow/real_dis.yaml \
    --dataset src/flowgym/config/JHTDB.yaml
```

Two training entry points exist:

- `--mode train-supervised` runs the supervised loop in
  `src/train_supervised.py`.
- `--mode train` runs the reinforcement-learning loop in `src/train.py`.

Both consume the same `optimizer_config` and (for RL/replay)
`replay_buffer_config` keys inside the estimator's `config:` block.
Checkpoints are written under the run's `out_dir`. See the
[supervised training example](../examples/supervised-training.md) for a
small end-to-end run with validation and checkpoints.

### Periodic validation during training

Add a `validation:` sub-dict to the dataset config to run periodic
evaluation during training:

```yaml
validation:
    dataset: path/to/val_dataset.yaml   # or an inline dict
    interval: 1                         # validate every N batches
    num_batches: 1                      # batches per validation pass
```

`src/main.py` parses this block, loads the validation dataset, and
hands it to the training loop, which calls into `src/eval.py` on the
configured cadence.

## What's in `flowgym.training`

The training-side helpers that the workflows above pull on:

- `flowgym.training.optimizer` — builds Optax transformations from
  `optimizer_config` (e.g. `{"name": "adam", "learning_rate": 1e-3}`).
- `flowgym.training.schedules` — learning-rate schedule builders used
  by `optimizer.py`.
- `flowgym.training.losses` — loss builders selected via the estimator
  config (e.g. supervised flow loss, RL reward terms).
- `flowgym.training.replay` — replay buffer used by RL training,
  configured under `replay_buffer_config`.
- `flowgym.training.caching` — `CacheManager` wired by the dataset
  `caching:` block; used in eval and (optionally) in training to skip
  recomputing expensive derived quantities.
- `flowgym.training.target_transforms` — transforms applied to
  ground-truth targets before loss computation.
- `flowgym.training.exploration` — exploration policies used by the RL
  loop.

For exact signatures, see the
[training and optimization API reference](../api/optimization.md).

## Related pages

- [Estimator API overview](estimator-api.md)
- [Configuration and data flow](configuration-and-data.md)
- [Example workflows](../examples/index.md)
