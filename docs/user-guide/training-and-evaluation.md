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

### Test evaluation after training

An optional `test:` block evaluates the trained model once training
ends: the best-on-validation checkpoint when `validation:` is set,
otherwise the final weights.

```yaml
test:
    dataset: path/to/test_dataset.yaml  # or an inline dict
    num_batches: 30                     # batches in the test pass
```

The results are logged under `test/` (with `test/step`, the checkpoint
used) and written to `<run dir>/test_summary.json`. The test batch size
may differ from the training one.

### Photometric augmentation

Set `photometric_augmentation:` in the dataset config to randomise the
lighting of every training pair (a gain and offset shared by both frames,
and a small gain per frame), so the model cannot rely on a recording's
brightness. Positions are untouched;
labels, cached scores and the initial estimator state use the original
frames.

```yaml
photometric_augmentation: {}    # defaults, for images in [0, 255]
```

Ranges and their defaults: `flowgym.common.augment.photometric_augment`.

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

## Hydra experiment sweeps

The optional Hydra entrypoint reuses the experiment launcher while adding
configuration composition and multirun overrides. The original
`uv run python experiments/run.py --exp example_experiment` command keeps
its existing behavior.

From the repository root:

```bash
uv run --extra experiments python -m experiments.run_hydra \
  exp=example_experiment --multirun seed=0,1,2 \
  '+model_overrides.config.jit=true,false'
```

`exp` selects an existing experiment folder. `exp_root` defaults to
`experiments`; `models=[models/dummy_a.yaml]` selects a subset. `seed=null`
uses the experiment's seeds, while a concrete `seed` runs just that seed.
Use `+experiment_overrides.<key>`, `+model_overrides.<key>`, and
`+dataset_overrides.<key>` for nested overrides. Dataset overrides apply
after model-specific dataset overrides; the selected seed takes precedence
at execution time. Each selected model receives the model overrides.

Each Hydra job stores its resolved configuration in `.hydra/`, generated
model/dataset YAML and summaries in `_generated/`, and aggregate metrics in
`results.json`. Training artifacts go under that job's `runs/` directory.
Source experiment YAML, README results, and existing generated files are
left untouched. Relative dataset, estimator-library, and entrypoint paths
still resolve from the repository root, including with `hydra.job.chdir=true`.
Each child process uses a separate Goggles socket.

Hydra's default launcher runs jobs serially. Install and select an appropriate
Hydra launcher plugin for local parallel or cluster execution; allocate GPUs
through that launcher or its worker environment. Custom Hydra config groups
can represent conditional model configurations using `--config-dir` and
`--config-name`, with `hydra_conf/config.yaml` as the starting schema. A flat
comma-separated sweep is a Cartesian product and does not infer conditions.
See the [Hydra multirun documentation](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/)
for composition, sweepers, and launcher plugins.
