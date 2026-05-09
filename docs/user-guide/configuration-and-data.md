# Configuration and data flow

Flow Gym uses YAML configuration files to wire estimators, datasets, and
repository-level workflows together. Two kinds of config carry the
load-bearing decisions: an **estimator config** describes which estimator
to build and how, and a **dataset config** describes the data source.
Repository workflows (`src/main.py`) take one of each.

## The two main kinds of config

### Estimator config

A minimal flow estimator config:

```yaml
estimator: dis_jax
estimate_type: flow
estimate_shape: [64, 64, 2]
config:
    jit: true
    preset: 1
    patch_size: 13
    grad_desc_iters: 16
```

The top-level keys split between two consumers:

- `estimator` and `config` are read by `flowgym.make.make_estimator`.
  The string in `estimator` selects the class
  (e.g. `dis_jax` → `DISJAXFlowFieldEstimator`) and the `config` dict
  is forwarded to its `from_config(...)` factory.
- `estimate_type` and `estimate_shape` are read by the orchestration
  layer in `src/main.py` to drive ground-truth selection and shape
  inference. They are not consumed by `make_estimator` itself.

Inside the `config` block, most keys are estimator-specific. A few are
shared across estimator families:

- `jit` (bool): whether to compile the estimation step with `jax.jit`.
- `history_size` (int): how many past frames/estimates to retain in
  runtime state. Defaults to 1.
- `preprocess` / `postprocess`: lists of named transforms applied
  around estimation (see [Preprocessing and postprocessing](#preprocessing-and-postprocessing)).
- `optimizer_config`, `replay_buffer_config`: present only on
  trainable estimators; consumed by `flowgym.training.optimizer` and
  the training loop.

### Dataset config

A minimal dataset config (synthetic frames driven by a flow file
list):

```yaml
seed: 0
batch_size: 16
image_shape: [1216, 1936]
flow_fields_per_batch: 16
batches_per_flow_batch: 1
loop: false
randomize: false
include_images: false
scheduler_files:
    - path/to/flows.h5
scheduler_class: ".h5"
```

Load-bearing keys:

- `seed`, `batch_size`, `image_shape`: reproducibility and per-step
  shape.
- `loop`, `randomize`, `include_images`: sampler behavior. Note that
  `src/main.py` overrides some of these per `--mode` (e.g. eval modes
  force `loop: false`).
- `scheduler_files` and `scheduler_class`: the data source — typically
  a list of `.h5` or `.npy` files plus the file extension that selects
  the synthpix scheduler.
- `caching` (optional): a sub-dict that wires
  `flowgym.training.caching.CacheManager` for cache-backed evaluation.
  See the [caching examples](../examples/caching.md).
- `validation` (optional): a sub-dict with `dataset:`, `interval:`, and
  `num_batches:` keys, used during training to run periodic evaluation
  on a held-out dataset config.

## Preprocessing and postprocessing

The `preprocess` / `postprocess` keys inside an estimator's `config`
block are lists of named transforms. They can be written inline:

```yaml
config:
    # ...
    preprocess:
        - name: crop_special
          target_height: 832
          target_width: 1024
          fraction_v: 0.5
          fraction_h: 1.0
        - name: intensity_capping
          n: 1.1
    postprocess:
        - name: resize_flow
          target_height: 64
          target_width: 64
```

Or they can be replaced with a path to a reusable YAML under
`src/flowgym/config/estimators/pre-processing/` or
`.../post-processing/`:

```yaml
config:
    # ...
    preprocess: src/flowgym/config/estimators/pre-processing/real.yaml
    postprocess: src/flowgym/config/estimators/post-processing/real.yaml
```

The repository ships a few presets (e.g. `pre-processing/real.yaml`,
`post-processing/real.yaml`) for typical real-image workflows.

## How they come together

`src/main.py` is the user-facing entrypoint:

```bash
uv run python src/main.py \
    --mode eval \
    --estimator src/flowgym/config/estimators/flow/real_dis.yaml \
    --dataset src/flowgym/config/JHTDB.yaml
```

It reads both YAMLs, builds the estimator with
`make_estimator(estimator_config, image_shape=..., estimate_shape=...)`,
constructs the sampler or environment from the dataset config, and
dispatches into `src/eval.py`, `src/train.py`, or
`src/train_supervised.py` depending on `--mode`.

When you use Flow Gym as a package (no CLI), you pass the same
configuration dicts directly to `make_estimator`, as shown in the
[Estimator API overview](estimator-api.md) and the
[first-estimate](../examples/first-estimate.md) walkthrough.

## Where configs live

```text
src/flowgym/config/
├── JHTDB.yaml, cylinder.yaml, density.yaml, ...   # dataset configs
└── estimators/
    ├── flow/             # flow estimator presets (e.g. real_dis.yaml)
    ├── density/          # density estimator presets
    ├── pre-processing/   # reusable preprocess pipelines
    └── post-processing/  # reusable postprocess pipelines
```

Use these as starting points: copy a preset, adjust the keys you care
about, and pass the path to `--estimator` or `--dataset`.

## Related pages

- [Training and evaluation workflows](training-and-evaluation.md)
- [API reference](../api/index.md)
- [Caching examples](../examples/caching.md)
