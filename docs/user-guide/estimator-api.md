# Estimator API overview

Flow Gym organizes different methods around a common `Estimator`
interface. Whether a method is classical or learning-based, package-level
use follows the same basic contract for creating runtime state, stepping
through observations, and carrying estimator parameters.

For exact signatures and docstring-level details, see:

- [Package and factory](../api/package.md) for `flowgym.make` and
  `make_estimator(...)`
- [Base classes and environment](../api/base.md) for the base `Estimator`
  class and trainable-state types
- [Estimators](../api/estimators.md) for concrete estimator implementations

## What an `Estimator` is

In Flow Gym, an `Estimator` is an object that maps observations to
quantities of interest through a shared interface.

For flow-field quantification, the observation is often a tracer-particle
image or image pair, and the quantity of interest is a flow estimate. The
same interface can also be used for related tasks, such as density
estimation.

The point of the interface is not that all estimators work the same way
internally. The point is that they can be used through the same external
contract even when their internals are very different.

## The call signature

The `Estimator` interface follows JAX's functional style. Each estimation
step has the shape:

```python
new_state, metrics = estimator(images, state, trainable_state, cache_payload=None)
```

Each call receives:

- `images`: the latest observation, batched as `(B, H, W)`.
- `state`: the current runtime state — a dict; see [Runtime state and
  trainable state](#runtime-state-and-trainable-state).
- `trainable_state`: the long-lived parameters of the estimator.
- `cache_payload` (optional): pre-computed quantities for this batch
  from `flowgym.training.caching`. Used only for this step; not
  persisted in `state`.

and returns:

- an updated runtime `state`
- a `metrics` dictionary for logging or inspection

The call is side-effect free: the input `images` and the
`trainable_state` are not mutated in place. Instead, the next runtime
state is returned explicitly.

That makes the data flow visible. The caller can see exactly what
observation goes in, what context is carried forward, and what metrics
come out.

### `estimator(...)` vs `compute_estimate_fn(...)`

`make_estimator(...)` returns a `compute_estimate_fn` that wraps this
contract and (optionally) compiles it with `jax.jit`. The arguments
and return types are the same either way:

```python
new_state, metrics = compute_estimate_fn(images, state, trained_state)
# is equivalent to (modulo JIT compilation):
new_state, metrics = estimator(images, state, trained_state)
```

Because the wrapper preserves the signature whether or not JIT is on,
toggling the `jit` key inside `config` does not require any change to
the surrounding eval or training loop. The key is nested under
`config:` in YAML (not a literal dotted `config.jit:` key); you can
develop and debug with `jit: false` and flip it to `true` for
production runs.

## Runtime state and trainable state

### Why the split

Flow Gym separates runtime state from trainable state because they
change on different timescales and serve different roles:

- runtime state changes as a sequence is processed
- trainable state changes only when the estimator is trained,
  restored, or otherwise updated persistently

This split also matches how JAX works best:

- the estimator step can be treated as a pure state transition
- the computation can be compiled with `jax.jit`
- batching, checkpointing, and repeated evaluation are easier to
  reason about because the evolving context is explicit

### Runtime state

The runtime `state` is a dict that captures the context propagated
across successive calls. After `create_state(...)`, two keys are
always present:

- `state["images"]`: shape `(B, image_history_size, H, W)` — the
  rolled history of input frames.
- `state["estimates"]`: shape
  `(B, estimate_history_size, *estimate_shape)` — the rolled history
  of estimates.

Two more keys are optional:

- `state["keys"]`: per-batch PRNG keys, present only when a `rng` was
  passed to `create_state(...)`. Stochastic estimators split each
  per-batch key on every call so randomness is reproducible across
  steps.
- Estimator-specific extras: extra fields registered by the
  estimator's `_create_extras()` (e.g. recurrent hidden state, reward
  history). Each is batched `(B, ...)` and rolled forward like the
  base fields.

Because the state is explicit, the same interface supports both:

- one-shot estimators: methods that only need the current observation
  or image pair, like classical PIV methods
- recurrent estimators: methods that exploit short-term temporal
  context across calls, like learning-based recurrent estimators or
  classical methods with multi-frame processing

When image pairs are processed independently, such as during
benchmarking on a shuffled dataset, the runtime state can simply be
re-initialized before each estimation.

### Trainable state

The `trainable_state` captures the long-term parameters of the
estimator. For learning-based methods, this usually includes estimator
parameters, optimizer state, and related training information.

Classical estimators usually do not have trainable parameters. In
those cases, Flow Gym still passes an empty trainable-state container
through the same interface. That is what allows classical and
learning-based methods to be swapped into the same workflow without
changing the surrounding pipeline.

## Using `make_estimator(...)`

Most users do not instantiate an `Estimator` subclass directly. They
start with `flowgym.make.make_estimator(...)`, which builds the
estimator and returns the pieces needed to run it in a package-level
workflow.

`make_estimator(...)` returns four objects:

- a `trainable_state`
- a `create_state_fn` helper for initializing runtime state
- a `compute_estimate_fn` helper for stepping the estimator
- the concrete `Estimator` object

In practice, the calling code looks like this:

```python
trained_state, create_state_fn, compute_estimate_fn, estimator = make_estimator(
    estimator_config=estimator_config,
    image_shape=image_shape,
    estimate_shape=estimate_shape,
    rng=0,
)

state = create_state_fn(first_frame, rng)
next_state, metrics = compute_estimate_fn(next_frame, state, trained_state)
```

The helpers are not a different abstraction from the `Estimator`
contract — they are the package-level way Flow Gym prepares that
contract for real use, including shape inference, compilation, and
estimator construction.

For the generated reference docs behind this flow, see:

- [Package and factory](../api/package.md) for `make_estimator(...)`
  and the factory helpers
- [Base classes and environment](../api/base.md) for the base
  estimator methods and trainable-state types

## Hooks around estimation

The interface gives Flow Gym standard places to attach processing steps
around estimation:

- **preprocessing**: applied to `images` before estimation. Configured
  on every `Estimator` via `config.preprocess`.
- **postprocessing**: applied to the produced estimate. Available on
  flow-field estimators (`FlowFieldEstimator`) via `config.postprocess`.

Both are configured in the estimator YAML as a list of named
transforms:

```yaml
config:
    preprocess:
        - name: crop_special
          target_height: 832
          target_width: 1024
        - name: intensity_capping
          n: 1.1
    postprocess:
        - name: resize_flow
          target_height: 64
          target_width: 64
```

Either key can also point at a reusable YAML under
`src/flowgym/config/estimators/pre-processing/` or `.../post-processing/`
instead of an inline list — see
[Configuration and data flow](configuration-and-data.md#preprocessing-and-postprocessing).

## Related pages

- [Quick overview](../getting-started/quick-overview.md)
- [API reference](../api/index.md)
- [Configuration and data flow](configuration-and-data.md)
- [Training and evaluation workflows](training-and-evaluation.md)
