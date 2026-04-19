# Estimator API overview

The `Estimator` API is the center of FlowGym's public package surface.

Most users start with `flowgym.make.make_estimator`, which builds an
estimator from a configuration dictionary and returns the pieces 
needed to run it.

FlowGym is organized around a unified `Estimator` interface: different
flow-field quantification methods, and even related tasks such as density
estimation, can exist as instances of an `Estimator` as long as they expose the
same estimation contract.

## The main pieces

At a high level, `make_estimator` returns:

- an `EstimatorTrainableState` object:
  the learned parameters and optimizer-related state for the estimator, if
  applicable, or an empty state if the estimator is not trainable
- a state-construction function:
  a helper that creates a `History` object from an input frame
- an estimate function:
  a helper that advances the estimator on the next frame and returns the new
  runtime state together with metrics
- an `Estimator` object:
  the concrete estimator implementation

This means that FlowGym separates two different kinds of state:

- trainable state:
  model parameters, optimizer state, and other persistent learned values
- runtime state:
  the per-sequence history needed to estimate the next frame, such as image
  history, estimate history, and optional RNG/history extras

## A mental model for the two states

One useful way to think about the API is:

- `trainable_state` answers:
  "what has this estimator learned?"
- runtime `state` answers:
  "what does this estimator currently remember about this sequence?"

`trainable_state` is the long-term part of the estimator. For learning-based
methods, it typically contains model weights, optimizer state, and related
training objects. For classical methods, it can simply be an empty
container, which lets FlowGym run classical and learning-based estimators
through the same pipeline.

The runtime `state` is the short-term context propagated across successive
calls. It changes every time you advance through a sequence of images.

That is why FlowGym keeps them separate: one is long-lived model knowledge,
the other is run-time context.

## The usual flow

Most package-level use follows the same pattern:

1. define a `model_config`
2. call `make_estimator(...)`
3. initialize the runtime state from an input frame
4. compute estimates on subsequent frames

In practice, that usually looks like this:

```python
trained_state, create_state_fn, compute_estimate_fn, model = make_estimator(
    ...
)

state = create_state_fn(first_frame, rng)
state, metrics = compute_estimate_fn(next_frame, state, trained_state)
```

The key point is that `compute_estimate_fn(...)` does not hide state updates
inside the estimator object. It returns the next runtime state explicitly.

## Why the API is split this way

By this point, the main design choice in FlowGym is hopefully visible:
estimation is expressed as an explicit state transition, not as hidden
mutation inside an object.

To align with JAX, that contract is stateless and functional. Each
estimation step has the shape:

```python
new_state, metrics = estimator(image, state, trainable_state)
```

That shape is deliberate. It matches how JAX works best, and it also makes
the estimator step easy to inspect: the inputs, the evolving history, and
the returned outputs are all visible in one place.

It also brings the usual JAX benefits:

- `jax.jit` can compile the computation ahead of time
- the compiled function can execute efficiently on GPUs and other
  accelerators
- data flow is explicit, which makes batching, checkpointing, and repeated
  evaluation easier to reason about
- there is less hidden Python-side mutation that would interfere with JAX's
  tracing and compilation model

## What lives in the runtime state

The runtime `state` is created by `Estimator.create_state(...)` and is
designed as a JAX-compatible history object.

It usually contains at least:

- `"images"`:
  the image history seen so far
- `"estimates"`:
  the estimate history produced so far
- optional `"keys"`:
  per-example random keys for deterministic JAX randomness
- optional extras:
  estimator-specific history fields

In practice, this means the same interface can support both:

- one-shot estimators:
  methods that only need the current image pair
- recurrent estimators:
  methods that retain short-term history, previous estimates, recurrent
  variables, or controlled randomness across calls

The updated `state` returned by each call includes the rolled history and,
when used, updated PRNG keys to carry controlled randomness into the next
step.

## Where the pieces live

- `flowgym.make`:
  factory helpers for building estimators, compiling the step functions, and
  handling checkpoints
- `flowgym.common.base`:
  shared interfaces and state types
- `flowgym.flow` and `flowgym.density`:
  concrete estimator families

If you want to see the exact implementation shape, the two most relevant
places are:

- `src/flowgym/make.py`
- `src/flowgym/common/base/estimator.py`

## Related pages

- [Quick overview](../getting-started/quick-overview.md)
- [API reference](../api/index.md)
- [Training and evaluation workflows](training-and-evaluation.md)
