# Estimator API overview

FlowGym organizes different methods around a common `Estimator` interface.
Whether a method is classical or learning-based, package-level use follows
the same basic contract for creating state, stepping through observations,
and carrying estimator parameters.

For exact signatures and docstring-level details, see:

- [Package and factory](../api/package.md) for `flowgym.make` and
  `make_estimator(...)`
- [Base classes and environment](../api/base.md) for the base `Estimator`
  class and trainable-state types
- [Estimators](../api/estimators.md) for concrete estimator implementations

## What an `Estimator` is

In FlowGym, an `Estimator` is an object that maps observations to
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
new_state, metrics = estimator(image, state, trainable_state)
```

Each call receives:

- the latest observation
- the current runtime `state`
- the long-lived `trainable_state`

and returns:

- an updated runtime `state`
- a `metrics` dictionary for logging or inspection

The call is side-effect free: the input observation and the
`trainable_state` are not mutated in place. Instead, the next runtime state
is returned explicitly.

That makes the data flow visible. The caller can see exactly what
observation goes in, what context is carried forward, and what metrics come
out.

## Runtime state and trainable state

An estimator in FlowGym has two kinds of state:

- runtime `state`:
  the short-term context needed to estimate the next observation
- `trainable_state`:
  the long-term parameters of the estimator

### Runtime state

The runtime `state` captures the context propagated across successive calls.
For simple algorithms, that context may be limited to the current image pair
or the rolled image and estimate history. More advanced methods may retain
previous estimates, recurrent internal variables, or controlled randomness.

In practice, the runtime state commonly includes:

- image history
- estimate history
- optional estimator-specific extras
- optional PRNG keys carried across calls

Because the state is explicit, the same interface supports both:

- one-shot estimators:
  methods that only need the current observation or image pair, like
  classical PIV methods
- recurrent estimators:
  methods that exploit short-term temporal context across calls, like 
  learned recurrent models or classical methods with multi-frame processing

When image pairs are processed independently, such as during benchmarking on
a shuffled dataset, the runtime state can simply be re-initialized before
each estimation.

The runtime state is created through the estimator-state helpers exposed by
the package. The exact state object and initialization details are
documented in [Base classes and environment](../api/base.md) and
[Package and factory](../api/package.md).

### Trainable state

The `trainable_state` captures the long-term parameters of the estimator.
For learning-based methods, this usually includes model weights, optimizer
state, and related training information.

Classical estimators usually do not have trainable parameters. In those cases,
FlowGym still passes an empty trainable-state container through the same
interface. That is what allows classical and learning-based methods to be
swapped into the same workflow without changing the surrounding pipeline.

## Why the split is useful

FlowGym separates runtime state from trainable state because they change on
different timescales and serve different roles.

- runtime state changes as a sequence is processed
- trainable state changes only when the estimator is trained, restored, or
  otherwise updated persistently

This split also matches how JAX works best:

- the estimator step can be treated as a pure state transition
- the computation can be compiled with `jax.jit`
- batching, checkpointing, and repeated evaluation are easier to reason
  about because the evolving context is explicit

## Using `make_estimator(...)`

Most users do not instantiate an `Estimator` subclass directly. They start
with `flowgym.make.make_estimator(...)`, which builds the estimator and
returns the pieces needed to run it in a package-level workflow.

At a high level, `make_estimator(...)` returns:

- a `trainable_state`
- a `create_state_fn` helper for initializing runtime state
- a `compute_estimate_fn` helper for stepping the estimator
- the concrete `Estimator` object

That means the usual public flow looks like this:

```python
trained_state, create_state_fn, compute_estimate_fn, estimator = make_estimator(
    estimator_config=estimator_config,
    ...
)

state = create_state_fn(first_frame, rng)
next_state, metrics = compute_estimate_fn(next_frame, state, trained_state)
```

The helper functions are not a different abstraction from the `Estimator`
contract. They are the package-level way FlowGym prepares that contract for
real use, including shape inference, compilation, and estimator
construction.

For the generated reference docs behind this flow, see:

- [Package and factory](../api/package.md) for `make_estimator(...)` and
  the factory helpers
- [Base classes and environment](../api/base.md) for the base estimator
  methods and trainable-state types

## Hooks around estimation

The interface also gives FlowGym standard places to attach processing steps
around estimation.

- pre-processing:
  configured on the base `Estimator` and applied before estimation
- post-processing:
  available on flow-field estimators through the `FlowFieldEstimator`
  subclass

This lets pre-processing, estimation, and post-processing be specified
independently instead of being entangled inside each algorithm.

The generated reference docs for those hooks are in:

- [Base classes and environment](../api/base.md)
- [Estimators](../api/estimators.md)

## Related pages

- [Quick overview](../getting-started/quick-overview.md)
- [API reference](../api/index.md)
- [Configuration and data flow](configuration-and-data.md)
- [Training and evaluation workflows](training-and-evaluation.md)
