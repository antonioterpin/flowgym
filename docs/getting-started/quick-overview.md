# Quick overview

This page shows a minimal example of the core estimator API.

It is not meant to be a full tutorial, but rather a quick sketch of the main
steps for configuring, building, and running an estimator. For more context on
the API design and the estimator model, see the
[User guide](../user-guide/index.md).

## Configure the estimator

To begin, decide on an estimator configuration. This is a dictionary that
specifies the estimator family, the type of estimate, and any relevant
configuration options. For example:

```python
estimator_config = {
    "estimator": "dis_jax",
    "estimate_type": "flow",
    "config": {
        "jit": False,
        "preset": 1,
    },
}

image_shape = (1, 64, 64)
estimate_shape = (1, 64, 64, 2)
```

This example uses a JAX implementation of the
[Dense Inverse Search (DIS) optical flow algorithm](https://arxiv.org/abs/1603.03590),
the `dis_jax` flow estimator, but the same factory pattern is used for other
estimator families as well.

## Build the estimator

Next, call `make_estimator(...)` to build the estimator and get the pieces
needed to run it:

```python
import jax
import jax.numpy as jnp

from flowgym.make import make_estimator

trained_state, create_state_fn, compute_estimate_fn, estimator = make_estimator(
    estimator_config=estimator_config,
    image_shape=image_shape,
    estimate_shape=estimate_shape,
    rng=0,
)
```

The returned objects separate two kinds of state:

- `trained_state`:
  learned parameters and other persistent estimator state
- `create_state_fn` and `compute_estimate_fn`:
  helpers for the runtime estimation loop

This split is what lets FlowGym use the same interface for classical and
learning-based methods. The `trained_state` is an empty dictionary for 
non-trainable methods, but it can still be passed around and used in the same way as for
trainable methods, which makes it easier to swap methods in and out of the same workflow.

## Deploy the estimator

Now you can initialize the runtime state from an input image and run one
estimate of the flow between the first and second image:

```python
import jax
import jax.numpy as jnp

prev = jnp.zeros(image_shape, dtype=jnp.float32)
curr = jnp.ones(image_shape, dtype=jnp.float32)
rng = jax.random.PRNGKey(0)

estimator_state = create_state_fn(prev, rng)
new_state, metrics = compute_estimate_fn(curr, estimator_state, trained_state)
```

In this pattern, the updated sequence state is returned explicitly rather
than being hidden inside the estimator object. That is an important part of
the FlowGym design: the same estimation step can support one-off or
consecutive workflows while remaining compatible with JAX-style execution.

This example uses synthetic arrays to illustrate the current public API. For
dataset-backed runs, benchmarking, and repository workflows, move to the
example and guide pages below.

In particular, the [Example workflows](../examples/index.md) include walkthroughs of
- running an estimator on a dataset with `src.main`
- training a learning-based estimator with `src.main`
- using the cache for expensive derived quantities

Instead, if you want to understand the API design and the estimator model in more depth, see the
[User guide](../user-guide/index.md) for a detailed introduction to the
estimator state, configuration, and runtime concepts.

## Next places to look

- [User guide](../user-guide/index.md) for the estimator model, runtime
  state, and
  configuration concepts
- [Example workflows](../examples/index.md) for evaluation, training, and
  caching walkthroughs
- [API reference](../api/index.md) for modules, classes, and functions
- [Contributing guide](../guides/contributing.md) for repository-level
  workflow
