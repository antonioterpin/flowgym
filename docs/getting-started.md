# Getting started

This page gives you a minimal local path from install to first estimator.

## Install FlowGym

For the package only:

```bash
uv add flow-gym-suite
```

With `pip` instead:

```bash
pip install flow-gym-suite
```

If you want to work on the repository itself:

```bash
uv sync --group dev
```

To build the documentation locally:

```bash
uv sync --group docs
uv run --group docs make -C docs html
```

## Create an estimator

```python
from flowgym.make import make_estimator

model_config = {
    "estimator": "dis_jax",
    "estimate_type": "flow",
    "config": {"jit": True},
}

trained_state, create_state_fn, compute_estimate_fn, model = make_estimator(
    model_config=model_config,
    image_shape=image_shape,
)
```

## Run one estimate

```python
est_state = create_state_fn(prev, rng)
new_est_state, metrics = compute_estimate_fn(curr, est_state, trained_state)
```

## Next places to look

- [API reference](api/index.md) for the import surface and core modules
- [Contributing guide](guides/contributing.md) for local development
- [Architecture guide](guides/architecture.md) for the repository layout
- [Project docs](project-docs.md) for the full documentation map
