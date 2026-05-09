"""Smallest meaningful FlowGym example: build an estimator and run it once."""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_API_KEY", "example")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from flowgym.make import make_estimator


def make_test_images(shape: tuple[int, int] = (64, 64)) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create two tiny grayscale images with a shifted bright square."""
    height, width = shape
    prev = jnp.zeros((1, height, width), dtype=jnp.float32)
    curr = jnp.zeros((1, height, width), dtype=jnp.float32)

    prev = prev.at[:, 20:36, 18:34].set(255.0)
    curr = curr.at[:, 20:36, 21:37].set(255.0)
    return prev, curr


def run_example() -> None:
    """Instantiate a DIS estimator and run it on one image pair."""
    estimator_config = {
        "estimator": "dis_jax",
        "estimate_type": "flow",
        "config": {
            "jit": False,
            "preset": 1,
            "patch_size": 7,
            "patch_stride": 7,
            "grad_desc_iters": 2,
            "levels": 1,
            "output_full_res": True,
        },
    }

    prev, curr = make_test_images()
    image_shape = prev.shape
    estimate_shape = (*image_shape, 2)

    trained_state, create_state_fn, compute_estimate_fn, estimator = make_estimator(
        estimator_config=estimator_config,
        image_shape=image_shape,
        estimate_shape=estimate_shape,
        rng=0,
    )

    rng = jax.random.PRNGKey(0)
    state = create_state_fn(prev, rng)
    new_state, metrics = compute_estimate_fn(curr, state, trained_state)
    flow_estimate = new_state["estimates"][:, -1]

    print("Estimator:", type(estimator).__name__)
    print("Input image shape:", tuple(prev.shape))
    print("Estimate tensor shape:", tuple(flow_estimate.shape))
    print("Metrics keys:", sorted(metrics.keys()))
    print(
        "Flow summary:",
        {
            "mean_u": float(jnp.mean(flow_estimate[..., 0])),
            "mean_v": float(jnp.mean(flow_estimate[..., 1])),
            "max_magnitude": float(jnp.max(jnp.linalg.norm(flow_estimate, axis=-1))),
        },
    )
    print("Estimate lives in state['estimates'][:, -1].")


if __name__ == "__main__":
    run_example()
