"""Tests for the elementwise ``clip`` transform in the optimizer chain."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax

from flowgym.training.optimizer import (
    TRANSFORM_REGISTRY,
    build_optimizer_from_config,
)


def test_clip_is_in_transform_registry():
    """``optax.clip`` must be registered as ``"clip"`` so that
    ``optimizer_config.chain`` can request elementwise clipping by name."""
    assert "clip" in TRANSFORM_REGISTRY
    assert TRANSFORM_REGISTRY["clip"] is optax.clip


def test_chain_with_clip_clips_gradients_elementwise():
    """A chain entry of ``{"name": "clip", "kwargs": {"max_delta": v}}`` must
    clip incoming gradients to ``[-v, v]`` before the base optimizer scales
    them by the learning rate."""
    max_delta = 1.0
    learning_rate = 0.5

    tx = build_optimizer_from_config(
        {
            "name": "sgd",
            "hyperparams": {"learning_rate": learning_rate},
            "chain": [
                {"name": "clip", "kwargs": {"max_delta": max_delta}},
            ],
        }
    )

    params = {"w": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([5.0, -5.0, 0.3], dtype=jnp.float32)}

    opt_state = tx.init(params)
    updates, _ = tx.update(grads, opt_state, params)

    # Elementwise clip to [-1, 1] then SGD applies -lr * clipped_grads.
    expected = jnp.array(
        [
            -learning_rate * max_delta,
            learning_rate * max_delta,
            -learning_rate * 0.3,
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(updates["w"], expected, atol=1e-6)


def test_chain_without_clip_does_not_clip():
    """The default code path (no chain) must not clip — gradients pass through
    to the base optimizer unchanged."""
    learning_rate = 0.5
    tx = build_optimizer_from_config(
        {"name": "sgd", "hyperparams": {"learning_rate": learning_rate}}
    )

    params = {"w": jnp.array([0.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([5.0], dtype=jnp.float32)}

    opt_state = tx.init(params)
    updates, _ = tx.update(grads, opt_state, params)

    expected = jnp.array([-learning_rate * 5.0], dtype=jnp.float32)
    np.testing.assert_allclose(updates["w"], expected, atol=1e-6)
