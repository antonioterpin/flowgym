"""Tests for training_utils module."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.training.replay import ReplayBuffer
from flowgym.types import RLExperience


def make_experience(seed=0):
    """Utility to generate a minimal deterministic RLExperience."""
    H, W = 4, 4
    state = {
        "images": jnp.zeros((1, 2, H, W)),
        "estimates": jnp.zeros((1, 1, H, W, 2)),
    }
    reward = jnp.array([float(seed)])
    obs = (jnp.ones((1, H, W)) * seed, jnp.ones((1, H, W)) * (seed + 1))
    old_obs = (jnp.ones((1, H, W)) * (seed - 1), jnp.ones((1, H, W)) * seed)
    old_flow_estimate = jnp.zeros((1, H, W, 2))

    return RLExperience(
        state=state,
        reward=reward,
        obs=obs,
        old_obs=old_obs,
        old_flow_estimate=old_flow_estimate,
    )


# ---------------------------------------------------------------------------
# Basic push and len
# ---------------------------------------------------------------------------


def test_push_and_length():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0

    exp = make_experience(0)
    buf.push(exp)

    assert len(buf) == 1


# ---------------------------------------------------------------------------
# Sampling random batches
# ---------------------------------------------------------------------------


def test_sample_batch_shapes():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(make_experience(i))

    batch = buf.sample(batch_size=3)

    # Check batch shapes
    assert batch.state["images"].shape == (3, 1, 2, 4, 4)
    assert batch.reward.shape == (3, 1)
    assert batch.obs[0].shape == (3, 1, 4, 4)
    assert batch.old_flow_estimate.shape == (3, 1, 4, 4, 2)


# ---------------------------------------------------------------------------
# Sampling deterministically by index
# ---------------------------------------------------------------------------


def test_sample_at_indices():
    buf = ReplayBuffer(capacity=5)
    for i in range(5):
        buf.push(make_experience(i))

    batch = buf.sample_at([0, 3])
    assert batch.reward.shape == (2, 1)

    # Ensure deterministic: rewards equal seeds
    assert float(batch.reward[0, 0]) == 0.0
    assert float(batch.reward[1, 0]) == 3.0


# ---------------------------------------------------------------------------
# Error: sampling too large batch
# ---------------------------------------------------------------------------


def test_sample_too_large_batch_raises():
    buf = ReplayBuffer(capacity=3)
    buf.push(make_experience(0))

    with pytest.raises(ValueError):
        buf.sample(batch_size=5)


# ---------------------------------------------------------------------------
# Error: sample_at invalid indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_indices",
    [
        [-1],  # negative index
        [0, 5],  # out of range
        7,  # scalar invalid
    ],
)
def test_sample_at_invalid_indices_raises(bad_indices):
    buf = ReplayBuffer(capacity=3)
    for i in range(3):
        buf.push(make_experience(i))

    with pytest.raises((IndexError, ValueError)):
        buf.sample_at(bad_indices)


# ---------------------------------------------------------------------------
# Clearing the buffer
# ---------------------------------------------------------------------------


def test_clear_buffer():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(make_experience(i))

    assert len(buf) == 5
    buf.clear()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# CPU device check
# ---------------------------------------------------------------------------


def test_arrays_are_on_cpu_after_push():
    buf = ReplayBuffer(capacity=4)

    exp = make_experience(0)
    buf.push(exp)

    stored = buf.buffer[0]
    # All fields should be placed on CPU
    for leaf in jax.tree_util.tree_leaves(stored):
        assert leaf.device == buf.cpu_device
