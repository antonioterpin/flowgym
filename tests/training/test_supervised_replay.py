"""Tests for supervised replay buffer integration."""

import jax
import jax.numpy as jnp

from flowgym.training.replay import ReplayBuffer
from flowgym.types import SupervisedExperience


def test_supervised_experience_pytree():
    """Verify SupervisedExperience is a valid JAX PyTree."""
    state = {"h": jnp.zeros((1, 2))}
    exp = SupervisedExperience(
        state=state,
        obs=(jnp.ones((1, 4, 4)), jnp.ones((1, 4, 4)) * 2),
        ground_truth=jnp.ones((1, 4, 4, 2)),
    )

    # Flatten and unflatten
    leaves, treedef = jax.tree_util.tree_flatten(exp)
    exp_new = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(exp_new, SupervisedExperience)
    assert jnp.array_equal(exp_new.obs[0], exp.obs[0])
    assert exp_new.state["h"].shape == (1, 2)


def test_replay_buffer_with_supervised_experience():
    """Verify ReplayBuffer works with SupervisedExperience."""
    buf = ReplayBuffer(capacity=10)

    for i in range(5):
        exp = SupervisedExperience(
            state={"h": jnp.zeros((1, 2))},
            obs=(jnp.ones((1, 4, 4)) * i, jnp.ones((1, 4, 4)) * (i + 1)),
            ground_truth=jnp.ones((1, 4, 4, 2)) * i,
        )
        buf.push(exp)

    assert len(buf) == 5

    # Sample a batch
    batch = buf.sample(batch_size=3)
    assert batch.obs[0].shape == (3, 1, 4, 4)
    assert batch.ground_truth.shape == (3, 1, 4, 4, 2)
    # Check that it moved to CPU
    assert batch.obs[0].device == buf.cpu_device


def test_replay_buffer_sample_at_with_supervised_experience():
    """Verify sample_at works with SupervisedExperience."""
    buf = ReplayBuffer(capacity=10)
    for i in range(2):
        exp = SupervisedExperience(
            state={"h": jnp.zeros((1, 2))},
            obs=(jnp.ones((1, 4, 4)) * i, jnp.ones((1, 4, 4))),
            ground_truth=jnp.ones((1, 4, 4, 2)),
        )
        buf.push(exp)

    batch = buf.sample_at([0, 1])
    assert jnp.array_equal(
        batch.obs[0],
        jnp.stack([jnp.ones((1, 4, 4)) * 0, jnp.ones((1, 4, 4)) * 1]),
    )
