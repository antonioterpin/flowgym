"""Mock estimators, samplers, environments and trainable states.

These back the unit and integration tests for the training stack without
requiring a real estimator, dataset, or environment.
"""

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest


@pytest.fixture
def mock_dependencies():
    """Fixture providing mock estimator, env, observations for training tests."""
    estimator = MagicMock()
    estimator.create_train_step.return_value = MagicMock(
        return_value=(0.1, MagicMock(), {})
    )
    estimator.process_metrics.side_effect = lambda x: x

    env = MagicMock()
    # env.reset returns (obs, state, done)
    # obs is (prev, curr)
    obs = (jnp.zeros((2, 4, 4)), jnp.zeros((2, 4, 4)))
    env_state = (MagicMock(), jnp.zeros((2, 4, 4, 2)))
    done = jnp.array([False, False])
    env.reset.return_value = (obs, env_state, done)

    # env.step returns (obs, state, reward, done)
    reward = jnp.array([0.0, 0.0])
    env.step.return_value = (obs, env_state, reward, jnp.array([True, True]))

    return estimator, env, obs, env_state


@pytest.fixture
def dummy_trainable_state():
    """Create a minimal NNEstimatorTrainableState for testing."""
    import optax
    from flax.core import FrozenDict

    from flowgym.common.base.trainable_state import NNEstimatorTrainableState

    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(1.0, jnp.float32)})
    tx = optax.sgd(0.01)
    return NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )


@pytest.fixture
def mock_sampler():
    """Create a mock sampler that yields synthetic batches."""
    B, H, W = 2, 32, 32
    batch = MagicMock()
    batch.images1 = jnp.zeros((B, H, W))
    batch.images2 = jnp.zeros((B, H, W))
    batch.flow_fields = jnp.zeros((B, H, W, 2))
    batch.params = None
    batch.mask = None

    sampler = MagicMock()
    # Return 10 batches then stop
    sampler.__iter__.return_value = iter([batch] * 10)
    sampler.shutdown = MagicMock()
    sampler.reset = MagicMock()
    # Set grain_iterator to None so save_estimator skips sampler serialization
    sampler.grain_iterator = None
    return sampler


@pytest.fixture
def mock_env():
    """Create a mock environment for RL training."""
    env = MagicMock()
    B, H, W = 2, 32, 32
    obs = (jnp.zeros((B, H, W)), jnp.zeros((B, H, W)))
    env_state = (MagicMock(), jnp.zeros((B, H, W, 2)))

    env.reset.return_value = (obs, env_state, jnp.array([False, False]))
    env.step.return_value = (
        obs,
        env_state,
        jnp.array([1.0, 1.0]),
        jnp.array([True, True]),
    )
    return env, obs, env_state
