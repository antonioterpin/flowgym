"""Tests for ReplayBuffer integration in train.py."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from flowgym.training.replay import ReplayBuffer
from train import train


def test_train_replay_initialization(mock_dependencies):
    model, env, obs, env_state = mock_dependencies

    # We want to check if ReplayBuffer is initialized
    with patch("train.ReplayBuffer", wraps=ReplayBuffer) as mock_buffer:
        train(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=MagicMock(),
            out_dir="tmp",
            create_state_fn=lambda img, key: {
                "images": jnp.zeros((2, 1, 4, 4)),
                "estimates": jnp.zeros((2, 1, 4, 4, 2)),
            },
            compute_estimate_fn=lambda img, state, ts: (state, {}),
            env=env,
            env_state=env_state,
            num_episodes=1,
            save_every=10,
            log_every=1,
            obs=obs,
            key=jax.random.PRNGKey(0),
            replay_buffer_capacity=100,
            replay_ratio=0.0,
        )

        mock_buffer.assert_called_once()


def test_train_replay_execution(mock_dependencies):
    model, env, obs, env_state = mock_dependencies
    train_step_fn = model.create_train_step.return_value

    train(
        model=model,
        model_config={"config": {"jit": False}},
        trainable_state=MagicMock(),
        out_dir="tmp",
        create_state_fn=lambda img, key: {
            "images": jnp.zeros((2, 1, 4, 4)),
            "estimates": jnp.zeros((2, 1, 4, 4, 2)),
        },
        compute_estimate_fn=lambda img, state, ts: (state, {"foo": 0.0}),
        env=env,
        env_state=env_state,
        num_episodes=1,
        save_every=10,
        log_every=1,
        obs=obs,
        key=jax.random.PRNGKey(0),
        replay_buffer_capacity=100,
        replay_ratio=1.0,  # Force at least one replay step
    )

    # Verify train_step_fn called for collected experience and replay
    # In one episode with one step, it should be called:
    # 1. Once for the environment step
    # 2. Once for replay step (since replay_ratio=1.0 and buf has B=2 items)
    assert train_step_fn.call_count >= 2

    # Verify signature
    _, kwargs = train_step_fn.call_args_list[1]
    assert "experience" in kwargs
    assert "trainable_state" in kwargs
    assert "target_state" in kwargs
