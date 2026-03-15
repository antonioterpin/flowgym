"""Integration tests for replay buffer in train_supervised."""

from unittest.mock import MagicMock

import jax.numpy as jnp

from train_supervised import train_supervised


def test_train_supervised_integration_with_replay():
    """Verify train_supervised uses the replay buffer."""
    # Mock model
    model = MagicMock()
    train_step_fn = MagicMock()

    # Return dummy values: loss, trainable_state, metrics
    train_step_fn.return_value = (
        jnp.array(1.0),
        MagicMock(),
        {"m": jnp.array(0.0)},
    )
    model.create_train_step.return_value = train_step_fn
    model.process_metrics.side_effect = lambda x: x
    # Mock prepare_experience_for_replay to just return the experience
    model.prepare_experience_for_replay.side_effect = lambda exp, state: exp

    # Mock sampler
    batch = MagicMock()
    # Use standard 4D shape (B, C, H, W) or 3D if that's what's expected
    # train_supervised seems to expect images of shape (B, H, W) or (B, C, H, W)
    # Replay uses batch size B
    B, H, W = 2, 4, 4
    batch.images1 = jnp.zeros((B, H, W))
    batch.images2 = jnp.zeros((B, H, W))
    batch.flow_fields = jnp.zeros((B, H, W, 2))
    batch.params = None

    sampler = MagicMock()
    sampler.__iter__.return_value = iter([batch])
    sampler.shutdown = MagicMock()

    # Call train_supervised
    train_supervised(
        model=model,
        model_config={"config": {"jit": False}},
        trainable_state=MagicMock(),
        out_dir="/tmp",
        # Use a real dict for state so it can be tree_mapped/stacked
        create_state_fn=lambda img, key: {"images": img},
        compute_estimate_fn=MagicMock(),
        sampler=sampler,
        num_batches=1,
        replay_buffer_capacity=10,
        replay_ratio=1.0,  # 1 replay step per batch
    )

    # train_step_fn called twice: once for real batch, once for replay
    assert train_step_fn.call_count == 2

    # Check arguments of the second call (replay)
    # kwarg 'experience' is expected
    _, kwargs = train_step_fn.call_args_list[1]
    assert "experience" in kwargs
    assert "trainable_state" in kwargs
    # Replay uses batch size B (which is 2 here)
    # The images in replay_exp.obs should be (B, H, W)
    assert kwargs["experience"].obs[0].shape == (B, H, W)
    assert kwargs["experience"].state["images"].shape == (B, H, W)
