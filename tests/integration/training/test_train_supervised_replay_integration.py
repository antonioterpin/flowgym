"""Integration tests for train_supervised replay-buffer pipeline.

Verifies that the replay buffer is populated and sampled during training.
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
from synthpix import SynthpixBatch

from flowgym.flow.dummy import DummyEstimator
from train_supervised import train_supervised


@patch("train_supervised.Checkpointer")
def test_train_supervised_integration_with_replay(
    mock_checkpointer, tmp_path, dummy_trainable_state
):
    """train_supervised issues one real step and one replay step per batch."""
    estimator = DummyEstimator(train_type="supervised")
    train_step_fn = MagicMock(wraps=estimator.create_train_step())
    estimator.create_train_step = MagicMock(return_value=train_step_fn)

    # Mock sampler
    B, H, W = 2, 4, 4
    batch = SynthpixBatch(
        images1=jnp.zeros((B, H, W)),
        images2=jnp.zeros((B, H, W)),
        flow_fields=jnp.zeros((B, H, W, 2)),
    )

    sampler = MagicMock()
    sampler.__iter__.return_value = iter([batch])
    sampler.shutdown = MagicMock()

    # Call train_supervised
    train_supervised(
        estimator=estimator,
        estimator_config={"config": {"jit": False}},
        trainable_state=dummy_trainable_state,
        out_dir=str(tmp_path),
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
