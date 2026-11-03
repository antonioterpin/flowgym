"""Integration tests for photometric augmentation in train_supervised."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from train_supervised import train_supervised

B, H, W = 4, 8, 8


def _run(**kwargs):
    """Run one step on constant frames; return the experience it was fed."""
    estimator = MagicMock()
    train_step_fn = MagicMock(
        return_value=(jnp.array(1.0), MagicMock(), {"m": jnp.array(0.0)})
    )
    estimator.create_train_step.return_value = train_step_fn
    estimator.checkpoint_metric.return_value = ("m", False)
    estimator.prepare_experience_for_training.side_effect = lambda exp, s: exp

    batch = MagicMock()
    batch.images1 = jnp.full((B, H, W), 100.0)
    batch.images2 = jnp.full((B, H, W), 100.0)
    batch.flow_fields = jnp.zeros((B, H, W, 2))
    batch.params = None
    sampler = MagicMock()
    sampler.__iter__.return_value = iter([batch])

    train_supervised(
        estimator=estimator,
        estimator_config={"config": {"jit": False}},
        trainable_state=MagicMock(),
        out_dir="/tmp",
        create_state_fn=lambda img, key: {"images": img},
        compute_estimate_fn=MagicMock(),
        sampler=sampler,
        num_batches=1,
        **kwargs,
    )
    return train_step_fn.call_args.kwargs["experience"]


def test_frames_are_untouched_by_default():
    experience = _run()
    assert jnp.all(experience.obs[0] == 100.0)


def test_photometric_augmentation_changes_only_the_observed_frames():
    experience = _run(photometric_augmentation={})
    prev, curr = experience.obs
    assert prev.shape == curr.shape == (B, H, W)
    assert not jnp.allclose(prev, 100.0)
    assert 0.0 <= float(prev.min()) and float(prev.max()) <= 255.0
    # Each sample gets its own random draw.
    assert not jnp.allclose(prev[0], prev[1])
    # The estimator state keeps the original frames.
    assert jnp.all(experience.state["images"] == 100.0)


def test_invalid_augmentation_params_fail_before_training():
    with pytest.raises(ValueError, match="gain_range"):
        _run(photometric_augmentation={"gain_range": (2.0, 0.5)})
