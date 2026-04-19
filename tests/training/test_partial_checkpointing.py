"""Integration tests for checkpoint/resume with flowgym.save_model and synthpix.

Tests real save/restore cycles for SyntheticImageSampler and RealImageSampler,
validating sampler state is preserved across checkpoint boundaries.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import synthpix
from flax.core import FrozenDict

from flowgym.common.base.trainable_state import NNEstimatorTrainableState
from flowgym.make import load_model, save_model


def _make_synthetic_config(file_list: list[str]) -> dict:
    """Create a SyntheticImageSampler configuration using .npy flow fields."""
    return {
        "seed": 42,
        "batch_size": 1,
        "image_shape": [64, 64],
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "scheduler_class": ".npy",
        "file_list": file_list,
        "episode_length": 0,
        "randomize": False,
        "loop": True,
        "output_units": "pixels",
        "resolution": 1.0,
        "velocities_per_pixel": 1.0,
        "flow_field_size": [128, 128],
        "max_speed_x": 1.0,
        "max_speed_y": 1.0,
        "min_speed_x": 0.0,
        "min_speed_y": 0.0,
        "p_hide_img1": 0.0,
        "p_hide_img2": 0.0,
        "diameter_ranges": [[1, 2]],
        "diameter_var": 0.1,
        "intensity_ranges": [[200, 255]],
        "intensity_var": 10.0,
        "rho_ranges": [[0.5, 0.5]],
        "rho_var": 0.0,
        "dt": 1.0,
        "img_offset": [2, 2],
        "seeding_density_range": [0.01, 0.01],
        "noise_uniform": 0.0,
        "noise_gaussian_mean": 0.0,
        "noise_gaussian_std": 0.0,
    }


def _make_real_config(file_list: list[str], dims: dict) -> dict:
    """Create a RealImageSampler configuration using .mat files with images."""
    return {
        "seed": 42,
        "batch_size": 1,
        "image_shape": [dims["height"], dims["width"]],
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "scheduler_class": ".mat",
        "file_list": file_list,
        "episode_length": 0,
        "randomize": False,
        "loop": True,
        "include_images": True,  # Key difference: triggers RealImageSampler
    }


def _make_model_state() -> NNEstimatorTrainableState:
    """Create a minimal trainable state for checkpointing."""

    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(2.0, jnp.float32)})
    tx = optax.adam(0.1)
    return NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )


def _compare_sampler_states(
    original: dict, restored: dict, skip_keys: set[str] | None = None
) -> None:
    """Compare two sampler state dictionaries, handling JAX and NumPy arrays.

    Args:
        original: Original sampler state before save.
        restored: Restored sampler state after load.
        skip_keys: Keys to skip comparison
    """
    skip_keys = skip_keys or set()

    for key, orig_val in original.items():
        if key in skip_keys:
            continue

        rest_val = restored.get(key)

        if orig_val is None:
            assert rest_val is None, (
                f"Key '{key}' mismatch: expected None, got {type(rest_val)}"
            )
        elif rest_val is None:
            pytest.fail(
                f"Key '{key}' mismatch: expected {type(orig_val)}, got None"
            )
        elif isinstance(
            orig_val, (np.ndarray, jnp.ndarray, jax.ShapeDtypeStruct)
        ):
            np.testing.assert_allclose(
                np.array(rest_val),
                np.array(orig_val),
                err_msg=f"Key '{key}' array mismatch",
            )
        elif isinstance(orig_val, dict):
            # Recursive comparison for nested dicts
            _compare_sampler_states(orig_val, rest_val, skip_keys)
        else:
            assert rest_val == orig_val, (
                f"Key '{key}' mismatch: expected {orig_val}, got {rest_val}"
            )


def test_synthetic_sampler_checkpoint_integration(tmp_path, npy_flow_files):
    """Test checkpoint/restore cycle for SyntheticImageSampler.

    Verifies that:
    1. flowgym.save_model correctly saves sampler state alongside model state
    2. synthpix.make(load_from=...) correctly restores sampler state
    3. The restored sampler produces identical outputs to original
    """
    config = _make_synthetic_config(npy_flow_files)

    # 1. Create sampler via synthpix.make
    sampler = synthpix.make(config, use_grain_scheduler=True)
    restored_sampler = None

    try:
        assert sampler.grain_iterator is not None, (
            "Sampler should have a grain iterator"
        )

        # 2. Advance state by iterating before checkpoint
        next(sampler)
        next(sampler)

        # 3. Create model state and save checkpoint
        model_state = _make_model_state()
        model_name = "CheckpointTest_synthetic"

        save_path_str = save_model(
            state=model_state,
            out_dir=tmp_path,
            step=10,
            model_name=model_name,
            sampler=sampler,
        )
        save_path = pathlib.Path(save_path_str)
        assert save_path.exists(), f"Checkpoint path should exist: {save_path}"

        # 4. Get next batch from original sampler (post-checkpoint)
        original_batch = next(sampler)
        original_state = sampler.state

        # 5. Restore sampler via synthpix.make with load_from
        restored_sampler = synthpix.make(
            config, use_grain_scheduler=True, load_from=save_path.parent
        )

        # 6. Get next batch from restored sampler - should match original
        restored_batch = next(restored_sampler)
        restored_state = restored_sampler.state

        # 7. Verify sampler states match after iteration (includes jax_seeds)
        _compare_sampler_states(original_state, restored_state)

        # 8. Verify batch outputs match
        assert jnp.allclose(
            original_batch.flow_fields,
            restored_batch.flow_fields,
        ), "Flow fields should match"

        # 9. Verify model state can also be restored independently
        template_state = _make_model_state()
        restored_model = load_model(save_path, template_state, mode="resume")
        assert restored_model.step == 10, "Model step should be restored"
        assert jnp.allclose(
            restored_model.params["w"], model_state.params["w"]
        ), "Model params should be restored"
    finally:
        sampler.shutdown()
        if restored_sampler is not None:
            restored_sampler.shutdown()


def test_real_sampler_checkpoint_integration(tmp_path, mock_mat_files):
    """Test checkpoint/restore cycle for RealImageSampler.

    Verifies that:
    1. flowgym.save_model correctly saves sampler state alongside model state
    2. synthpix.make(load_from=...) correctly restores sampler state
    3. The restored sampler produces identical outputs to original
    """
    file_list, dims = mock_mat_files
    config = _make_real_config(file_list, dims)

    # 1. Create sampler via synthpix.make
    sampler = synthpix.make(config, use_grain_scheduler=True)
    restored_sampler = None

    try:
        assert sampler.grain_iterator is not None, (
            "Sampler should have a grain iterator"
        )

        # 2. Advance state by iterating before checkpoint
        next(sampler)
        next(sampler)

        # 3. Create model state and save checkpoint
        model_state = _make_model_state()
        model_name = "CheckpointTest_real"

        save_path_str = save_model(
            state=model_state,
            out_dir=tmp_path,
            step=10,
            model_name=model_name,
            sampler=sampler,
        )
        save_path = pathlib.Path(save_path_str)
        assert save_path.exists(), f"Checkpoint path should exist: {save_path}"

        # 4. Get next batch from original sampler (post-checkpoint)
        original_batch = next(sampler)
        original_state = sampler.state

        # 5. Restore sampler via synthpix.make with load_from
        restored_sampler = synthpix.make(
            config, use_grain_scheduler=True, load_from=save_path.parent
        )

        # 6. Get next batch from restored sampler - should match original
        restored_batch = next(restored_sampler)
        restored_state = restored_sampler.state

        # 7. Verify sampler states match after iteration
        _compare_sampler_states(original_state, restored_state)

        # 8. Verify batch outputs match
        assert jnp.allclose(
            original_batch.flow_fields,
            restored_batch.flow_fields,
        ), "Flow fields should match"

        # 9. Verify model state can also be restored independently
        template_state = _make_model_state()
        restored_model = load_model(save_path, template_state, mode="resume")
        assert restored_model.step == 10, "Model step should be restored"
        assert jnp.allclose(
            restored_model.params["w"], model_state.params["w"]
        ), "Model params should be restored"
    finally:
        sampler.shutdown()
        if restored_sampler is not None:
            restored_sampler.shutdown()
