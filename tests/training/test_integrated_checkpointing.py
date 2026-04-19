"""Tests of integration between flowgym and synthpix."""

import pathlib

import jax.numpy as jnp
import numpy as np
import optax
import synthpix
from flax.core import FrozenDict

from flowgym.common.base.trainable_state import NNEstimatorTrainableState
from flowgym.make import load_model, save_model


def test_real_integration_checkpointing(tmp_path):
    """Test integration between flowgym.save_model and synthpix.make.

    Uses SyntheticImageSampler with real Grain-based scheduler for
    checkpointing.
    """
    # 1. Prepare dummy data for NumpyDataSource
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # NumpyDataSource expects flow_*.npy files
    flow_file = data_dir / "flow_001.npy"
    np.save(flow_file, np.random.rand(128, 128, 2).astype(np.float32))

    # 2. Setup SynthPix config
    dataset_config = {
        "seed": 0,
        "batch_size": 1,
        "image_shape": [64, 64],
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": 1,
        "scheduler_class": ".npy",
        "file_list": [str(flow_file)],
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

    # 3. Create real sampler
    sampler = synthpix.make(dataset_config, use_grain_scheduler=True)
    assert sampler.grain_iterator is not None, (
        "Sampler should have a grain iterator"
    )

    # Burn a few steps to move the state
    next(sampler)
    next(sampler)

    # Get reference batches BEFORE saving (what sampler would produce next)
    reference_batches = [next(sampler) for _ in range(3)]

    # Reset to the state after burn-in by recreating and burning again
    sampler = synthpix.make(dataset_config, use_grain_scheduler=True)
    next(sampler)
    next(sampler)

    # 4. Prepare Mock Model State
    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(2.0, jnp.float32)})
    tx = optax.adam(0.1)
    state = NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )

    # 5. SAVE ATOMIC
    model_name = "IntegrationTest"
    save_path_str = save_model(
        state=state,
        out_dir=tmp_path,
        step=10,
        model_name=model_name,
        sampler=sampler,
    )
    save_path = pathlib.Path(save_path_str)
    assert save_path.exists()

    # 6. RESTORE SAMPLER via synthpix.make
    restored_sampler = synthpix.make(
        dataset_config, use_grain_scheduler=True, load_from=save_path.parent
    )

    # 7. Verify restored sampler produces identical batches
    print("\nComparing batch outputs:")
    for i, ref_batch in enumerate(reference_batches):
        restored_batch = next(restored_sampler)

        # Compare flow fields
        np.testing.assert_allclose(
            np.array(restored_batch.flow_fields),
            np.array(ref_batch.flow_fields),
            rtol=1e-5,
            err_msg=f"Batch {i} flow_fields mismatch",
        )

        # Compare images
        np.testing.assert_allclose(
            np.array(restored_batch.images1),
            np.array(ref_batch.images1),
            rtol=1e-5,
            err_msg=f"Batch {i} images1 mismatch",
        )
        np.testing.assert_allclose(
            np.array(restored_batch.images2),
            np.array(ref_batch.images2),
            rtol=1e-5,
            err_msg=f"Batch {i} images2 mismatch",
        )
        print(f"  Batch {i}: ✓ identical")

    # 8. RESTORE MODEL via load_model (Partial Restoring)
    template_state = NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )
    restored_model = load_model(save_path, template_state, mode="resume")
    assert restored_model.step == 10
    assert jnp.allclose(restored_model.params["w"], state.params["w"])
