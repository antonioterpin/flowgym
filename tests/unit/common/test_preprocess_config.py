"""Unit tests for flowgym.common.preprocess (config-driven pipeline).

Covers end-to-end equivalence of config-constructed preprocessing pipelines
against direct function calls, invalid-param rejection, and JIT throughput.
"""

import random
import timeit

import jax
import pytest
from jax import numpy as jnp
from jax import random as jrandom

from flowgym.common.base import Estimator
from flowgym.common.preprocess import (
    background_suppression,
    clahe,
    crop_special,
    gaussian_blur,
    high_pass_filter,
    intensity_capping,
    intensity_clipping,
    resize_image,
    stretch_contrast,
)
from flowgym.make import compile_model
from flowgym.utils import load_configuration

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


class DummyEstimator(Estimator):
    def _estimate(
        self,
        image: jnp.ndarray,
        _,
        __,
        ___,
    ) -> tuple[jnp.ndarray, dict, dict]:
        return image, {}, {}


all_pre = [
    (resize_image, {"target_height": 64, "target_width": 64}),
    (resize_image, {"target_height": 128, "target_width": 128}),
    (resize_image, {"target_height": 256, "target_width": 256}),
    (
        crop_special,
        {
            "target_height": 16,
            "target_width": 16,
            "fraction_v": 0.5,
            "fraction_h": 0.5,
        },
    ),
    (background_suppression, {"threshold": 1}),
    (background_suppression, {"threshold": 5}),
    (stretch_contrast, {}),
    (stretch_contrast, {"scale": 255.0}),
    (intensity_capping, {"n": 2.0}),
    (intensity_capping, {"n": 1.0}),
    (intensity_capping, {"n": 3.0}),
    (intensity_clipping, {"n": 2.0}),
    (intensity_clipping, {"n": 1.0}),
    (intensity_clipping, {"n": 3.0}),
    (clahe, {"clip_limit": 2.0, "tile_grid_size": (8, 8), "nbins": 256}),
    (clahe, {"clip_limit": 0.01, "tile_grid_size": (3, 3), "nbins": 256}),
    (clahe, {"clip_limit": 0.1, "tile_grid_size": (2, 2), "nbins": 128}),
    (high_pass_filter, {"sigma": 1.0}),
    (high_pass_filter, {"sigma": 2.0, "truncate": 6.0}),
    (gaussian_blur, {"sigma": 1.0}),
]

bad_params = [
    (resize_image, {"target_height": -64, "target_width": 64}),
    (crop_special, {"target_height": 2045, "target_width": "2048"}),
    (background_suppression, {"threshold": "invalid_type"}),
    (background_suppression, {"threshold": -5}),
    (stretch_contrast, {"invalid_param": 123}),
    (intensity_capping, {"n": -1.0}),
    (intensity_clipping, {"n": "invalid_type"}),
    (clahe, {"clip_limit": -0.5, "tile_grid_size": (8, 8), "nbins": 256}),
    (clahe, {"clip_limit": 2.0, "tile_grid_size": (-3, 3), "nbins": 256}),
    (high_pass_filter, {"sigma": -1.0}),
    (high_pass_filter, {"sigma": 2.0, "truncate": "invalid_type"}),
    (stretch_contrast, {"scale": "invalid_type"}),
    (intensity_capping, {"n": None}),
    (intensity_clipping, {"n": []}),
    (clahe, {"clip_limit": 0.0, "tile_grid_size": (0, 0), "nbins": 256}),
    (clahe, {"clip_limit": 1.0, "tile_grid_size": (8, -8), "nbins": 128}),
    (high_pass_filter, {"sigma": 0}),
    (high_pass_filter, {"sigma": 3.0, "truncate": None}),
    (stretch_contrast, {"scale": -1}),
    (intensity_capping, {"n": "string_value"}),
    (intensity_clipping, {"n": {}}),
    (clahe, {"clip_limit": -1.0, "tile_grid_size": (4, 4), "nbins": 0}),
    (high_pass_filter, {"sigma": "string_value"}),
    (high_pass_filter, {"sigma": 1.0, "truncate": -5}),
    (gaussian_blur, {"sigma": 0}),
]


# Every step on its own, then a few multi-step pipelines (batched and not).
pipelines = [
    pytest.param(16, [idx], id=f"{fn.__name__}-{idx}")
    for idx, (fn, _) in enumerate(all_pre)
] + [
    pytest.param(
        B,
        random.Random(seed).choices(range(len(all_pre)), k=N),
        id=f"B{B}-N{N}-seed{seed}",
    )
    for B in (1, 16)
    for N, seed in ((5, 42), (15, 123))
]


def apply_from_idxs(idxs, image):
    """Apply preprocessing functions from given indices."""
    for idx in idxs:
        image = all_pre[idx][0](image, **all_pre[idx][1])
    return image


@pytest.mark.parametrize("B, idxs", pipelines)
def test_preprocess_config(B, idxs):
    """Config-driven pipeline matches direct function calls."""
    key, subkey = jrandom.split(jrandom.PRNGKey(0))

    # Create a random image
    image = jrandom.uniform(
        subkey, (B, 64, 64), minval=0, maxval=255, dtype=jnp.float32
    )

    # Apply preprocessing functions
    processed_image_reference = apply_from_idxs(idxs, image.copy())

    # Create a pre_processing configuration
    pre_processing_config = {
        "preprocessing_steps": [
            {"name": all_pre[idx][0].__name__} | all_pre[idx][1] for idx in idxs
        ]
    }

    # Instantiate the dummy estimator
    estimator = DummyEstimator.from_config(pre_processing_config)
    trainable_state = estimator.create_trainable_state(image, key)
    compiled = compile_model(estimator, processed_image_reference, False)
    create_state_fn = compiled.create_state
    compute_estimate_fn = compiled.compute_estimate
    # Create the state
    state = create_state_fn(image, key)
    state, _ = compute_estimate_fn(image, state, trainable_state)
    processed_image = state["estimates"][:, -1, ...]

    # Check if the processed image matches the reference
    assert jnp.allclose(
        processed_image, processed_image_reference, atol=1e-5
    ), (
        "Processed image does not match reference after applying "
        "preprocessing functions"
    )


@pytest.mark.parametrize(
    "fn, params",
    [
        pytest.param(fn, params, id=f"{fn.__name__}-{i}")
        for i, (fn, params) in enumerate(bad_params)
    ],
)
def test_preprocess_config_invalid_params(fn, params):
    """Each invalid preprocessing parameter raises ValueError or TypeError."""
    pre_processing_config = {
        "preprocessing_steps": [{"name": fn.__name__} | params]
    }
    with pytest.raises((ValueError, TypeError)):
        # Instantiate the dummy estimator
        DummyEstimator.from_config(pre_processing_config)


@pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()),
    reason="No GPU available.",
)
@pytest.mark.parametrize(
    "B, H, time_limit",
    [
        (1, 64, 8e-4),
        (64, 64, 8.1e-4),
        (1, 128, 8e-4),
        (64, 128, 8e-4),
        (1, 256, 8e-4),
        (64, 256, 1.1e-3),
        (1, 1024, 8e-4),
    ],
)
@pytest.mark.parametrize("seed", [42])
def test_preprocess_jit(B, H, time_limit, seed):
    """JIT-compiled preprocessing pipeline meets GPU throughput time limits."""
    # Sample idxs
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)

    # Create a random image
    image = jrandom.uniform(
        subkey, (B, H, H), minval=0, maxval=255, dtype=jnp.float32
    )

    # Create a pre_processing configuration
    pre_processing_config = {
        "preprocessing_steps": [
            {"name": "intensity_capping", "n": 2.0},
            {"name": "stretch_contrast"},
        ]
    }
    # Instantiate the dummy estimator
    estimator = DummyEstimator.from_config(pre_processing_config)
    trainable_state = estimator.create_trainable_state(image, key)
    compiled = compile_model(estimator, image, True)
    create_state_fn = compiled.create_state
    compute_estimate_fn = compiled.compute_estimate
    # Warm up
    state = create_state_fn(image, key)
    state, _ = compute_estimate_fn(image, state, trainable_state)

    def fn():
        state = create_state_fn(image, key)
        state, _ = compute_estimate_fn(image, state, trainable_state)
        processed_image = state["estimates"][:, -1, ...]
        processed_image.block_until_ready()

    runtime = timeit.repeat(
        fn,
        repeat=REPETITIONS,
        number=NUMBER_OF_EXECUTIONS,
    )
    runtime = min(runtime) / NUMBER_OF_EXECUTIONS

    assert runtime < time_limit, (
        f"JIT preprocessing took too long: {runtime:.6f} seconds per execution,"
        f" time limit: {time_limit:.6f} seconds"
    )
