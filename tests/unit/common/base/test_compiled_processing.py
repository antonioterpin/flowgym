"""Standalone processing must match the estimator's configured pipeline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym import make
from flowgym.flow.dummy import DummyEstimator


@pytest.mark.parametrize("jit", [False, True])
def test_compiled_preprocess_matches_initialization_and_estimation(jit):
    """Export preprocessing without running the estimator or storing extras."""
    model = DummyEstimator(
        preprocessing_steps=[
            {
                "name": "resize_image",
                "target_height": 4,
                "target_width": 6,
            }
        ]
    )
    preprocess = make.compile_preprocess(model, jit=jit)
    images = jnp.arange(128, dtype=jnp.float32).reshape(2, 8, 8)
    state = model.create_state(
        images,
        jnp.zeros((2, 4, 6, 2)),
        image_history_size=1,
        rng=jax.random.PRNGKey(0),
    )
    processed = preprocess(images)
    np.testing.assert_allclose(processed, state["images"][:, -1])
    trained = model.create_trainable_state(images, jax.random.PRNGKey(0))
    updated, _ = model(images, state, trained)
    np.testing.assert_allclose(processed, updated["images"][:, -1])


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "steps",
    [
        [],
        [{"name": "temporal_smoothing_ema", "alpha": 0.5}],
        [{"name": "constant_threshold_filter", "vel_min": 1.0, "vel_max": 2.0}],
        [
            {
                "name": "resize_flow",
                "target_height": 4,
                "target_width": 6,
            }
        ],
    ],
)
def test_compiled_postprocess_matches_full_estimator(jit, steps):
    """Standalone postprocessing preserves the flow and diagnostic metrics."""
    model = DummyEstimator(postprocessing_steps=steps)
    postprocess = make.compile_postprocess(model, jit=jit)
    images = jnp.ones((2, 8, 8))
    resize = any(step["name"] == "resize_flow" for step in steps)
    shape = (2, 4, 6, 2) if resize else (2, 8, 8, 2)
    state = model.create_state(
        images,
        jnp.ones(shape),
        image_history_size=1,
        rng=jax.random.PRNGKey(0),
    )
    trained = model.create_trainable_state(images, jax.random.PRNGKey(0))
    processed, metrics = postprocess(
        jnp.zeros((2, 8, 8, 2)), images, state, trained
    )
    updated, expected_metrics = model(images, state, trained)
    np.testing.assert_allclose(processed, updated["estimates"][:, -1])
    assert metrics.keys() == expected_metrics.keys(), "Metrics must match"
    for key, value in metrics.items():
        np.testing.assert_allclose(value, expected_metrics[key])
