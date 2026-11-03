"""Debugging tests for postprocessing mask semantics and metrics."""

import jax
import jax.numpy as jnp

from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.postprocess import (
    tile_average_interpolation,
    universal_median_test,
)


class DummyFlowEstimator(FlowFieldEstimator):
    """Simple estimator returning a deterministic flow from the input image."""

    def _estimate(
        self,
        image: jnp.ndarray,
        _,
        __,
        ___,
    ) -> tuple[jnp.ndarray, dict, dict]:
        return jnp.tile(image[..., None], (1, 1, 1, 2)), {}, {}


def test_validation_and_interpolation_mask_semantics_debug():
    """Ensure validation/interpolation share ``True=inlier`` semantics."""
    flow = jnp.zeros((1, 7, 7, 2), dtype=jnp.float32)
    flow = flow.at[0, 3, 3, 0].set(10.0)

    _, valid_mask, _ = universal_median_test(
        flow,
        r_threshold=2.0,
        epsilon=0.1,
        radius=1,
        valid=None,
        state=None,
    )
    assert valid_mask is not None
    assert not bool(valid_mask[0, 3, 3])

    # Current wiring: pass validation mask directly to interpolation
    # (`True` means inlier).
    flow_current, _, _ = tile_average_interpolation(
        flow, valid=valid_mask, radius=1, state=None
    )
    # Debug reference: invert mask before interpolation.
    flow_inverted, _, _ = tile_average_interpolation(
        flow, valid=~valid_mask, radius=1, state=None
    )

    center_mag_current = float(jnp.linalg.norm(flow_current[0, 3, 3]))
    center_mag_inverted = float(jnp.linalg.norm(flow_inverted[0, 3, 3]))

    # With correct semantics, outlier is replaced by neighbors (~0).
    assert center_mag_current < 1.0
    # With inverted semantics, outlier tends to survive.
    assert center_mag_inverted > 1.0


def test_flow_estimator_emits_rejected_percentage_metrics():
    """Ensure postprocessing emits rejection-percentage debug metrics."""
    model = DummyFlowEstimator(
        postprocessing_steps=[
            {"name": "universal_median_test", "r_threshold": 2.0, "radius": 1},
            {"name": "tile_average_interpolation", "radius": 1},
        ]
    )

    image = jnp.ones((2, 8, 8), dtype=jnp.float32)
    estimates = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    state = model.create_state(
        images=image,
        estimates=estimates,
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    trainable_state = model.create_trainable_state(image, jax.random.PRNGKey(0))

    _, metrics = model(image, state, trainable_state)

    rej_key = "postprocess_universal_median_test_0_rejected_percentage"
    outlier_key = "postprocess_tile_average_interpolation_1_outlier_percentage"
    mask_key = "postprocess_combined_rejected_mask"
    assert rej_key in metrics
    assert outlier_key in metrics
    assert mask_key in metrics

    rej = jnp.asarray(metrics[rej_key])
    outlier = jnp.asarray(metrics[outlier_key])
    rejected_mask = jnp.asarray(metrics[mask_key])
    assert rej.shape == (2,)
    assert outlier.shape == (2,)
    assert rejected_mask.shape == (2, 8, 8)
    assert rejected_mask.dtype == jnp.bool_
    assert jnp.all(rej >= 0.0)
    assert jnp.all(rej <= 100.0)
    assert jnp.all(outlier >= 0.0)
    assert jnp.all(outlier <= 100.0)
