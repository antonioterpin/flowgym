"""Unit tests for flowgym.density.simple.

Covers SimpleDensityEstimator: threshold validation, pixel-fraction
density on known arrays, and accuracy on synthetic particle images.
"""

import jax
import jax.numpy as jnp
import pytest
from synthpix.data_generate import (
    ImageGenerationSpecification,
    generate_images_from_flow,
)

from flowgym.density.simple import SimpleDensityEstimator


@pytest.mark.parametrize("threshold", [-1.0, 260.0])
def test_invalid_threshold(threshold):
    """Out-of-range threshold raises ValueError on construction."""
    with pytest.raises(ValueError):
        SimpleDensityEstimator(threshold)


@pytest.mark.parametrize(
    "image, threshold, expected",
    [
        (jnp.array([[[0, 1], [2, 3]]], dtype=jnp.float32), 1.0, 0.5),
        (jnp.array([[[0, 1], [2, 3]]], dtype=jnp.float32), 2.0, 0.25),
        (jnp.array([[[0, 1], [2, 3]]], dtype=jnp.float32), 3.0, 0.0),
    ],
)
def test_particles_per_pixel(image, threshold, expected):
    """Estimated density equals the fraction of pixels above threshold."""
    config = {
        "threshold": threshold,
    }
    estimator = SimpleDensityEstimator.from_config(config)
    state = estimator.create_state(
        image, estimates=jnp.zeros((image.shape[0], 1)), image_history_size=1
    )
    state, _ = estimator(image, state, None)
    estimated_density = state["estimates"][:, -1]
    # Call the function and check the result
    assert jnp.isclose(estimated_density, expected, atol=1e-5)


@pytest.mark.parametrize(
    "image_shape",
    [
        (256, 256),
    ],
)
@pytest.mark.parametrize("density", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("intensity_range", [(50, 200), (70, 200)])
def test_density_simple_synthetic_image(image_shape, density, intensity_range):
    """Density estimate is within 50 % of the true value."""
    key = jax.random.PRNGKey(0)
    threshold = intensity_range[0] / 2.5

    image_generation_parameters = ImageGenerationSpecification(
        batch_size=1,
        image_shape=image_shape,
        img_offset=(0, 0),
        intensity_ranges=[intensity_range],
        seeding_density_range=(density, density),
    )
    # Generate a synthetic image with a given density
    image1, _, _ = generate_images_from_flow(
        key=key,
        flow_field=jnp.zeros((1, *image_shape, 2)),
        parameters=image_generation_parameters,
        position_bounds=image_shape,
    )
    img = image1
    config = {
        "threshold": threshold,
    }
    estimator = SimpleDensityEstimator.from_config(config)
    state = estimator.create_state(
        img, estimates=jnp.zeros((img.shape[0], 1)), image_history_size=1
    )
    state, _ = estimator(img, state, None)
    estimated_density = state["estimates"][:, -1]
    # Call the function and check the result
    assert jnp.isclose((estimated_density - density) / density, 0, atol=5e-1), (
        f"Estimated density {estimated_density} does not match expected "
        f"density {density}."
    )
