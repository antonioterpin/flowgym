"""Test the flow field computation using synthetic images."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.flow.dummy import DummyEstimator


@pytest.fixture(scope="session")
def estimator():
    """Create a FlowFieldEstimator instance for testing."""
    estimator = DummyEstimator()
    return estimator


@pytest.mark.parametrize("image_shape", [(128, 64), (64, 64)])
@pytest.mark.parametrize("history_length", [1, 2, 3])
def test_estimation_create_state(estimator, image_shape, history_length):
    """Test the creation of the estimator state."""
    # Create a synthetic image
    key = jax.random.PRNGKey(0)

    img = jax.random.uniform(
        key, shape=image_shape, minval=0, maxval=255, dtype=jnp.float32
    )
    # Add a batch dimension
    img = jnp.expand_dims(img, axis=0)

    # Create the estimator state
    state = estimator.create_state(
        img, image_history_size=history_length, estimates=jnp.zeros(img.shape + (2,))
    )

    # Check that the state has the expected shape
    assert (
        state["images"].shape
        == (
            img.shape[0],
            history_length,
        )
        + img.shape[1:]
    )
    assert state["estimates"].shape == (
        img.shape[0],
        history_length,
    ) + img.shape[
        1:
    ] + (2,)
