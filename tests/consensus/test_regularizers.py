import pytest
import jax.numpy as jnp

from flowgym.flow.consensus.regularizers import (
    smoothness_regularizer,
    divergence_free_regularizer,
    tv_regularizer,
    laplacian_regularizer,
)


@pytest.mark.parametrize(
    "batch_shape",
    [
        (1, 3, 3, 2),
        (2, 5, 6, 2),
        (4, 10, 10, 2),
    ],
)
def test_smoothness_regularizer_valid(batch_shape):
    """Test smoothness regularizer with valid flow shapes."""
    flow = jnp.ones(batch_shape)
    result = smoothness_regularizer(flow)
    assert jnp.isscalar(result)
    assert result == 0


@pytest.mark.parametrize(
    "batch_shape, expected",
    [
        ((3, 3, 2), 2),
    ],
)
def test_smoothness_regularizer_ramp(batch_shape, expected):
    """Test smoothness regularizer returns the expected value for a ramp flow."""
    _, W, _ = batch_shape
    # Ramp along width, all batches and channels identical
    ramp = jnp.arange(W).reshape(1, W, 1)
    flow = jnp.broadcast_to(ramp, batch_shape)
    result = smoothness_regularizer(flow)
    assert jnp.isscalar(result)
    assert result == expected


@pytest.mark.parametrize(
    "batch_shape",
    [
        (3, 3, 2),
        (5, 6, 2),
    ],
)
def test_divergence_free_regularizer_zero(batch_shape):
    """Should return 0 for a zero flow (trivially divergence-free)."""
    flow = jnp.zeros(batch_shape)
    result = divergence_free_regularizer(flow)
    assert jnp.isscalar(result)
    assert result == 0


@pytest.mark.parametrize(
    "batch_shape",
    [
        (3, 3, 2),
        (4, 4, 2),
    ],
)
def test_divergence_free_regularizer_ramp(batch_shape):
    """Test divergence regularizer on a known divergent flow (ramp along x)."""
    H, W, _ = batch_shape
    # u_x increases along x, u_y = 0. So div = du_x/dx = 1 at all interior points.
    ramp = jnp.arange(W).reshape(1, W, 1)
    ramp = jnp.broadcast_to(ramp, (H, W, 1))
    zeros = jnp.zeros((H, W, 1))
    flow = jnp.concatenate([ramp, zeros], axis=-1)  # shape (H,W,2)
    result = divergence_free_regularizer(flow)
    # For each interior pixel, divergence is 1, so squared is 1.
    # Number of interior points: B*(H-2)*(W-2)
    expected = (H - 2) * (W - 2)
    assert jnp.isscalar(result)
    assert result == expected


@pytest.mark.parametrize(
    "batch_shape",
    [
        (3, 3, 2),
        (5, 5, 2),
    ],
)
def test_tv_regularizer_zero(batch_shape):
    """TV of a constant flow field should be sum(sqrt(eps)) at all locations."""
    flow = jnp.ones(batch_shape)
    eps = 1e-4
    result = tv_regularizer(flow, eps=eps)
    # Each dx, dy is 0 everywhere, so grad_norm = sqrt(eps)
    H, W, C = batch_shape
    N = (H - 2) * (W - 2) * C  # Number of points in output
    expected = N * jnp.sqrt(eps)
    assert jnp.isclose(result, expected, atol=1e-8)


@pytest.mark.parametrize(
    "batch_shape",
    [
        (3, 3, 2),
        (4, 4, 2),
    ],
)
def test_tv_regularizer_ramp(batch_shape):
    """TV of a ramp flow: only one gradient direction is nonzero (dx=1, dy=0)."""
    H, W, C = batch_shape
    eps = 1e-4
    ramp = jnp.arange(W).reshape(1, W, 1)
    flow = jnp.broadcast_to(ramp, batch_shape)
    result = tv_regularizer(flow, eps=eps)
    # dx = 1 everywhere, dy = 0, so grad_norm = sqrt(1^2 + 0^2 + eps)
    grad_norm = jnp.sqrt(1.0 + eps)
    N = (H - 2) * (W - 2) * C
    expected = N * grad_norm
    assert jnp.isclose(result, expected, atol=1e-8)


@pytest.mark.parametrize(
    "batch_shape",
    [
        (1, 4, 4, 2),
        (2, 5, 5, 2),
    ],
)
def test_laplacian_regularizer_zero(batch_shape):
    """Flat flow field: Laplacian should be zero everywhere, loss = 0."""
    flow = jnp.ones(batch_shape)
    result = laplacian_regularizer(flow)
    assert jnp.isscalar(result)
    assert result == 0


@pytest.mark.parametrize(
    "batch_shape",
    [
        (1, 4, 4, 2),
        (2, 5, 5, 2),
    ],
)
def test_laplacian_regularizer_ramp(batch_shape):
    """Test laplacian_regularizer on a ramp flow field."""
    flow = jnp.arange(batch_shape[2]).reshape(1, 1, batch_shape[2], 1)
    flow = jnp.broadcast_to(flow, batch_shape)
    result = laplacian_regularizer(flow)
    # The Laplacian of a ramp is zero everywhere in the interior.
    expected = 0
    assert jnp.isscalar(result)
    assert result == expected


@pytest.mark.parametrize(
    "batch_shape",
    [
        (4, 4, 2),  # Will yield (1,0,0,2) for laplacian, sum is zero
        (5, 5, 2),  # Will yield (1,1,1,2)
    ],
)
def test_laplacian_regularizer_quadratic(batch_shape):
    """Test laplacian_regularizer output shape and value for quadratic input."""
    H, W, _ = batch_shape
    x_coords = jnp.arange(W)
    quad = (x_coords**2).reshape(1, W, 1)
    quad = jnp.broadcast_to(quad, (H, W, 1))
    zeros = jnp.zeros((H, W, 1))
    flow = jnp.concatenate([quad, zeros], axis=-1)
    result = laplacian_regularizer(flow)
    # The Laplacian shape is (B, H-4, W-4)
    N = max(H - 2, 0) * max(W - 2, 0)  # only the u channel is nonzero
    if N == 0:
        expected = 0
    else:
        expected = N * (2**2)  # Laplacian is 2 everywhere in the interior
    assert jnp.isscalar(result)
    assert result == expected
