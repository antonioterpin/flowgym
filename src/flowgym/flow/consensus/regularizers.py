"""Regularizers for refinement of flow field estimations."""

import jax.numpy as jnp
from flowgym.flow.utils import (
    compute_divergence,
    compute_gradients,
    compute_laplacian,
)


def smoothness_regularizer(flow: jnp.ndarray) -> jnp.ndarray:
    """Encourages spatial smoothness in flow field: sum of squared gradients.

    Args:
        flow (jnp.ndarray): Flow field of shape (H, W, 2)

    Returns:
        jnp.ndarray: Scalar loss from sum of squared gradients.
    """
    flow = flow[jnp.newaxis, ...]  # Add batch dimension
    dx, dy = compute_gradients(flow)  # both (B, H-2, W-2, 2)

    return jnp.sum(dx**2 + dy**2)  # sum removes batch dimension


def divergence_free_regularizer(flow: jnp.ndarray) -> jnp.ndarray:
    """Divergence-free regularization for flow field.

    Args:
        flow (jnp.ndarray): Flow field of shape (H, W, 2)

    Returns:
        jnp.ndarray: Scalar loss from sum of squared divergences.
    """
    flow = flow[jnp.newaxis, ...]  # Add batch dimension
    div = compute_divergence(flow)  # (B, H-2, W-2)
    return jnp.sum(div**2)  # sum removes batch dimension


def tv_regularizer(flow: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """Total Variation regularization for flow field.

    Encourages smoothness while preserving edges.

    Args:
        flow (jnp.ndarray): Flow field of shape (H, W, 2)
        eps (float): small constant to avoid division by zero

    Returns:
        jnp.ndarray: Scalar loss from sum of gradient norms
    """
    # Compute gradients in x and y directions
    flow = flow[jnp.newaxis, ...]  # Add batch dimension
    dx, dy = compute_gradients(flow)  # (B, H-2, W-2, 2)

    # Compute the norm of the gradients
    grad_norm = jnp.sqrt(dx**2 + dy**2 + eps)

    return jnp.sum(grad_norm)  # sum removes batch dimension


def laplacian_regularizer(flow: jnp.ndarray) -> jnp.ndarray:
    """Laplacian regularization for flow field.

    Penalizes flow curvature via Laplacian.

    Args:
        flow (jnp.ndarray): Flow field of shape (H, W, 2)

    Returns:
        jnp.ndarray: Scalar loss from sum of squared Laplacians.
    """
    flow = flow[jnp.newaxis, ...]  # Add batch dimension
    lap = compute_laplacian(flow)

    return jnp.sum(lap**2)  # sum removes batch dimension


REGULARIZERS = {
    "smoothness": smoothness_regularizer,
    "divergence": divergence_free_regularizer,
    "tv": tv_regularizer,
    "laplacian": laplacian_regularizer,
}


def total_regularization_loss(
    flow: jnp.ndarray, regularizers: list[str], weights: dict[str, float]
) -> float:
    """Composable regularization loss from multiple terms.

    Combines multiple regularization terms based on active flags and weights.

    Args:
        flow (jnp.ndarray): Flow field of shape (H, W, 2)
        regularizers (list[str]): List of active regularizer names.
        weights (dict[str, float]): Weights for each regularizer.

    Returns:
        jnp.ndarray: Total regularization loss as a scalar.
    """
    total = 0.0
    for name in regularizers:
        reg_fn = REGULARIZERS[name]
        total += weights[name] * reg_fn(flow)
    return total
