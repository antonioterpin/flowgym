"""Interpolation for optical-flow tensors with missing/outlier vectors."""

import jax.numpy as jnp
from jax import lax

from goggles.history.types import History


def tile_average_interpolation_validate_params(radius: int):
    """Validate parameters for tile average interpolation.

    Args:
        radius: Radius of the tile to average over.

    Raises:
        ValueError: If radius is not a positive integer.
    """
    if not isinstance(radius, int) or radius < 1:
        raise ValueError(f"Invalid radius: {radius}. Must be a positive integer.")


def tile_average_interpolation(
    flow: jnp.ndarray,
    valid: jnp.ndarray,
    radius: int,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Fill outliers with the average of their surrounding tile.

    Args:
        flow: Flow tensor of shape (B, H, W, C).
        valid: Mask of shape (B, H, W) where 1 means valid.
        radius: Radius of the tile to average over.
        state: Current state of the estimator.

    Returns:
        Interpolated flow tensor of the same shape as `flow`.
    """
    w = 2 * radius + 1
    window = (1, w, w, 1)
    strides = (1, 1, 1, 1)

    valid_f = valid[..., None].astype(flow.dtype)
    masked = flow * valid_f

    sum_valid = lax.reduce_window(valid_f, 0.0, lax.add, window, strides, "SAME")

    sum_flow = lax.reduce_window(masked, 0.0, lax.add, window, strides, "SAME")

    denom = jnp.maximum(sum_valid, 1.0)
    avg = sum_flow / denom

    return (
        jnp.where(valid_f, flow, avg),
        jnp.logical_or(valid_f, sum_valid > 0.1).squeeze(-1),
        state,
    )


def _jacobi_step(curr: jnp.ndarray, fixed: jnp.ndarray) -> jnp.ndarray:
    """One Jacobi relaxation step respecting *fixed* pixels (valid data).

    Args:
        curr: Current flow estimate of shape (B, H, W, C).
        fixed: Mask of shape (B, H, W, 1) where 1 means fixed.

    Returns:
        Updated flow estimate after one Jacobi step.
    """
    padded = jnp.pad(curr, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    neigh = (
        padded[:, :-2, 1:-1, :]
        + padded[:, 2:, 1:-1, :]
        + padded[:, 1:-1, :-2, :]
        + padded[:, 1:-1, 2:, :]
    ) * 0.25
    return jnp.where(fixed, curr, neigh)


def laplace_interpolation_validate_params(num_iter: int):
    """Validate parameters for Laplace interpolation.

    Args:
        num_iter: Number of iterations for Jacobi relaxation.

    Raises:
        ValueError: If num_iter is not a positive integer.
    """
    if not isinstance(num_iter, int) or num_iter < 1:
        raise ValueError(f"Invalid num_iter: {num_iter}. Must be a positive integer.")


def laplace_interpolation(
    flow: jnp.ndarray,
    valid: jnp.ndarray,
    num_iter: int = 512,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Solve Laplace equation with fixed boundary via Jacobi relaxation.

    Args:
        flow: Flow tensor of shape (B, H, W, C).
        valid: Mask of shape (B, H, W) where 1 means valid.
        num_iter: Number of Jacobi iterations to perform.
        state: Current state of the estimator.

    Returns:
        - Interpolated flow tensor of the same shape as `flow`.
        - Mask indicating valid pixels after interpolation.
    """
    curr = flow
    for _ in range(num_iter):
        curr = _jacobi_step(curr, valid[..., None])
    return curr, jnp.ones_like(valid).astype(jnp.bool_), state
