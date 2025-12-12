"""Data smoothing functions for optical flow estimation."""

import jax.numpy as jnp
from jax import lax
from flowgym.common.filters import gaussian_smoothing as gs
from flowgym.common.median import median
from goggles.history.types import History


def average_smoothing_validate_params(radius: int):
    """Validate parameters for average smoothing.

    Args:
        radius: Radius of the averaging window.

    Raises:
        ValueError: If radius is not a positive integer.
    """
    if not isinstance(radius, int) or radius < 1:
        raise ValueError(f"Invalid radius: {radius}. Must be a positive integer.")


def average_smoothing(
    flow: jnp.ndarray,
    radius: int,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Uniform average smoothing (channel-wise).

    Args:
        flow: Input tensor of shape (B, H, W, 2).
        radius: Radius of the averaging window.
            Window size will be 2 * radius + 1.
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
        state: Current state of the estimator.

    Returns:
        Smoothed flow with the same shape as the input.
    """
    k = 2 * radius + 1
    pad = k // 2

    window_shape = (1, k, k, 1)
    summed = lax.reduce_window(
        jnp.pad(flow, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge"),
        0.0,
        lax.add,
        window_shape,
        window_strides=(1, 1, 1, 1),
        padding="VALID",
    )
    return summed / (k * k), valid, state


def median_smoothing_validate_params(radius: int):
    """Validate parameters for median smoothing.

    Args:
        radius: Radius of the median filter.

    Raises:
        ValueError: If radius is not a positive integer.
    """
    if not isinstance(radius, int) or radius < 1:
        raise ValueError(f"Invalid radius: {radius}. Must be a positive integer.")


def median_smoothing(
    flow: jnp.ndarray,
    radius: int = 3,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Sliding-window median filter (channel-wise).

    Args:
        flow: Input tensor of shape (B, H, W, 2).
        radius: Radius of the median filter.
            Window size will be 2 * radius + 1.
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
        state: Current state of the estimator.

    Returns:
        Smoothed flow with the same shape as the input.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    B, H, W, C = flow.shape
    WSIZE = 2 * radius + 1
    PATCH_PIX = WSIZE * WSIZE

    patches = lax.conv_general_dilated_patches(
        flow,
        filter_shape=(WSIZE, WSIZE),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    patches = patches.reshape(B, H, W, C, PATCH_PIX)
    # Compute the median along the last dimension (patch pixels)
    return median(patches).reshape(B, H, W, C), valid, state


def gaussian_smoothing_validate_params(sigma: float, truncate: float = 4.0):
    """Validate parameters for Gaussian smoothing.

    Args:
        sigma: Standard deviation of the Gaussian kernel.
        truncate: Truncate the kernel at this many standard deviations.

    Raises:
        ValueError: If sigma is not positive or if truncate is not positive.
    """
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError(f"Invalid sigma: {sigma}. Must be a positive number.")
    if not isinstance(truncate, (int, float)) or truncate <= 0:
        raise ValueError(f"Invalid truncate: {truncate}. Must be a positive number.")


def gaussian_smoothing(
    flow: jnp.ndarray,
    sigma: float,
    truncate: float = 4.0,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
    mode: str = "same",
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Gaussian smoothing (channel-wise).

    Args:
        flow: Input tensor of shape (B, H, W, 2).
        sigma: Standard deviation of the Gaussian kernel.
        truncate: Truncate the kernel at this many standard deviations.
        valid: Optional mask of shape (B, H, W) where 1 means valid.
        mode: Padding mode, either "same" or "reflect".
        state: Current state of the estimator.

    Returns:
        Smoothed flow with the same shape as the input.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    return (
        gs(
            flow,
            sigma=sigma,
            truncate=truncate,
            mode=mode,
        ),
        valid,
        state,
    )
