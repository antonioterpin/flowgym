"""Module for data validation and outlier detection in optical flow estimation."""

from jax import lax
import jax.numpy as jnp
from flowgym.common.median import median
from flowgym.utils import DEBUG
from goggles.history.types import History


def constant_threshold_filter_validate_params(
    vel_min: float,
    vel_max: float,
):
    """Validate parameters for constant threshold filter.

    Args:
        vel_min: Minimum threshold for the magnitude.
        vel_max: Maximum threshold for the magnitude.

    Raises:
        ValueError: If vel_min is not less than vel_max.
    """
    if not isinstance(vel_min, (int, float)):
        raise ValueError(f"Invalid vel_min: {vel_min}. Must be a number.")
    if not isinstance(vel_max, (int, float)):
        raise ValueError(f"Invalid vel_max: {vel_max}. Must be a number.")
    if vel_min > vel_max:
        raise ValueError(f"Invalid thresholds: {vel_min} > {vel_max}.")


def constant_threshold_filter(
    flow_field: jnp.ndarray,
    vel_min: float,
    vel_max: float,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Mark outliers based on constant thresholds u_min and u_max.

    Args:
        flow_field: Input array of shape (B, H, W, 2)
        vel_min: Minimum threshold for the magnitude.
        vel_max: Maximum threshold for the magnitude.
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
            If provided, the outlier mask will be combined with this mask.
        state: Current state of the estimator.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    mag = jnp.linalg.norm(flow_field, axis=-1)
    valid = valid if valid is not None else jnp.ones(mag.shape, dtype=bool)
    return flow_field, valid & ((mag < vel_min) | (mag > vel_max)), state


def adaptive_global_filter_validate_params(n_sigma: float):
    """Validate parameters for adaptive global filter.

    Args:
        n_sigma: Number of standard deviations to use for thresholding.

    Raises:
        ValueError: If n_sigma is not a positive number.
    """
    if not isinstance(n_sigma, (int, float)):
        raise ValueError(f"Invalid n_sigma: {n_sigma}. Must be a number.")
    if n_sigma <= 0:
        raise ValueError(f"Invalid n_sigma: {n_sigma}. Must be positive.")


def adaptive_global_filter(
    flow_field: jnp.ndarray,
    n_sigma: float,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Thresholding based on mean and standard deviation of magnitudes.

    Args:
        flow_field: Input array of shape (B, H, W, 2).
        n_sigma: Number of standard deviations to use for thresholding.
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
            If provided, the outlier mask will be combined with this mask.
        state: Current state of the estimator.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    mag = jnp.linalg.norm(flow_field, axis=-1)
    mu = jnp.mean(mag)
    sigma = jnp.std(mag)
    lo = mu - n_sigma * sigma
    hi = mu + n_sigma * sigma
    valid = valid if valid is not None else jnp.ones(mag.shape, dtype=bool)
    return flow_field, ((mag < lo) | (mag > hi)) & valid, state


def adaptive_local_filter_validate_params(
    n_sigma: float,
    radius: int = 1,
):
    """Validate parameters for adaptive local filter.

    Args:
        n_sigma: Number of standard deviations to use for thresholding.
        radius: Radius for the local neighborhood
            (patch = (2*radius+1, 2*radius+1)).

    Raises:
        ValueError: If n_sigma is not a positive number or radius is negative.
    """
    if not isinstance(n_sigma, (int, float)):
        raise ValueError(f"Invalid n_sigma: {n_sigma}. Must be a number.")
    if n_sigma <= 0:
        raise ValueError(f"Invalid n_sigma: {n_sigma}. Must be positive.")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError(f"Invalid radius: {radius}. Must be a non-negative integer.")


def adaptive_local_filter(
    flow_field: jnp.ndarray,
    n_sigma: float,
    radius: int = 1,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Adaptive local thresholding based on mean and standard deviation.

    Args:
        flow_field: Input array of shape (B, H, W, 2).
        n_sigma: Number of standard deviations to use for thresholding.
        radius: Radius for the local neighborhood
            (patch = (2*radius+1, 2*radius+1)).
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
            If provided, the outlier mask will be combined with this mask.
        state: Current state of the estimator.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    if DEBUG:
        assert isinstance(flow_field, jnp.ndarray), "Flow field must be a jnp.ndarray."
        assert flow_field.ndim == 4, (
            "Flow field must be 4D (batch_size, height, width, channels)."
            + f" Got {flow_field.shape}."
        )
        assert radius >= 0, "Radius must be non-negative."
        assert n_sigma > 0, "n_sigma must be positive."

    B, H, W, _ = flow_field.shape
    magnitudes = jnp.linalg.norm(flow_field, axis=-1)
    WSIZE = 2 * radius + 1
    PATCH_PIX = WSIZE * WSIZE
    CENTER_FLAT = radius * WSIZE + radius

    patches = lax.conv_general_dilated_patches(
        magnitudes[..., None],
        filter_shape=(WSIZE, WSIZE),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    # Also discards the Channel dimension
    patches = patches.reshape(B, H, W, PATCH_PIX)

    # for each patch, compute the median and std
    means = jnp.mean(patches, axis=-1, keepdims=True)
    stds = jnp.std(patches, axis=-1, keepdims=True)
    centers = patches[..., CENTER_FLAT, None]

    # compute the threshold
    upper_bound = means + n_sigma * stds
    lower_bound = means - n_sigma * stds

    valid = valid if valid is not None else jnp.ones(magnitudes.shape, dtype=bool)

    # if outside the bounds, mark as outlier
    return (
        flow_field,
        jnp.squeeze((centers < lower_bound) | (centers > upper_bound), axis=-1) & valid,
        state,
    )


def universal_median_test_validate_params(
    r_threshold: float = 2.0,
    epsilon: float = 0.1,
    radius: int = 1,
):
    """Validate parameters for universal median test.

    Args:
        r_threshold: Threshold for the ratio of median to mean.
        epsilon: Small value to avoid division by zero.
        radius: Radius for the local neighborhood
            (patch = (2*radius+1, 2*radius+1)).

    Raises:
        ValueError: If r_threshold is not positive, epsilon is not positive,
            or radius is negative.
    """
    if not isinstance(r_threshold, (int, float)) or r_threshold <= 0:
        raise ValueError(f"Invalid r_threshold: {r_threshold}. Must be positive.")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError(f"Invalid epsilon: {epsilon}. Must be positive.")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError(f"Invalid radius: {radius}. Must be a non-negative integer.")


def universal_median_test(
    flow_field: jnp.ndarray,
    r_threshold: float = 2.0,
    epsilon: float = 0.1,
    radius: int = 1,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Universal outlier detection by median test.

    See https://link.springer.com/article/10.1007/s00348-005-0016-6

    Args:
        flow_field: Input array of shape (B, H, W, 2).
        r_threshold: Threshold for the ratio of median to mean.
        epsilon: Small value to avoid division by zero.
        radius: Radius for the local neighborhood
            (patch = (2*radius+1, 2*radius+1)).
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
            If provided, the outlier mask will be combined with this mask.
        state: Current state of the estimator.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    if DEBUG:
        assert isinstance(flow_field, jnp.ndarray), "Flow field must be a jnp.ndarray."
        assert flow_field.ndim == 4, (
            "Flow field must be 4D (batch_size, height, width, channels)."
            + f" Got {flow_field.shape}."
        )
        assert radius >= 0, "Radius must be non-negative."
        assert r_threshold > 0, "Threshold must be positive."
        assert epsilon > 0, "Epsilon must be positive."
    B, H, W, C = flow_field.shape
    WSIZE = 2 * radius + 1
    PATCH_PIX = WSIZE * WSIZE
    CENTER_FLAT = radius * WSIZE + radius

    patches = lax.conv_general_dilated_patches(
        flow_field,
        filter_shape=(WSIZE, WSIZE),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    patches = patches.reshape(B, H, W, C, PATCH_PIX)

    neighbours = jnp.delete(patches, CENTER_FLAT, axis=-1)

    med = median(neighbours)
    residuals = jnp.abs(neighbours - med[..., None])
    rm = median(residuals)
    r0 = jnp.abs(patches[..., CENTER_FLAT] - med) / (rm + epsilon)

    valid = valid if valid is not None else jnp.ones((B, H, W), dtype=bool)

    return (
        flow_field,
        jnp.any(r0 > r_threshold, axis=-1) & valid,
        state,
    )
