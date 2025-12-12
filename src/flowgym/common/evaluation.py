"""Evaluation module for flow field estimation."""

import jax.numpy as jnp
from goggles.types import Metrics

from flowgym.flow.utils import compute_divergence_and_vorticity
from flowgym.flow.process import apply_flow_to_image_forward


def loss_supervised_density(
    density: jnp.ndarray, density_gt: jnp.ndarray
) -> jnp.ndarray:
    """Compute the density estimation error.

    Args:
        density: Estimated density field.
        density_gt: Ground truth density field.

    Returns:
        Computed density error at each pixel.
    """
    return jnp.mean(jnp.square(density - density_gt) / density_gt)


def angle_error(flow_field: jnp.ndarray, flow_field_gt: jnp.ndarray) -> jnp.ndarray:
    """Compute the angle error between two flow fields.

    Args:
        flow_field: Estimated flow field.
        flow_field_gt: Ground truth flow field.

    Returns:
        Computed angle error at each pixel.
    """
    # Normalize the flow fields
    norm_flow_field = jnp.linalg.norm(flow_field, axis=-1)
    norm_flow_field_gt = jnp.linalg.norm(flow_field_gt, axis=-1)

    # Avoid division by zero
    norm = jnp.clip(norm_flow_field * norm_flow_field_gt, 1e-3, None)

    # Compute the cosine of the angle between the two flow fields
    cos_angle = jnp.sum(flow_field * flow_field_gt, axis=-1) / norm

    # Clip the cosine values to avoid numerical issues
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)

    # Compute the angle error in radians
    angle_error = jnp.arccos(cos_angle)

    # Normalize the angle error to be in the range [0, 1]
    return angle_error / jnp.pi


def relative_error(flow_field: jnp.ndarray, flow_field_gt: jnp.ndarray) -> jnp.ndarray:
    """Compute the relative error between two flow fields.

    Args:
        flow_field: Estimated flow field.
        flow_field_gt: Ground truth flow field.

    Returns:
        Computed relative error at each pixel.
    """
    # Compute the norm of the flow fields
    norm_flow_field = jnp.linalg.norm(flow_field, axis=-1, keepdims=True)
    norm_flow_field_gt = jnp.linalg.norm(flow_field_gt, axis=-1, keepdims=True)

    # Avoid division by zero
    norm = jnp.clip(norm_flow_field_gt, 1e-10, None)

    # Compute the relative error
    relative_error = jnp.abs(norm_flow_field - norm_flow_field_gt) / norm

    return relative_error


def absolute_error(flow_field: jnp.ndarray, flow_field_gt: jnp.ndarray) -> jnp.ndarray:
    """Compute the absolute error between two flow fields.

    Args:
        flow_field: Estimated flow field.
        flow_field_gt: Ground truth flow field.

    Returns:
        Computed absolute error at each pixel.
    """
    # Compute the absolute error
    absolute_error = jnp.linalg.norm(flow_field - flow_field_gt, axis=-1)

    return absolute_error


def loss_supervised(flow_field: jnp.ndarray, flow_field_gt: jnp.ndarray) -> jnp.ndarray:
    """Compute the loss for supervised training.

    Args:
        flow_field: Estimated flow field.
        flow_field_gt: Ground truth flow field.

    Returns:
        Computed loss.
    """
    error = jnp.mean(jnp.sum(jnp.square(flow_field - flow_field_gt), axis=-1))
    return error


def loss_unsupervised(
    img1: jnp.ndarray, img2: jnp.ndarray, flow_field: jnp.ndarray
) -> jnp.ndarray:
    """Compute the loss for unsupervised training.

    Args:
        img1: First image.
        img2: Second image.
        flow_field: Estimated flow field.

    Returns:
        Computed loss.
    """
    img2_warped = apply_flow_to_image_forward(img1, flow_field, 1.0)
    return jnp.mean(jnp.square(img2 - img2_warped), axis=(1, 2))


def constraint_violation(
    flow_field: jnp.ndarray,
    max_speed: float,
    thresholds: dict,
) -> Metrics:
    """Compute the constraint violation for the flow field.

    Args:
        flow_field: Estimated flow field.
        max_speed: Maximum allowed speed.
        thresholds: Dictionary containing the thresholds for speed and divergence.

    Returns:
        Dictionary containing the statistics for speed and divergence violations.
    """
    # Compute divergence, vorticity, and flow speed
    divergence, _ = compute_divergence_and_vorticity(flow_field)
    speed = jnp.linalg.norm(flow_field, axis=-1)

    # Compute the constraint violation
    speed_violation = jnp.maximum(speed - max_speed, 0)
    divergence_violation = divergence**2

    # Extract the thresholds
    speed_threshold = thresholds.get("speed", 0.01)
    divergence_threshold = thresholds.get("divergence", 0.01)

    # Return the statistics
    return {
        **{
            f"speed/{k}": v
            for k, v in compute_stats(
                speed_violation.flatten(), speed_threshold
            ).items()
        },
        **{
            f"divergence/{k}": v
            for k, v in compute_stats(
                divergence_violation.flatten(), divergence_threshold
            ).items()
        },
    }


def compute_stats(errors: jnp.ndarray, threshold: float = 0.5) -> Metrics:
    """Compute statistics for the errors / constraint violations.

    Args:
        errors: Array of estimation errors.
        threshold: Threshold for acceptable.

    Returns:
        Mean and standard deviation of the errors.
    """
    errors = errors.flatten()  # ensure 1D array (batches are flattened!)
    mean_error = jnp.mean(errors)
    std_error = jnp.std(errors)
    n_below_threshold = jnp.sum(errors < threshold)
    # compute lq, up, median, lw, uw
    lq = jnp.percentile(errors, 25)
    up = jnp.percentile(errors, 75)
    median = jnp.median(errors)
    iqr = up - lq
    lw = lq - 1.5 * iqr
    uw = up + 1.5 * iqr

    return {
        "mean": mean_error.item(),
        "std": std_error.item(),
        "fraction_below_threshold": ((n_below_threshold / errors.shape[0]).item()),
        "lower_quartile": lq.item(),
        "upper_quartile": up.item(),
        "median": median.item(),
        "lower_whisker": lw.item(),
        "upper_whisker": uw.item(),
    }
