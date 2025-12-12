"""Postprocessing module for flow estimation."""

import jax.image as jimg
import jax.numpy as jnp
from goggles.history.types import History

from .data_interpolation import (  # noqa: F401
    tile_average_interpolation,
    tile_average_interpolation_validate_params,
    laplace_interpolation,
    laplace_interpolation_validate_params,
)

from .data_smoothing import (  # noqa: F401
    median_smoothing,
    median_smoothing_validate_params,
    average_smoothing,
    average_smoothing_validate_params,
    gaussian_smoothing,
    gaussian_smoothing_validate_params,
)

from .data_validation import (  # noqa: F401
    constant_threshold_filter,
    constant_threshold_filter_validate_params,
    adaptive_global_filter,
    adaptive_global_filter_validate_params,
    adaptive_local_filter,
    adaptive_local_filter_validate_params,
    universal_median_test,
    universal_median_test_validate_params,
)


def quantize_validate_params(
    min_val: jnp.ndarray,
    max_val: jnp.ndarray,
    dtype: jnp.dtype = jnp.uint8,
):
    """Validate parameters for quantization.

    Args:
        min_val (jnp.ndarray): Minimum value for quantization.
        max_val (jnp.ndarray): Maximum value for quantization.
        dtype (jnp.dtype): Data type for the quantized output.

    Raises:
        ValueError: If the parameters are invalid.
    """
    if not isinstance(min_val, jnp.ndarray) or not isinstance(max_val, jnp.ndarray):
        raise ValueError("min_val and max_val must be jnp.ndarray.")
    if min_val.shape != max_val.shape:
        raise ValueError("min_val and max_val must have the same shape.")
    if min_val.shape != (2,):
        raise ValueError("min_val and max_val must have shape (2,).")
    if not isinstance(dtype, jnp.dtype):
        raise ValueError("dtype must be a jnp.dtype instance.")
    if dtype not in (jnp.uint4, jnp.uint8):
        raise ValueError("dtype must be one of [jnp.uint4, jnp.uint8].")


def quantize(
    flow: jnp.ndarray,
    min_val: jnp.ndarray,
    max_val: jnp.ndarray,
    dtype: jnp.dtype = jnp.uint8,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Quantize flow values to a specified number of bits.

    Args:
        flow (jnp.ndarray): Flow values to quantize.
        min_val (jnp.ndarray): Minimum value for quantization.
        max_val (jnp.ndarray): Maximum value for quantization.
        dtype (jnp.dtype): Data type for the quantized output.
        valid (jnp.ndarray | None):
            Optional mask of shape (B, H, W) where 1 means valid.
        state (History | None): Current state of the estimator.

    Returns:
        jnp.ndarray: Quantized flow values.
        jnp.ndarray | None: mask of shape (B, H, W) where 1 means valid.
        History | None: Current state of the estimator.
    """
    flow = flow.astype(jnp.float32)
    flow = (flow - min_val) / (max_val - min_val + 1e-22)
    flow = jnp.clip(flow, 0, 1)
    flow *= jnp.iinfo(dtype).max
    return flow.astype(dtype), valid, state


def resize_flow_validate_params(
    target_height: int,
    target_width: int,
):
    """Validate parameters for flow resizing.

    Args:
        target_height (int): Target height for resizing.
        target_width (int): Target width for resizing.

    Raises:
        ValueError: If the parameters are invalid.
    """
    if not isinstance(target_height, int) or not isinstance(target_width, int):
        raise ValueError("target_height and target_width must be integers.")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_height and target_width must be positive integers.")


def resize_flow(
    flow: jnp.ndarray,
    target_height: int,
    target_width: int,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Resize flow to a target shape.

    Args:
        flow (jnp.ndarray): Flow values to resize.
        target_height (int): Target height.
        target_width (int): Target width.
        valid (jnp.ndarray | None):
            Optional mask of shape (B, H, W) where 1 means valid.
        state (History | None): Current state of the estimator.

    Returns:
        jnp.ndarray: Resized flow values.
        jnp.ndarray | None: mask of shape (B, H, W) where 1 means valid.
        History | None: Current state of the estimator.
    """
    B = flow.shape[0]
    target_shape = (B, target_height, target_width, 2)

    resized_flow = jimg.resize(
        flow,
        shape=target_shape,
        method="bilinear",
    )

    # we need to reshape valid as well
    # TODO: for now, we just return all valid
    valid = jnp.ones((B, target_height, target_width), dtype=jnp.bool_)

    return resized_flow, valid, state


def temporal_smoothing_ema_validate_params(
    alpha: float,
):
    """Validate parameters for temporal smoothing.

    Args:
        alpha (float): Smoothing factor, should be in the range [0, 1].

    Raises:
        ValueError: If the parameters are invalid.
    """
    if not isinstance(alpha, float):
        raise ValueError("alpha must be a float.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in the range [0, 1].")


def temporal_smoothing_ema(
    flow: jnp.ndarray,
    alpha: float,
    state: History,
    valid: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Apply exponential moving average smoothing to the flow.

    Args:
        flow (jnp.ndarray): Flow values to smooth.
        alpha (float): Smoothing factor, should be in the range [0, 1].
        state (History): Current state of the estimator.
        valid (jnp.ndarray | None):
            Optional mask of shape (B, H, W) where 1 means valid.

    Returns:
        jnp.ndarray: Smoothed flow values.
    """
    alpha_arr = jnp.asarray(alpha, dtype=flow.dtype)
    prev_estimate = state["estimates"][:, -1]
    return alpha_arr * flow + (1 - alpha_arr) * prev_estimate, valid, state


def validate_params(
    postprocessing_step_name: str,
    **kwargs,
):
    """Validate preprocessing step parameters.

    Args:
        postprocessing_step_name (str): Name of the postprocessing step.
        **kwargs: Parameters for the postprocessing step.

    Returns:
        bool: True if parameters are valid, False otherwise.
    """
    validate_func_name = f"{postprocessing_step_name}_validate_params"
    if validate_func_name in globals():
        validate_func = globals()[validate_func_name]
        validate_func(**kwargs)
    else:
        raise ValueError(f"Unknown postprocessing step {postprocessing_step_name}")


def apply_postprocessing(
    flow: jnp.ndarray,
    name: str,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Apply a series of preprocessing steps to an image.

    Args:
        flow (jnp.ndarray): Estimated flow.
        name (str): Name of the preprocessing function to apply.
        valid (jnp.ndarray | None):
            Optional mask of shape (B, H, W) where 1 means valid.
            Used for some preprocessing functions.
            All the postprocessing functions should accept this parameter
            to ensure concatenation of multiple postprocessing steps.
        state (History | None):
            Current state of the estimator.
            Used for some preprocessing functions.
            All the postprocessing functions should accept this parameter
            to ensure concatenation of multiple postprocessing steps.
        **kwargs: Additional parameters for the preprocessing function.

    Returns:
        jnp.ndarray: Postprocessed flow.
        jnp.ndarray | None: mask of shape (B, H, W) where 1 means valid.
        History | None: Current state of the estimator.
    """
    return globals()[name](flow, valid=valid, state=state, **kwargs)


__all__ = [
    "tile_average_interpolation",
    "tile_average_interpolation_validate_params",
    "laplace_interpolation",
    "laplace_interpolation_validate_params",
    "median_smoothing",
    "median_smoothing_validate_params",
    "average_smoothing",
    "average_smoothing_validate_params",
    "gaussian_smoothing",
    "gaussian_smoothing_validate_params",
    "constant_threshold_filter",
    "constant_threshold_filter_validate_params",
    "adaptive_global_filter",
    "adaptive_global_filter_validate_params",
    "adaptive_local_filter",
    "adaptive_local_filter_validate_params",
    "universal_median_test",
    "universal_median_test_validate_params",
    "quantize_validate_params",
    "quantize",
    "resize_flow_validate_params",
    "resize_flow",
    "temporal_smoothing_ema_validate_params",
    "temporal_smoothing_ema",
    "validate_params",
    "apply_postprocessing",
]
