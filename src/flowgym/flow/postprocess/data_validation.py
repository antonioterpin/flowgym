"""Data validation and outlier detection utilities for optical flow."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from goggles.history.types import History
from jax import lax

from flowgym.common.base import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.common.median import median
from flowgym.flow.postprocess.oracle_model import OracleMaskCNN
from flowgym.make import load_model
from flowgym.utils import DEBUG

_DEFAULT_ORACLE_FEATURES = (16, 32)
_LEARNED_ORACLE_OPTIMIZER_CONFIG = {
    "name": "adam",
    "learning_rate": 1e-3,
}
_LEARNED_ORACLE_STATE_CACHE: dict[
    tuple[str, tuple[int, ...], bool], tuple[EstimatorTrainableState, int]
] = {}


def _normalize_oracle_features(
    features: list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Validate and normalize learned-oracle CNN feature sizes."""
    if features is None:
        return _DEFAULT_ORACLE_FEATURES
    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ValueError("`features` must be a non-empty list/tuple.")
    normalized = tuple(features)
    if not all(
        isinstance(feature, int) and feature > 0
        for feature in normalized
    ):
        raise ValueError(
            "`features` must contain only positive integers, got "
            f"{features}."
        )
    return normalized


def _load_learned_oracle_state_from_checkpoint(
    load_from: str,
    features: tuple[int, ...],
    include_image_pair: bool = False,
) -> tuple[EstimatorTrainableState, int]:
    """Load and cache a learned-oracle trainable state from checkpoint."""
    checkpoint_path = str(Path(load_from).resolve())
    cache_key = (checkpoint_path, features, include_image_pair)
    cached = _LEARNED_ORACLE_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = OracleMaskCNN(features=features)
    in_channels = 5 if include_image_pair else 3
    dummy_flow = jnp.zeros((1, 8, 8, in_channels), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_flow)["params"]
    template_state = NNEstimatorTrainableState.from_config(
        apply_fn=model.apply,
        params=params,
        optimizer_config=_LEARNED_ORACLE_OPTIMIZER_CONFIG,
    )
    loaded_state = load_model(
        ckpt_dir=checkpoint_path,
        template_state=template_state,
        mode="params_only",
    )
    model_input_channels = _infer_oracle_input_channels(loaded_state.params)
    loaded = (loaded_state, model_input_channels)
    _LEARNED_ORACLE_STATE_CACHE[cache_key] = loaded
    return loaded


def preload_learned_oracle_state(
    load_from: str,
    features: list[int] | tuple[int, ...] | None = None,
    include_image_pair: bool = False,
) -> EstimatorTrainableState:
    """Preload learned-oracle state for a jittable inference path."""
    state, _ = _load_learned_oracle_state_from_checkpoint(
        load_from=load_from,
        features=_normalize_oracle_features(features),
        include_image_pair=include_image_pair,
    )
    return state


def _infer_oracle_input_channels(params: Any) -> int:
    """Infer expected input channels from first convolution params.

    Falls back to ``3`` when parameters do not expose ``Conv_0/kernel``.
    """
    try:
        conv0_kernel = jnp.asarray(params["Conv_0"]["kernel"])
    except Exception:
        return 3
    if conv0_kernel.ndim != 4:
        raise ValueError(
            "Invalid Conv_0/kernel shape for learned oracle checkpoint: "
            f"{conv0_kernel.shape}."
        )
    in_channels = int(conv0_kernel.shape[-2])
    if in_channels not in (2, 3, 5):
        raise ValueError(
            "Unsupported learned-oracle input channels. Expected 2 (legacy "
            "flow-only), 3 (flow+index), or 5 (flow+index+image-pair), got "
            f"{in_channels}."
        )
    return in_channels


def _build_oracle_model_input(
    flow_for_model: jnp.ndarray,
    estimator_indices: jnp.ndarray | None = None,
    estimator_count: int | None = None,
    input_channels: int = 3,
    previous_image: jnp.ndarray | None = None,
    current_image: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Build model input for learned oracle (2ch/3ch/5ch variants)."""
    if flow_for_model.ndim != 4 or flow_for_model.shape[-1] != 2:
        raise ValueError(
            "Expected flow input shape (B, H, W, 2), got "
            f"{flow_for_model.shape}."
        )
    if input_channels == 2:
        return flow_for_model.astype(jnp.float32)
    if input_channels not in (3, 5):
        raise ValueError(
            "input_channels must be 2, 3, or 5, got "
            f"{input_channels}."
        )

    b, h, w, _ = flow_for_model.shape
    if estimator_indices is None:
        estimator_indices_f = jnp.zeros((b,), dtype=jnp.float32)
    else:
        estimator_indices_arr = jnp.asarray(estimator_indices)
        if (
            estimator_indices_arr.ndim != 1
            or estimator_indices_arr.shape[0] != b
        ):
            raise ValueError(
                "estimator_indices must have shape (B,), got "
                f"{estimator_indices_arr.shape}."
            )
        estimator_indices_f = estimator_indices_arr.astype(jnp.float32)
    if estimator_count is not None:
        norm_denom = float(max(estimator_count - 1, 1))
    else:
        norm_denom = jnp.maximum(jnp.max(estimator_indices_f), 1.0)
    estimator_index_channel = jnp.broadcast_to(
        (estimator_indices_f / norm_denom)[:, None, None, None],
        (b, h, w, 1),
    )
    channels = [flow_for_model.astype(jnp.float32), estimator_index_channel]
    if input_channels == 5:
        if previous_image is None or current_image is None:
            raise ValueError(
                "`previous_image` and `current_image` are required when "
                "input_channels is 5."
            )
        prev = jnp.asarray(previous_image)
        curr = jnp.asarray(current_image)
        if prev.ndim == 4 and prev.shape[-1] == 1:
            prev = jnp.squeeze(prev, axis=-1)
        if curr.ndim == 4 and curr.shape[-1] == 1:
            curr = jnp.squeeze(curr, axis=-1)
        if prev.shape != (b, h, w):
            raise ValueError(
                "previous_image must have shape "
                f"(B, H, W)=({b}, {h}, {w}), got {prev.shape}."
            )
        if curr.shape != (b, h, w):
            raise ValueError(
                "current_image must have shape "
                f"(B, H, W)=({b}, {h}, {w}), got {curr.shape}."
            )
        channels.extend(
            [
                prev.astype(jnp.float32)[..., None],
                curr.astype(jnp.float32)[..., None],
            ]
        )
    return jnp.concatenate(channels, axis=-1)


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
    **kwargs: Any,
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
        **kwargs: Additional keyword arguments.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    mag = jnp.linalg.norm(flow_field, axis=-1)
    valid = valid if valid is not None else jnp.ones(mag.shape, dtype=bool)
    return flow_field, valid & ~((mag < vel_min) | (mag > vel_max)), state


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
    **kwargs: Any,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Thresholding based on mean and standard deviation of magnitudes.

    Args:
        flow_field: Input array of shape (B, H, W, 2).
        n_sigma: Number of standard deviations to use for thresholding.
        valid:
            Optional mask of shape (B, H, W) where 1 means valid.
            If provided, the outlier mask will be combined with this mask.
        state: Current state of the estimator.
        **kwargs: Additional keyword arguments.

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
    return flow_field, valid & ~((mag < lo) | (mag > hi)), state


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
        raise ValueError(
            f"Invalid radius: {radius}. Must be a non-negative integer."
        )


def adaptive_local_filter(
    flow_field: jnp.ndarray,
    n_sigma: float,
    radius: int = 1,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
    **kwargs: Any,
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
        **kwargs: Additional keyword arguments.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    if DEBUG:
        assert isinstance(flow_field, jnp.ndarray), (
            "Flow field must be a jnp.ndarray."
        )
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

    valid = (
        valid if valid is not None else jnp.ones(magnitudes.shape, dtype=bool)
    )

    # if outside the bounds, mark as outlier
    return (
        flow_field,
        valid
        & ~jnp.squeeze(
            (centers < lower_bound) | (centers > upper_bound), axis=-1
        ),
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
        raise ValueError(
            f"Invalid r_threshold: {r_threshold}. Must be positive."
        )
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError(f"Invalid epsilon: {epsilon}. Must be positive.")
    if not isinstance(radius, int) or radius < 0:
        raise ValueError(
            f"Invalid radius: {radius}. Must be a non-negative integer."
        )


def universal_median_test(
    flow_field: jnp.ndarray,
    r_threshold: float = 2.0,
    epsilon: float = 0.1,
    radius: int = 1,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
    **kwargs: Any,
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
        **kwargs: Additional keyword arguments.

    Returns:
        Original flow field.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    if DEBUG:
        assert isinstance(flow_field, jnp.ndarray), (
            "Flow field must be a jnp.ndarray."
        )
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
        valid & ~jnp.any(r0 > r_threshold, axis=-1),
        state,
    )


def learned_oracle_threshold(
    flow_field: jnp.ndarray,
    trainable_state: EstimatorTrainableState | None = None,
    valid: jnp.ndarray | None = None,
    state: History | None = None,
    threshold_value: float = 0.5,
    load_from: str | None = None,
    features: list[int] | tuple[int, ...] | None = None,
    include_image_pair: bool = False,
    previous_image: jnp.ndarray | None = None,
    current_image: jnp.ndarray | None = None,
    estimator_indices: jnp.ndarray | None = None,
    estimator_index: int | float | None = None,
    estimator_count: int | None = None,
    oracle_trainable_state: EstimatorTrainableState | None = None,
    **kwargs: Any,
) -> tuple[jnp.ndarray, jnp.ndarray | None, History | None]:
    """Apply a learned oracle threshold from a trainable mask model.

    Args:
        flow_field: Input array of shape (B, H, W, 2).
        trainable_state: Trainable state of the estimator. If omitted or
            missing ``apply_fn``, ``load_from`` is used to load one.
        valid:
             Optional mask of shape (B, H, W) where 1 means valid.
             If provided, the outlier mask will be combined with this mask.
        state: Current state of the estimator.
        load_from: Optional checkpoint directory path for the learned oracle.
        features: CNN feature sizes used by the learned oracle checkpoint.
        include_image_pair:
            Whether to append the current image pair as additional input
            channels (requires a 5-channel learned-oracle checkpoint).
        previous_image: Previous image frame with shape (B, H, W).
        current_image: Current image frame with shape (B, H, W).
        estimator_indices:
            Optional estimator indices. For K flows, shape (K,) or (B, K).
            For single-flow input, shape (B,). If omitted, defaults to
            [0..K-1] for K-flow input and all zeros for single-flow input.
        estimator_index:
            Optional scalar index used for single-flow input. This is useful
            when postprocessing one specific estimator at a time.
        estimator_count:
            Optional total number of estimators used for index normalization.
            If provided, estimator index values are normalized by
            ``max(estimator_count - 1, 1)``.
        oracle_trainable_state:
            Optional preloaded learned-oracle state. When provided, checkpoint
            loading via ``load_from`` is skipped.
        **kwargs: Additional keyword arguments.

    Returns:
        Filtered flow field with outliers removed.
        mask of shape (B, H, W) where 1 means valid.
        Current state of the estimator.
    """
    model_trainable_state = (
        oracle_trainable_state
        if oracle_trainable_state is not None
        else trainable_state
    )
    model_input_channels = 3
    if (
        model_trainable_state is None
        or model_trainable_state.apply_fn is None
    ):
        if load_from is None:
            raise ValueError("trainable_state.apply_fn cannot be None.")
        model_trainable_state, model_input_channels = (
            _load_learned_oracle_state_from_checkpoint(
                load_from=load_from,
                features=_normalize_oracle_features(features),
                include_image_pair=include_image_pair,
            )
        )
    else:
        model_input_channels = _infer_oracle_input_channels(
            model_trainable_state.params
        )

    if include_image_pair and model_input_channels != 5:
        # For non-standard lightweight states (e.g., tests with custom
        # apply_fn) where channel count cannot be inferred reliably, force 5ch.
        # For real learned-oracle checkpoints/states with explicit Conv_0
        # kernel, enforce consistency.
        has_explicit_conv0 = False
        try:
            _ = model_trainable_state.params["Conv_0"]["kernel"]
            has_explicit_conv0 = True
        except Exception:
            has_explicit_conv0 = False
        if has_explicit_conv0:
            raise ValueError(
                "include_image_pair=True requires a 5-channel learned-oracle "
                "model checkpoint/state, but the provided state expects "
                f"{model_input_channels} channels."
            )
        model_input_channels = 5

    if model_trainable_state.apply_fn is None:
        raise ValueError("trainable_state.apply_fn cannot be None.")

    flow_in = jnp.asarray(flow_field)
    original_shape = flow_in.shape
    flat_estimator_indices: jnp.ndarray | None = None
    flat_prev_image: jnp.ndarray | None = None
    flat_curr_image: jnp.ndarray | None = None
    if estimator_indices is not None and estimator_index is not None:
        raise ValueError(
            "Use either `estimator_indices` or `estimator_index`, not both."
        )
    if flow_in.ndim == 5:
        b, k, h, w, c = flow_in.shape
        if c != 2:
            raise ValueError(
                "Expected flow field shape (B, K, H, W, 2), got "
                f"{flow_in.shape}."
            )
        flow_for_model = flow_in.reshape(b * k, h, w, c)
        if estimator_index is not None:
            raise ValueError(
                "`estimator_index` is only valid for single-flow input."
            )
        if estimator_indices is None:
            idx = jnp.broadcast_to(
                jnp.arange(k, dtype=jnp.int32)[None, :], (b, k)
            )
        else:
            idx = jnp.asarray(estimator_indices)
            if idx.ndim == 1 and idx.shape[0] == k:
                idx = jnp.broadcast_to(idx[None, :], (b, k))
            elif idx.ndim == 2 and idx.shape == (b, k):
                pass
            else:
                raise ValueError(
                    "For K-flow input, estimator_indices must have shape "
                    f"(K,) or (B, K), got {idx.shape}."
                )
        flat_estimator_indices = idx.reshape(b * k)
        if model_input_channels == 5:
            if previous_image is None or current_image is None:
                raise ValueError(
                    "`previous_image` and `current_image` are required when "
                    "using a 5-channel learned-oracle model."
                )
            prev = jnp.asarray(previous_image)
            curr = jnp.asarray(current_image)
            if prev.ndim == 4 and prev.shape[-1] == 1:
                prev = jnp.squeeze(prev, axis=-1)
            if curr.ndim == 4 and curr.shape[-1] == 1:
                curr = jnp.squeeze(curr, axis=-1)
            if prev.shape != (b, h, w) or curr.shape != (b, h, w):
                raise ValueError(
                    "Image-pair shapes must be (B, H, W) for K-flow input, "
                    f"got prev={prev.shape}, curr={curr.shape}, expected "
                    f"({b}, {h}, {w})."
                )
            flat_prev_image = jnp.broadcast_to(
                prev[:, None, ...], (b, k, h, w)
            ).reshape(b * k, h, w)
            flat_curr_image = jnp.broadcast_to(
                curr[:, None, ...], (b, k, h, w)
            ).reshape(b * k, h, w)
    elif flow_in.ndim == 4:
        if flow_in.shape[-1] != 2:
            raise ValueError(
                "Expected flow field shape (B, H, W, 2), got "
                f"{flow_in.shape}."
            )
        flow_for_model = flow_in
        if estimator_indices is not None:
            idx = jnp.asarray(estimator_indices)
            if idx.ndim != 1 or idx.shape[0] != flow_in.shape[0]:
                raise ValueError(
                    "For single-flow input, estimator_indices must have shape "
                    f"(B,), got {idx.shape}."
                )
            flat_estimator_indices = idx
        elif estimator_index is not None:
            flat_estimator_indices = jnp.full(
                (flow_in.shape[0],), estimator_index, dtype=jnp.float32
            )
        if model_input_channels == 5:
            if previous_image is None or current_image is None:
                raise ValueError(
                    "`previous_image` and `current_image` are required when "
                    "using a 5-channel learned-oracle model."
                )
            prev = jnp.asarray(previous_image)
            curr = jnp.asarray(current_image)
            if prev.ndim == 4 and prev.shape[-1] == 1:
                prev = jnp.squeeze(prev, axis=-1)
            if curr.ndim == 4 and curr.shape[-1] == 1:
                curr = jnp.squeeze(curr, axis=-1)
            b, h, w, _ = flow_in.shape
            if prev.shape != (b, h, w) or curr.shape != (b, h, w):
                raise ValueError(
                    "Image-pair shapes must be (B, H, W) for single-flow "
                    f"input, got prev={prev.shape}, curr={curr.shape}, "
                    f"expected ({b}, {h}, {w})."
                )
            flat_prev_image = prev
            flat_curr_image = curr
    else:
        raise ValueError(
            "Expected flow field shape (B, H, W, 2) or (B, K, H, W, 2), got "
            f"{flow_in.shape}."
        )
    model_input = _build_oracle_model_input(
        flow_for_model=flow_for_model,
        estimator_indices=flat_estimator_indices,
        estimator_count=estimator_count,
        input_channels=model_input_channels,
        previous_image=flat_prev_image,
        current_image=flat_curr_image,
    )

    try:
        output = model_trainable_state.apply_fn(
            {"params": model_trainable_state.params}, model_input, **kwargs
        )
    except (TypeError, KeyError):
        try:
            output = model_trainable_state.apply_fn(
                {"params": model_trainable_state.params}, model_input
            )
        except (TypeError, KeyError):
            try:
                output = model_trainable_state.apply_fn(
                    model_input, model_trainable_state.params, **kwargs
                )
            except (TypeError, KeyError, AttributeError):
                output = model_trainable_state.apply_fn(
                    model_input, model_trainable_state.params
                )

    logits = output[0] if isinstance(output, tuple) else output
    logits = jnp.asarray(logits)
    if logits.ndim == 4 and logits.shape[-1] == 1:
        logits = jnp.squeeze(logits, axis=-1)
    if logits.ndim != 3:
        raise ValueError(
            "Learned oracle threshold model must return logits of shape "
            f"(B, H, W) or (B, H, W, 1), got {logits.shape}."
        )
    mask_probs = jax.nn.sigmoid(logits)

    if flow_in.ndim == 5:
        mask_probs = mask_probs.reshape(original_shape[:-1])

    if valid is None:
        valid_mask = jnp.ones(mask_probs.shape, dtype=bool)
    else:
        valid_mask = jnp.asarray(valid, dtype=jnp.bool_)
        if flow_in.ndim == 5 and valid_mask.ndim == 3:
            valid_mask = jnp.broadcast_to(
                valid_mask[:, None, ...], mask_probs.shape
            )
        if valid_mask.shape != mask_probs.shape:
            raise ValueError(
                f"valid mask shape {valid_mask.shape} does not match "
                f"mask shape {mask_probs.shape}."
            )
    return flow_field, valid_mask & (mask_probs > threshold_value), state


def learned_oracle_threshold_validate_params(
    threshold_value: float = 0.5,
    load_from: str | None = None,
    features: list[int] | tuple[int, ...] | None = None,
    include_image_pair: bool = False,
    estimator_index: int | float | None = None,
    estimator_count: int | None = None,
    oracle_trainable_state: EstimatorTrainableState | None = None,
):
    """Validate parameters for learned oracle thresholding.

    Args:
        threshold_value: Probability cutoff used to keep an inlier.
        load_from: Optional checkpoint directory for the learned oracle model.
        features: Optional learned oracle CNN feature sizes.
        include_image_pair: Whether to append the image pair as input.
        estimator_index: Optional scalar estimator index.
        estimator_count: Optional total number of estimators.
        oracle_trainable_state: Optional preloaded oracle trainable state.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if not isinstance(threshold_value, (int, float)) or not (
        0.0 < threshold_value < 1.0
    ):
        raise ValueError(
            "`threshold_value` must be in (0, 1), got "
            f"{threshold_value}."
        )
    if load_from is not None and not isinstance(load_from, str):
        raise ValueError(
            "`load_from` must be a string path when provided, got "
            f"{type(load_from)}."
        )
    if not isinstance(include_image_pair, bool):
        raise ValueError(
            "`include_image_pair` must be a boolean, got "
            f"{type(include_image_pair)}."
        )
    if estimator_index is not None and not isinstance(
        estimator_index, (int, float)
    ):
        raise ValueError(
            "`estimator_index` must be numeric when provided, got "
            f"{type(estimator_index)}."
        )
    if estimator_count is not None and (
        not isinstance(estimator_count, int) or estimator_count <= 0
    ):
        raise ValueError(
            "`estimator_count` must be a positive integer when provided, got "
            f"{estimator_count}."
        )
    if oracle_trainable_state is not None and not isinstance(
        oracle_trainable_state, EstimatorTrainableState
    ):
        raise ValueError(
            "`oracle_trainable_state` must be an EstimatorTrainableState, got "
            f"{type(oracle_trainable_state)}."
        )
    _normalize_oracle_features(features)
