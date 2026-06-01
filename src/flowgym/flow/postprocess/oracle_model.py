"""Shared model architecture for learned oracle outlier rejection."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class OracleMaskCNN(nn.Module):
    """Small CNN producing per-pixel mask logits from a flow field."""

    features: tuple[int, ...] = (16, 32)

    @nn.compact
    def __call__(self, flow_field: jnp.ndarray) -> jnp.ndarray:
        x = flow_field
        for feature in self.features:
            x = nn.Conv(feature, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        logits = nn.Conv(1, kernel_size=(1, 1), padding="SAME")(x)
        return jnp.squeeze(logits, axis=-1)


def build_oracle_model_input(
    flow_for_model: jnp.ndarray,
    estimator_indices: jnp.ndarray | None = None,
    estimator_count: int | None = None,
    input_channels: int = 3,
    previous_image: jnp.ndarray | None = None,
    current_image: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Assemble the learned-oracle model input (2ch/3ch/5ch variants).

    Single source of truth for the channel layout and estimator-index
    normalization shared by training (``LearnedOracleThresholdEstimator``)
    and inference (the ``learned_oracle_threshold`` postprocess step), so the
    two paths cannot drift.

    Args:
        flow_for_model: Flow field of shape (B, H, W, 2).
        estimator_indices: Optional per-sample estimator index of shape (B,).
            Defaults to all zeros.
        estimator_count: Total number of estimators K. When provided the
            index channel is normalized by ``max(K - 1, 1)``; otherwise by
            ``max(max(index), 1)`` (correct for K-flow input whose indices
            span ``0..K-1``).
        input_channels: 2 (flow only), 3 (flow + index), or 5 (flow + index +
            image pair).
        previous_image: Previous frame (B, H, W); required for 5 channels.
        current_image: Current frame (B, H, W); required for 5 channels.

    Returns:
        Model input of shape (B, H, W, input_channels).

    Raises:
        ValueError: On bad shapes, an unsupported ``input_channels``, or
            missing images for the 5-channel variant.
    """
    if flow_for_model.ndim != 4 or flow_for_model.shape[-1] != 2:
        raise ValueError(
            "Expected flow input shape (B, H, W, 2), got "
            f"{flow_for_model.shape}."
        )
    if input_channels == 2:
        return flow_for_model.astype(jnp.float32)
    if input_channels not in (3, 5):
        raise ValueError(
            f"input_channels must be 2, 3, or 5, got {input_channels}."
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
        # For K-flow input the indices span [0..K-1], so max(index) == K-1 and
        # this matches training. Single-flow callers (constant index) must pass
        # estimator_count; that case is guarded where the indices are built.
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
