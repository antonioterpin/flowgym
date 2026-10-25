"""Photometric augmentation for training on real recordings.

Only pixel intensities change, never positions, so displacements (the signal a
model should learn from) are untouched. This makes the model insensitive to the
lighting differences between recordings, which it would otherwise use as a
shortcut.
"""

import jax
import jax.numpy as jnp


def _validate_range(name: str, bounds: tuple[float, float]) -> None:
    if bounds[0] > bounds[1]:
        raise ValueError(f"{name} must satisfy low <= high, got {bounds}")


def photometric_augment_validate_params(
    gain_range: tuple[float, float] = (0.4, 1.6),
    offset_range: tuple[float, float] = (-20.0, 20.0),
    frame_gain_range: tuple[float, float] = (0.9, 1.1),
    max_value: float = 255.0,
) -> None:
    """Validate parameters for photometric augmentation.

    Args:
        gain_range: Bounds of the gain shared by both frames.
        offset_range: Bounds of the offset shared by both frames.
        frame_gain_range: Bounds of the extra gain drawn per frame.
        max_value: Maximum image intensity, e.g. 255 for 8-bit-like images.

    Raises:
        ValueError: If a range is reversed, a gain bound is not positive, or
            `max_value` is not positive.
    """
    for name, bounds in (
        ("gain_range", gain_range),
        ("offset_range", offset_range),
        ("frame_gain_range", frame_gain_range),
    ):
        _validate_range(name, bounds)
    for name, bounds in (
        ("gain_range", gain_range),
        ("frame_gain_range", frame_gain_range),
    ):
        if bounds[0] <= 0:
            raise ValueError(f"{name} must be positive, got {bounds}")
    if max_value <= 0:
        raise ValueError(f"max_value must be positive, got {max_value}")


def photometric_augment(
    pair: jnp.ndarray,
    key: jax.Array,
    gain_range: tuple[float, float] = (0.4, 1.6),
    offset_range: tuple[float, float] = (-20.0, 20.0),
    frame_gain_range: tuple[float, float] = (0.9, 1.1),
    max_value: float = 255.0,
) -> jnp.ndarray:
    """Randomize the lighting of an image pair.

    Applies a gain and an offset shared by both frames and a small extra gain
    per frame (the two laser pulses are never equally bright), then clips to
    ``[0, max_value]``. Use ``jax.vmap`` to augment a batch.

    Args:
        pair: Frame pair of shape (2, H, W) with values in ``[0, max_value]``.
        key: PRNG key.
        gain_range: Bounds of the gain shared by both frames.
        offset_range: Bounds of the offset shared by both frames.
        frame_gain_range: Bounds of the extra gain drawn per frame.
        max_value: Maximum image intensity.

    Returns:
        Augmented pair of shape (2, H, W).
    """
    k_gain, k_offset, k_frame = jax.random.split(key, 3)
    gain = jax.random.uniform(
        k_gain, minval=gain_range[0], maxval=gain_range[1]
    )
    offset = jax.random.uniform(
        k_offset, minval=offset_range[0], maxval=offset_range[1]
    )
    frame_gain = jax.random.uniform(
        k_frame,
        (2, 1, 1),
        minval=frame_gain_range[0],
        maxval=frame_gain_range[1],
    )
    return jnp.clip(pair * gain * frame_gain + offset, 0.0, max_value)
