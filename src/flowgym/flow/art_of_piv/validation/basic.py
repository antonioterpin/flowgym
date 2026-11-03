"""Basic scalar and flag validation helpers for ART of PIV."""

from typing import cast

import jax

from flowgym.types import PRNGKey

from .shared import TRAIN_TYPES, TrainType, _format_allowed


def validate_boolean_flags(
    output_full_res: bool,
    use_temporal_propagation: bool,
    use_last_estimate: bool,
) -> tuple[bool, bool, bool]:
    """Validate boolean control flags.

    Args:
        output_full_res: Whether to output full-resolution predictions.
        use_temporal_propagation: Whether to reuse temporal context across
            frames.
        use_last_estimate: Whether to feed the previous estimate back into
            the model.

    Returns:
        Tuple of validated boolean flags in the same order they were passed.

    Raises:
        TypeError: If any flag is not a boolean.
    """
    if not isinstance(output_full_res, bool):
        raise TypeError(
            f"output_full_res must be a boolean, got {output_full_res}."
        )

    if not isinstance(use_temporal_propagation, bool):
        raise TypeError(
            f"use_temporal_propagation must be a boolean, "
            f"got {use_temporal_propagation}."
        )

    if not isinstance(use_last_estimate, bool):
        raise TypeError(
            f"use_last_estimate must be a boolean, got {use_last_estimate}."
        )

    return (
        output_full_res,
        bool(use_temporal_propagation),
        bool(use_last_estimate),
    )


def validate_train_type(train_type: str) -> TrainType:
    """Validate the configured training paradigm.

    Args:
        train_type: Requested training paradigm.

    Returns:
        The validated training type.

    Raises:
        ValueError: If ``train_type`` is not one of the supported values.
    """
    if train_type not in TRAIN_TYPES:
        raise ValueError(
            f"train_type must be one of ({_format_allowed(TRAIN_TYPES)}); "
            f"got {train_type!r}."
        )
    return cast(TrainType, train_type)


def validate_rng(rng: PRNGKey | int | None) -> PRNGKey | None:
    """Validate and normalize the RNG input.

    Args:
        rng: Optional PRNG key or integer seed.

    Returns:
        A JAX PRNG key when given an integer seed, the original key when one
        is provided, or ``None``.
    """
    if isinstance(rng, int):
        return jax.random.PRNGKey(rng)
    return rng


def validate_gamma(gamma: float) -> float:
    """Validate the RL discount factor.

    Args:
        gamma: Discount factor for future rewards.

    Returns:
        The validated gamma value.

    Raises:
        TypeError: If ``gamma`` is not numeric.
        ValueError: If ``gamma`` is outside the range ``[0, 1)``.
    """
    if not isinstance(gamma, (int, float)):
        raise TypeError(f"gamma must be a float or int, got {gamma}.")

    if gamma < 0 or gamma >= 1:
        raise ValueError(f"gamma must be in the range [0, 1), got {gamma}.")

    return gamma
