"""Training-related validation helpers for ART of PIV."""

import goggles as gg

from .shared import (
    CLASSIFICATION_LOSSES,
    GREEDY_ONLY_MODES,
    TRAINING_MODES,
    TrainingMode,
    _format_allowed,
)

logger = gg.get_logger(__name__)


def validate_training_mode_and_loss(
    training_mode: str,
    loss_config: dict | None,
) -> TrainingMode:
    """Validate the training mode and its loss compatibility.

    Args:
        training_mode: Requested supervised training mode.
        loss_config: Optional loss configuration.

    Returns:
        The validated training mode.

    Raises:
        ValueError: If the training mode is unknown or the loss is
            incompatible with classification mode.
    """
    if training_mode not in TRAINING_MODES:
        raise ValueError(
            f"training_mode must be one of "
            f"({_format_allowed(TRAINING_MODES)}); got {training_mode!r}."
        )

    if training_mode == "classification":
        loss_name = loss_config.get("name", "").lower() if loss_config else ""
        if loss_name not in CLASSIFICATION_LOSSES:
            raise ValueError(
                f"Classification mode requires a classification loss. "
                f"Got {loss_name!r}, but expected one of "
                f"({_format_allowed(CLASSIFICATION_LOSSES)}). Use --loss "
                f"softmax_cross_entropy_with_integer_labels for "
                f"classification."
            )

    return training_mode


def validate_training_mode_exploration_compatibility(
    training_mode: TrainingMode,
    exploration_config: dict | None,
) -> None:
    """Warn about exploration settings that do not fit the mode well.

    Args:
        training_mode: Validated training mode.
        exploration_config: Optional exploration configuration.
    """
    if training_mode in GREEDY_ONLY_MODES:
        logger.warning(
            f"Training in {training_mode!r} mode. This will run ALL "
            "sub-estimators if values are not cached, which is slow. "
            "Please ensure caching is properly configured for optimal "
            "performance."
        )

        if exploration_config is not None:
            schedule = exploration_config.get("schedule", {})
            is_greedy = (
                exploration_config.get("name") == "epsilon_greedy"
                and schedule.get("name") == "constant"
                and schedule.get("value") == 0.0
            )
            if not is_greedy:
                logger.warning(
                    f"Exploration strategy is configured with "
                    f"non-zero exploration in '{training_mode}' mode. "
                    f"This will NOT affect training (all actions are "
                    f"trained), but WILL affect the reported accuracy "
                    f"metric. Consider using epsilon=0 for meaningful "
                    f"accuracy metrics during training."
                )


def validate_and_set_exploration_defaults(
    exploration_config: dict | None,
    training_mode: TrainingMode,
) -> dict:
    """Validate exploration config and fill in mode-appropriate defaults.

    Args:
        exploration_config: Optional exploration configuration.
        training_mode: Validated training mode used to choose defaults.

    Returns:
        A validated exploration configuration dictionary.

    Raises:
        TypeError: If the config or nested schedule is not a dictionary.
    """
    if exploration_config is None:
        if training_mode in GREEDY_ONLY_MODES:
            exploration_config = {
                "name": "epsilon_greedy",
                "schedule": {"name": "constant", "value": 0.0},
            }
        else:
            exploration_config = {
                "name": "epsilon_greedy",
                "schedule": {"name": "constant", "value": 0.1},
            }

    if not isinstance(exploration_config, dict):
        raise TypeError(
            "exploration_config must be a dict, "
            f"got {type(exploration_config)}."
        )

    schedule_cfg = exploration_config.get("schedule", {})
    if not isinstance(schedule_cfg, dict):
        raise TypeError(
            f"exploration_config 'schedule' must be a dict, got {schedule_cfg}."
        )

    return exploration_config
