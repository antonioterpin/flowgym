"""Configuration loading helpers for Art of PIV validation."""

from typing import Any

from flowgym.utils import load_configuration


def validate_and_load_estimators_config(
    estimators_list_path: str,
) -> dict[str, Any]:
    """Validate the estimator-list path and load its YAML config.

    Args:
        estimators_list_path: Path to the estimator-list YAML file.

    Returns:
        The loaded estimator configuration dictionary.

    Raises:
        TypeError: If the path is not a string.
        ValueError: If the path does not point to a ``.yaml`` file.
    """
    if not isinstance(estimators_list_path, str):
        raise TypeError(
            f"estimators_list_path must be a string, "
            f"got {estimators_list_path}."
        )

    if not estimators_list_path.endswith(".yaml"):
        raise ValueError(
            f"estimators_list_path must be a .yaml file, "
            f"got {estimators_list_path}."
        )

    return load_configuration(estimators_list_path)
