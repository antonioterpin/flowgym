"""Training module for flow field estimation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .caching import CacheManager, CachePayload, enrich_batch
    from .exploration import build_exploration_from_config
    from .losses import build_loss_from_config
    from .optimizer import build_optimizer_from_config
    from .replay import ReplayBuffer
    from .schedules import build_schedule_from_config
    from .target_transforms import build_target_transform_from_config

_EXPORT_MAP = {
    "CacheManager": (".caching", "CacheManager"),
    "CachePayload": (".caching", "CachePayload"),
    "ReplayBuffer": (".replay", "ReplayBuffer"),
    "build_exploration_from_config": (
        ".exploration",
        "build_exploration_from_config",
    ),
    "build_loss_from_config": (".losses", "build_loss_from_config"),
    "build_optimizer_from_config": (
        ".optimizer",
        "build_optimizer_from_config",
    ),
    "build_schedule_from_config": (".schedules", "build_schedule_from_config"),
    "build_target_transform_from_config": (
        ".target_transforms",
        "build_target_transform_from_config",
    ),
    "enrich_batch": (".caching", "enrich_batch"),
}

__all__ = [
    "CacheManager",
    "CachePayload",
    "ReplayBuffer",
    "build_exploration_from_config",
    "build_loss_from_config",
    "build_optimizer_from_config",
    "build_schedule_from_config",
    "build_target_transform_from_config",
    "enrich_batch",
]


def __getattr__(name: str) -> Any:
    """Lazily expose training symbols to avoid import-time cycles.

    Args:
        name: Symbol requested from this module namespace.

    Returns:
        The requested exported object.

    Raises:
        AttributeError: If ``name`` is not an exported symbol.
    """
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
