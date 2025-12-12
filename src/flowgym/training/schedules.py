"""Optax learning rate schedules registry and builder."""
from __future__ import annotations

from typing import Any

import optax
from optax import schedules as opt_schedules

from goggles import get_logger

logger = get_logger(__name__)


SCHEDULE_REGISTRY: dict[str, Any] = {
    "constant": opt_schedules.constant_schedule,
    "linear": opt_schedules.linear_schedule,
    "exponential_decay": opt_schedules.exponential_decay,
    "cosine_decay": opt_schedules.cosine_decay_schedule,
    "cosine_onecycle": opt_schedules.cosine_onecycle_schedule,
    "linear_onecycle": opt_schedules.linear_onecycle_schedule,
    "piecewise_constant": opt_schedules.piecewise_constant_schedule,
    "piecewise_interpolate": opt_schedules.piecewise_interpolate_schedule,
    "polynomial": opt_schedules.polynomial_schedule,
    "sgdr": opt_schedules.sgdr_schedule,
    "warmup_constant": opt_schedules.warmup_constant_schedule,
    "warmup_cosine_decay": opt_schedules.warmup_cosine_decay_schedule,
    "warmup_exponential_decay": opt_schedules.warmup_exponential_decay_schedule,
    "join_schedules": opt_schedules.join_schedules,
    }


def _build_schedule_from_config(config: dict[str, Any]) -> optax.Schedule:
    """Build an Optax Schedule from a config.

    Config forms:
        - {name: ..., kwargs: {...}} where name is in SCHEDULE_REGISTRY
        - special case for join_schedules with nested schedules.
    
    Args:
        config: Configuration dictionary for the schedule.
        
    Returns:
        An Optax Schedule instance.
    """
    if not isinstance(config, dict):
        raise ValueError(
            "Schedule config must be a dictionary; "
            f"got {type(config).__name__}"
        )

    if "name" not in config:
        raise ValueError("Schedule config must contain a 'name' field.")

    cfg = config.copy()
    name = str(cfg.pop("name"))
    name_key = name.lower()

    # Special case: join_schedules has nested schedules.
    if name_key == "join_schedules":
        # Expected:
        #   schedules: [<schedule_cfg>, ...]
        #   boundaries: [int, ...]
        try:
            sub_cfgs = cfg["schedules"]
            boundaries = cfg["boundaries"]
        except KeyError as exc:
            raise ValueError(
                "join_schedules config must contain 'schedules' and 'boundaries'."
            ) from exc

        if not isinstance(sub_cfgs, (list, tuple)):
            raise ValueError("join_schedules.schedules must be a list.")

        sub_schedules = [
            _build_schedule_from_config(c) for c in sub_cfgs
        ]
        return opt_schedules.join_schedules(sub_schedules, boundaries)

    fn = SCHEDULE_REGISTRY.get(name_key)
    if fn is None:
        raise ValueError(f"Unknown schedule name '{name}'.")

    kwargs = dict(cfg.get("kwargs", {}))
    return fn(**kwargs)
