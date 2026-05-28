"""Utilities to build Optax optimizers from config mappings."""

from collections.abc import Mapping
from typing import Any

import goggles as gg
import optax

from .schedules import build_schedule_from_config

logger = gg.get_logger(__name__)

OPTIMIZER_REGISTRY: dict[str, Any] = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "rmsprop": optax.rmsprop,
    "set_to_zero": optax.set_to_zero,
}

TRANSFORM_REGISTRY: dict[str, Any] = {
    "clip": optax.clip,
    "clip_by_global_norm": optax.clip_by_global_norm,
    "add_decayed_weights": optax.add_decayed_weights,
    "scale": optax.scale,
    "scale_by_schedule": optax.scale_by_schedule,
    "ema": optax.ema,
}


def _build_hyperparams(hcfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build hyperparams dict: scalars or schedules for inject_hyperparams.

    Args:
        hcfg: Hyperparameter config mapping.

    Returns:
        A dictionary of hyperparameters with schedules built where specified.
    """
    hyperparams: dict[str, Any] = {}

    for name, value in hcfg.items():
        # If it looks like a schedule config, build a schedule
        if isinstance(value, Mapping) and "schedule" in value:
            hyperparams[name] = build_schedule_from_config(value["schedule"])
        else:
            hyperparams[name] = value

    return hyperparams


def build_optimizer_from_config(
    config: dict[str, Any] | None,
) -> optax.GradientTransformation:
    """Build a GradientTransformation from a config mapping.

    Args:
        config: Configuration dictionary for the optimizer. If ``None``,
            defaults to ``{"name": "set_to_zero"}`` (a no-op optimizer)
            and emits a warning.

    Returns:
        An Optax GradientTransformation instance.

    Raises:
        ValueError: If config is invalid or optimizer name is unknown.

    Example:
        Example optimizer config::

            {
                "name": "adam",
                "hyperparams": {
                    "learning_rate": {
                        "schedule": {
                            "name": "exponential_decay",
                        },
                    },
                    "b1": 0.9,
                    "b2": 0.999,
                },
                "chain": [
                    {
                        "name": "clip_by_global_norm",
                        "kwargs": {"max_norm": 1.0},
                    },
                    {
                        "name": "add_decayed_weights",
                        "kwargs": {"weight_decay": 1.0e-4},
                    },
                ],
            }
    """
    if config is None:
        config = {"name": "set_to_zero"}
        logger.warning(
            "No optimizer_config provided; defaulting to 'set_to_zero'."
        )
    cfg = config.copy()
    if "name" not in cfg:
        raise ValueError("optimizer_config must contain a 'name' field.")

    opt_name = str(cfg["name"]).lower()
    base_constructor = OPTIMIZER_REGISTRY.get(opt_name)
    if base_constructor is None:
        raise ValueError(f"Unsupported optimizer name '{opt_name}'.")

    hyper_cfg = cfg.get("hyperparams", {}) or {}
    # Handle top-level keys as hyperparameters for backward compatibility.
    # Avoid known configuration keys.
    reserved_keys = {"name", "hyperparams", "chain"}
    for k, v in cfg.items():
        if k not in reserved_keys and k not in hyper_cfg:
            hyper_cfg[k] = v

    hyperparams = _build_hyperparams(hyper_cfg)

    # inject_hyperparams
    base_tx = optax.inject_hyperparams(base_constructor)(**hyperparams)

    # Optional extra transforms in a chain
    chain_cfg: list[dict[str, Any]] = config.get("chain", [])
    extra_txs: list[optax.GradientTransformation] = []
    for tcfg in chain_cfg:
        tname = str(tcfg["name"])
        kwargs = dict(tcfg.get("kwargs", {}))
        t_constructor = TRANSFORM_REGISTRY.get(tname)
        if t_constructor is None:
            raise ValueError(f"Unknown transform name '{tname}'.")
        extra_txs.append(t_constructor(**kwargs))

    if extra_txs:
        return optax.chain(*extra_txs, base_tx)
    return base_tx
