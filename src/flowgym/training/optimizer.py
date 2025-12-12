"""Utilities to build Optax optimizers from config mappings."""

from collections.abc import Mapping
from typing import Any
import optax

from .schedules import _build_schedule_from_config

OPTIMIZER_REGISTRY: dict[str, Any] = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "rmsprop": optax.rmsprop,
}

TRANSFORM_REGISTRY: dict[str, Any] = {
    "clip_by_global_norm": optax.clip_by_global_norm,
    "add_decayed_weights": optax.add_decayed_weights,
    "scale": optax.scale,
    "scale_by_schedule": optax.scale_by_schedule,
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
            hyperparams[name] = _build_schedule_from_config(value["schedule"])
        else:
            hyperparams[name] = value

    return hyperparams


def build_optimizer_from_config(
    config: dict[str, Any],
) -> optax.GradientTransformation:
    """Build a GradientTransformation from a config mapping.

    Config format example:
        name: "adam"
        hyperparams:
            learning_rate:
                schedule:
                    name: "exponential_decay"
                    ...  # schedule kwargs
            b1: 0.9
            b2: 0.999
        chain:
            - name: "clip_by_global_norm"
            kwargs: {max_norm: 1.0}
            - name: "add_decayed_weights"
            kwargs: {weight_decay: 1.0e-4}

    Args:
        config: Configuration dictionary for the optimizer.
        
    Returns:
        An Optax GradientTransformation instance.
    """
    cfg = config.copy()
    if "name" not in cfg:
        raise ValueError("optimizer_config must contain a 'name' field.")

    opt_name = str(cfg["name"]).lower()
    base_constructor = OPTIMIZER_REGISTRY.get(opt_name)
    if base_constructor is None:
        raise ValueError(f"Unsupported optimizer name '{opt_name}'.")

    hyper_cfg = cfg.get("hyperparams", {}) or {}
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
