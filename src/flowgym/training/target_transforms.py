"""Target transformation utilities for router training."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from goggles import get_logger

logger = get_logger(__name__)

TargetTransform = Callable[[jax.Array], jax.Array]


def _identity(x: jax.Array, **_: Any) -> jax.Array:
    """Return input unchanged.

    Args:
        x: Input array.
        **_: Ignored keyword arguments.

    Returns:
        Input array unchanged.
    """
    return x


def _log1p(x: jax.Array, eps: float = 0.0) -> jax.Array:
    """Log1p transform with epsilon and non-negative clamp.

    Args:
        x: Input array.
        eps: Small constant added before log.

    Returns:
        Log1p of (max(x, 0) + eps).
    """
    return jnp.log1p(jnp.maximum(x, 0.0) + eps)


def _sqrt(x: jax.Array, eps: float = 0.0) -> jax.Array:
    """Square root transform with epsilon and non-negative clamp.

    Args:
        x: Input array.
        eps: Small constant added before sqrt.

    Returns:
        Sqrt of (max(x, 0) + eps).
    """
    return jnp.sqrt(jnp.maximum(x, 0.0) + eps)


def _clip(
    x: jax.Array, min_val: float | None = None, max_val: float | None = None
) -> jax.Array:
    """Clip array to range [min_val, max_val].

    Args:
        x: Input array.
        min_val: Minimum value (unbounded if None).
        max_val: Maximum value (unbounded if None).

    Returns:
        Clipped array.
    """
    lo = -jnp.inf if min_val is None else min_val
    hi = jnp.inf if max_val is None else max_val
    return jnp.clip(x, lo, hi)


def _scale(x: jax.Array, factor: float = 1.0) -> jax.Array:
    """Scale array by a constant factor.

    Args:
        x: Input array.
        factor: Scaling factor.

    Returns:
        Scaled array.
    """
    return x * factor


TransformFn = Callable[..., jax.Array]

TRANSFORM_REGISTRY: dict[str, TransformFn] = {
    "identity": _identity,
    "log1p": _log1p,
    "sqrt": _sqrt,
    "clip": _clip,
    "scale": _scale,
}


def build_target_transform_from_config(
    config: dict[str, Any] | None,
) -> TargetTransform:
    """Build a target transform function from a config.

    Accepts either a single transform config or a pipeline of transforms.

    Config format:
        name: The transform name (from TRANSFORM_REGISTRY)
        **kwargs: Arguments for the transform function

    Pipeline format:
        pipeline: List of transform configs to apply sequentially.

    Example:
        config = {"name": "log1p", "eps": 1e-8}
        config = {
            "pipeline": [
                {"name": "clip", "min": 0.0, "max": 10.0},
                {"name": "log1p"}
            ]
        }

    Args:
        config: Configuration dictionary for the transform(s).

    Returns:
        A JAX-traceable function that applies the transformation(s).

    Raises:
        TypeError: If config or pipeline steps are not dictionaries.
        ValueError: If transform names are unknown or config is malformed.
    """
    if config is None:
        return _identity

    if "pipeline" in config:
        pipeline = config["pipeline"]
        if not isinstance(pipeline, (list, tuple)):
            pipeline_type = type(pipeline).__name__
            raise TypeError(
                f"Target transform pipeline must be a list or tuple; got "
                f"{pipeline_type}"
            )

        transforms = []
        for i, step in enumerate(pipeline):
            if not isinstance(step, dict):
                step_type = type(step).__name__
                raise TypeError(
                    f"Pipeline step {i} must be a dictionary; got {step_type}"
                )
            transforms.append(build_target_transform_from_config(step))

        def composed_transform(x: jax.Array) -> jax.Array:
            for t in transforms:
                x = t(x)
            return x

        return composed_transform

    name = config.get("name")
    if name is None:
        raise ValueError("Target transform config must contain a 'name' field.")

    name = str(name).lower()
    base_fn = TRANSFORM_REGISTRY.get(name)
    if base_fn is None:
        available = ", ".join(sorted(TRANSFORM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown transform name '{name}'. Available: {available}"
        )

    # Config ergonomics: support both inline kwargs and {kwargs: {...}}
    cfg = dict(config)
    kwargs = dict(cfg.get("kwargs", {}) or {})
    # Also allow inline kwargs (everything except 'name'/'pipeline'/'kwargs')
    inline = {
        k: v for k, v in cfg.items() if k not in {"name", "pipeline", "kwargs"}
    }
    kwargs.update(inline)

    # Special-case mapping for clip keys (min/max -> min_val/max_val)
    if name == "clip":
        kwargs = {
            "min_val": kwargs.pop("min", None),
            "max_val": kwargs.pop("max", None),
            **kwargs,
        }

    def transform_fn(x: jax.Array) -> jax.Array:
        return base_fn(x, **kwargs)

    return transform_fn
