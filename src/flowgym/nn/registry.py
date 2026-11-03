"""Registries for neural network primitives.

This module provides registries and lookup functions for:
- activation functions
- normalization layers
- pooling operations

Each registry follows the pattern used in flowgym.training for consistency.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import linen as nn


def _leaky_relu(x: jnp.ndarray) -> jnp.ndarray:
    """Apply Leaky ReLU with a default negative slope of 0.01.

    Args:
        x: Input tensor.

    Returns:
        Activated tensor.
    """
    return nn.leaky_relu(x, negative_slope=0.01)


def _identity(x: jnp.ndarray) -> jnp.ndarray:
    """Apply identity activation.

    Args:
        x: Input tensor.

    Returns:
        Input tensor unchanged.
    """
    return x


ACTIVATION_REGISTRY: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "swish": nn.swish,  # Alias for silu
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "leaky_relu": _leaky_relu,
    "elu": nn.elu,
    "softplus": nn.softplus,
    "none": _identity,
}

# Normalization types - validated keys, actual instantiation happens in blocks
NORM_REGISTRY: dict[str, str] = {
    "batch": "batch",
    "instance": "instance",
    "group": "group",
    "none": "none",
}


def _max_pool(
    x: jnp.ndarray,
    window_shape: tuple[int, int] = (2, 2),
    strides: tuple[int, int] = (2, 2),
    padding: str = "SAME",
) -> jnp.ndarray:
    """Apply max pooling.

    Args:
        x: Input tensor.
        window_shape: Pooling window shape.
        strides: Pooling stride.
        padding: Padding mode.

    Returns:
        Pooled tensor.
    """
    return nn.max_pool(
        x, window_shape=window_shape, strides=strides, padding=padding
    )


def _avg_pool(
    x: jnp.ndarray,
    window_shape: tuple[int, int] = (2, 2),
    strides: tuple[int, int] = (2, 2),
    padding: str = "SAME",
) -> jnp.ndarray:
    """Apply average pooling.

    Args:
        x: Input tensor.
        window_shape: Pooling window shape.
        strides: Pooling stride.
        padding: Padding mode.

    Returns:
        Pooled tensor.
    """
    return nn.avg_pool(
        x, window_shape=window_shape, strides=strides, padding=padding
    )


def _no_pool(x: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
    """Return the input tensor unchanged.

    Args:
        x: Input tensor.
        **kwargs: Unused keyword arguments.

    Returns:
        Input tensor unchanged.
    """
    return x


POOLING_REGISTRY: dict[str, Callable[..., jnp.ndarray]] = {
    "max": _max_pool,
    "avg": _avg_pool,
    "none": _no_pool,
}


def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get activation function by name.

    Args:
        name: Name of the activation function.

    Returns:
        Activation function.

    Raises:
        ValueError: If the activation name is not found in the registry.
    """
    name_lower = name.lower()
    if name_lower not in ACTIVATION_REGISTRY:
        available = ", ".join(sorted(ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation '{name}'. Available options: {available}"
        )
    return ACTIVATION_REGISTRY[name_lower]


def get_norm(name: str) -> str:
    """Get normalization type by name.

    This returns the validated string key rather than a layer instance,
    since normalization layers are instantiated inside Flax modules with
    specific parameters.

    Args:
        name: Name of the normalization type.

    Returns:
        Validated normalization type string.

    Raises:
        ValueError: If the normalization name is not found in the registry.
    """
    name_lower = name.lower()
    if name_lower not in NORM_REGISTRY:
        available = ", ".join(sorted(NORM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown normalization '{name}'. Available options: {available}"
        )
    return NORM_REGISTRY[name_lower]


def get_pooling(name: str) -> Callable[..., jnp.ndarray]:
    """Get pooling operation by name.

    Args:
        name: Name of the pooling operation.

    Returns:
        Pooling function.

    Raises:
        ValueError: If the pooling name is not found in the registry.
    """
    name_lower = name.lower()
    if name_lower not in POOLING_REGISTRY:
        available = ", ".join(sorted(POOLING_REGISTRY.keys()))
        raise ValueError(
            f"Unknown pooling '{name}'. Available options: {available}"
        )
    return POOLING_REGISTRY[name_lower]
