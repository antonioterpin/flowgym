"""MLP (Multi-Layer Perceptron) models and builders.

This module provides configurable MLP architectures for use in estimators.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from flowgym.nn.registry import get_activation


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture.

    Attributes:
        hidden_dims: Tuple of hidden layer dimensions.
        output_dim: Output dimension (default: 1 for scalar output).
        activation: Activation function (default: relu).
        use_bias: Whether to use bias in dense layers (default: True).
    """

    hidden_dims: tuple[int, ...]
    output_dim: int = 1
    activation: Callable = nn.relu
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through MLP.

        Args:
            x: Input features (..., input_dim).

        Returns:
            Output (..., output_dim).
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim, use_bias=self.use_bias)(x)
            x = self.activation(x)

        # Output layer
        x = nn.Dense(self.output_dim, use_bias=self.use_bias)(x)
        return x


def build_mlp_from_config(config: dict[str, Any]) -> MLP:
    """Build an MLP model from a configuration dictionary.

    Config structure:
        {
            "hidden_dims": [64, 32],      # required
            "output_dim": 1,               # optional, default 1
            "activation": "relu",          # optional, default "relu"
            "use_bias": true,              # optional, default true
        }

    Args:
        config: Configuration dictionary.

    Returns:
        Configured MLP model.

    Raises:
        KeyError: If required fields are missing.
        ValueError: If configuration values are invalid.
        TypeError: If configuration types are incorrect.
    """
    # Validate required fields
    if "hidden_dims" not in config:
        raise KeyError(
            "Config must contain 'hidden_dims'. "
            "Example: {'hidden_dims': [64, 32]}"
        )

    # Extract and validate hidden_dims
    hidden_dims = config["hidden_dims"]
    if not isinstance(hidden_dims, (list, tuple)):
        raise TypeError(
            "'hidden_dims' must be a list or tuple, "
            f"got {type(hidden_dims).__name__}"
        )
    if len(hidden_dims) == 0:
        raise ValueError("'hidden_dims' must be non-empty")
    if not all(isinstance(d, int) and d > 0 for d in hidden_dims):
        raise ValueError(
            "All hidden dimensions must be positive integers, "
            f"got {hidden_dims}"
        )
    hidden_dims = tuple(hidden_dims)

    # Extract optional fields with defaults
    output_dim = config.get("output_dim", 1)
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError(
            f"'output_dim' must be a positive integer, got {output_dim}"
        )

    activation_name = config.get("activation", "relu")
    if not isinstance(activation_name, str):
        raise TypeError(
            "'activation' must be a string, "
            f"got {type(activation_name).__name__}"
        )
    activation = get_activation(activation_name)

    use_bias = config.get("use_bias", True)
    if not isinstance(use_bias, bool):
        raise TypeError(
            f"'use_bias' must be a boolean, got {type(use_bias).__name__}"
        )

    return MLP(
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
        use_bias=use_bias,
    )
