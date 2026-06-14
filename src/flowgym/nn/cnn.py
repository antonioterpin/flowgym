"""Module implementing a CNN model using Flax."""

from collections.abc import Callable
from typing import Any, Literal, cast

import jax.numpy as jnp
from flax import linen as nn

from flowgym.nn.blocks import (
    ConvBlock,
    PostProcess,
    ResidualBlock,
    build_postprocess_pipeline,
)
from flowgym.nn.mlp import MLP
from flowgym.nn.registry import get_activation, get_norm


class CNNDensityModel(nn.Module):
    """A configurable CNN that outputs a single scalar per example.

    Attributes:
        features_list: List of feature dimensions for each layer.
        use_residual: Whether to use residual connections.
        norm_fn: Normalization type.
    """

    features_list: list
    use_residual: bool = False
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the CNN model to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the CNN model.
        """
        x = x[..., None]  # Add channel dimension
        # Build a sequence of blocks
        for features in self.features_list:
            if self.use_residual:
                x = ResidualBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
            else:
                x = ConvBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
                x = ConvBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
                x = nn.max_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
                )
        # Global average pooling over spatial dims
        x = x.mean(axis=(1, 2))
        # Final dense layer to single scalar
        x = nn.Dense(features=1)(x)
        # Squeeze channel dim
        return jnp.squeeze(x, axis=-1)


class CNNFlowFieldModel(nn.Module):
    """A configurable CNN that outputs a flow field.

    Attributes:
        features_list: List of feature dimensions for each layer.
        use_residual: Whether to use residual connections.
        norm_fn: Normalization type.
    """

    features_list: list
    use_residual: bool = False
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the CNN model to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the CNN model.
        """
        # Build a sequence of blocks
        for features in self.features_list:
            if self.use_residual:
                x = ResidualBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
            else:
                x = ConvBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
                x = ConvBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)

        # Return a flow field
        return ConvBlock(
            features=2,
            norm_fn=self.norm_fn,
        )(x)


class CNNQEstimatorModel(nn.Module):
    """A configurable CNN that outputs Q-values.

    Attributes:
        features_list: Tuple of feature dimensions for each layer.
        norm_fn: Normalization type.
        kernel_sizes: Tuple of kernel sizes for each layer.
        strides: Tuple of strides for each layer.
        num_q_values: Number of Q-values to output.
        mlp_hidden_dim: Hidden dimension for MLP layer.
        postprocess: Tuple of postprocessing functions.
    """

    features_list: tuple = ()
    norm_fn: Literal["none", "batch", "instance", "group"] = "none"
    kernel_sizes: tuple = ()
    strides: tuple = ()
    num_q_values: int = 3
    mlp_hidden_dim: int = 512
    postprocess: tuple[Callable[[jnp.ndarray], jnp.ndarray], ...] = ()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the CNN model to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the CNN model.
        """
        # Ensure input is in the range [0, 1]
        x = x / 255.0

        # Convolutional layers
        for i, features in enumerate(self.features_list):
            x = ConvBlock(
                features=features,
                norm_fn=self.norm_fn,
                kernel_size=(self.kernel_sizes[i], self.kernel_sizes[i]),
                strides=(self.strides[i], self.strides[i]),
            )(x)

        # Flatten the output for the MLP
        x = x.reshape((x.shape[0], -1))

        # MLP layer
        x = nn.Dense(self.mlp_hidden_dim)(x)
        x = nn.relu(x)

        # Final dense layer to output Q-values
        x = nn.Dense(self.num_q_values)(x)

        # Apply post-processing functions
        for fn in self.postprocess:
            x = fn(x)
        return x


# Model builders


def _parse_use_residual(config: dict[str, Any]) -> bool:
    """Parse and validate ``use_residual`` from a model config.

    Args:
        config: Model configuration dictionary.

    Returns:
        Parsed boolean value for ``use_residual``.

    Raises:
        ValueError: If ``use_residual`` is not a boolean.
    """
    raw_use_residual = config.get("use_residual", False)
    if not isinstance(raw_use_residual, bool):
        raise ValueError(
            "use_residual must be a boolean, "
            f"got type {type(raw_use_residual).__name__}"
        )
    return raw_use_residual


def build_cnn_density_model_from_config(
    config: dict[str, Any],
) -> CNNDensityModel:
    """Build a CNNDensityModel from a configuration dictionary.

    Config format:
        features_list: List of feature counts per layer (required)
        use_residual: Whether to use residual blocks (default: False)
        norm_fn: Normalization type from NORM_REGISTRY (default: "none")

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CNNDensityModel.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "features_list" not in config:
        raise ValueError(
            "CNNDensityModel config must contain 'features_list' field."
        )

    features_list = config["features_list"]
    if not isinstance(features_list, list):
        raise ValueError(
            f"features_list must be a list, got {type(features_list).__name__}"
        )
    if not features_list:
        raise ValueError("features_list must be non-empty")
    if not all(isinstance(f, int) and f > 0 for f in features_list):
        raise ValueError(
            "features_list must contain only positive integers, "
            f"got {features_list}"
        )

    use_residual = _parse_use_residual(config)

    # Validate and get normalization
    norm_fn = cast(
        Literal["none", "batch", "instance", "group"],
        get_norm(str(config.get("norm_fn", "none"))),
    )

    return CNNDensityModel(
        features_list=features_list,
        use_residual=use_residual,
        norm_fn=norm_fn,
    )


def build_cnn_flow_field_model_from_config(
    config: dict[str, Any],
) -> CNNFlowFieldModel:
    """Build a CNNFlowFieldModel from a configuration dictionary.

    Config format:
        features_list: List of feature counts per layer (required)
        use_residual: Whether to use residual blocks (default: False)
        norm_fn: Normalization type from NORM_REGISTRY (default: "none")

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CNNFlowFieldModel.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "features_list" not in config:
        raise ValueError(
            "CNNFlowFieldModel config must contain 'features_list' field."
        )

    features_list = config["features_list"]
    if not isinstance(features_list, list):
        raise ValueError(
            f"features_list must be a list, got {type(features_list).__name__}"
        )
    if not features_list:
        raise ValueError("features_list must be non-empty")
    if not all(isinstance(f, int) and f > 0 for f in features_list):
        raise ValueError(
            "features_list must contain only positive integers, "
            f"got {features_list}"
        )

    use_residual = _parse_use_residual(config)

    # Validate and get normalization
    norm_fn = cast(
        Literal["none", "batch", "instance", "group"],
        get_norm(str(config.get("norm_fn", "none"))),
    )

    return CNNFlowFieldModel(
        features_list=features_list,
        use_residual=use_residual,
        norm_fn=norm_fn,
    )


def build_cnn_q_estimator_model_from_config(
    config: dict[str, Any],
) -> CNNQEstimatorModel:
    """Build a CNNQEstimatorModel from a configuration dictionary.

    Config format:
        features_list: Tuple of feature counts per layer (required)
        kernel_sizes: Tuple of kernel sizes, must match
            len(features_list) (required)
        strides: Tuple of strides, must match len(features_list) (required)
        num_q_values: Number of Q-values to output (required)
        norm_fn: Normalization type from NORM_REGISTRY (default: "none")
        mlp_hidden_dim: Hidden dimension for MLP layer (default: 512)
        postprocess: List of postprocess operation names (default: [])

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CNNQEstimatorModel.

    Raises:
        ValueError: If required fields are missing or invalid.
    """

    # Validate features_list
    if "features_list" not in config:
        raise ValueError(
            "CNNQEstimatorModel config must contain 'features_list' field."
        )
    features_list = config["features_list"]
    if isinstance(features_list, list):
        features_list = tuple(features_list)
    if not isinstance(features_list, tuple):
        raise ValueError(
            "features_list must be a tuple or list, got "
            f"{type(features_list).__name__}"
        )
    if not features_list:
        raise ValueError("features_list must be non-empty")
    if not all(isinstance(f, int) and f > 0 for f in features_list):
        raise ValueError(
            "features_list must contain only positive integers, "
            f"got {features_list}"
        )

    # Validate kernel_sizes
    if "kernel_sizes" not in config:
        raise ValueError(
            "CNNQEstimatorModel config must contain 'kernel_sizes' field."
        )
    kernel_sizes = config["kernel_sizes"]
    if isinstance(kernel_sizes, list):
        kernel_sizes = tuple(kernel_sizes)
    if not isinstance(kernel_sizes, tuple):
        raise ValueError(
            "kernel_sizes must be a tuple or list, got "
            f"{type(kernel_sizes).__name__}"
        )
    if not all(isinstance(k, int) and k > 0 for k in kernel_sizes):
        raise ValueError(
            "kernel_sizes must contain only positive integers, "
            f"got {kernel_sizes}"
        )

    # Validate strides
    if "strides" not in config:
        raise ValueError(
            "CNNQEstimatorModel config must contain 'strides' field."
        )
    strides = config["strides"]
    if isinstance(strides, list):
        strides = tuple(strides)
    if not isinstance(strides, tuple):
        raise ValueError(
            f"strides must be a tuple or list, got {type(strides).__name__}"
        )
    if not all(isinstance(s, int) and s > 0 for s in strides):
        raise ValueError(
            f"strides must contain only positive integers, got {strides}"
        )

    # Validate lengths match
    if len(kernel_sizes) != len(features_list):
        raise ValueError(
            f"len(kernel_sizes)={len(kernel_sizes)} must equal "
            f"len(features_list)={len(features_list)}"
        )
    if len(strides) != len(features_list):
        raise ValueError(
            f"len(strides)={len(strides)} must equal "
            f"len(features_list)={len(features_list)}"
        )

    # Validate num_q_values
    if "num_q_values" not in config:
        raise ValueError(
            "CNNQEstimatorModel config must contain 'num_q_values' field."
        )
    num_q_values = int(config["num_q_values"])
    if num_q_values <= 0:
        raise ValueError(f"num_q_values must be positive, got {num_q_values}")

    # Optional fields with defaults
    norm_fn = cast(
        Literal["none", "batch", "instance", "group"],
        get_norm(str(config.get("norm_fn", "none"))),
    )
    mlp_hidden_dim = int(config.get("mlp_hidden_dim", 512))
    if mlp_hidden_dim <= 0:
        raise ValueError(
            f"mlp_hidden_dim must be positive, got {mlp_hidden_dim}"
        )

    # Build postprocess pipeline
    postprocess_names = config.get("postprocess", [])
    if not isinstance(postprocess_names, (list, tuple)):
        raise ValueError(
            "postprocess must be a list or tuple, got "
            f"{type(postprocess_names).__name__}"
        )
    postprocess = build_postprocess_pipeline(postprocess_names)

    return CNNQEstimatorModel(
        features_list=features_list,
        norm_fn=norm_fn,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_q_values=num_q_values,
        mlp_hidden_dim=mlp_hidden_dim,
        postprocess=postprocess,
    )


def _resolve_postprocess(spec: Any) -> PostProcess:
    """Accept either a list/tuple of postprocess op names or callables.

    Builder configs can carry ``postprocess`` as either:

    - a sequence of strings — names of ops registered in
      ``POSTPROCESS_REGISTRY``, which we resolve via
      ``build_postprocess_pipeline``,
    - or an already-built sequence of callables — typically because a
      caller (e.g. ``ArtOfPIVEstimator``) ran
      ``validate_and_build_postprocess`` before handing the dict to a
      builder. In that case we just freeze it into a tuple.

    Args:
        spec: The ``postprocess`` value from a builder config.

    Returns:
        A normalised ``PostProcess`` tuple ready to attach to a model.

    Raises:
        ValueError: If ``spec`` is not a list/tuple, or its elements are
            a mix of strings and callables.
    """
    if not isinstance(spec, (list, tuple)):
        raise ValueError(
            f"postprocess must be a list or tuple, got {type(spec).__name__}"
        )
    if len(spec) == 0:
        return ()
    if all(isinstance(p, str) for p in spec):
        return build_postprocess_pipeline(spec)
    if all(callable(p) for p in spec):
        return tuple(spec)
    raise ValueError(
        "postprocess must be a sequence of either all strings or all "
        f"callables, got mixed types: {[type(p).__name__ for p in spec]}"
    )


class CNNEncoderModel(nn.Module):
    """A configurable CNN encoder that outputs a feature vector.

    Attributes:
        features_list: List of feature dimensions for each layer.
        output_dim: Dimension of the output feature vector.
        use_residual: Whether to use residual blocks.
        norm_fn: Normalization type.
    """

    features_list: list[int]
    output_dim: int = 64
    use_residual: bool = False
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the CNN encoder to the input tensor.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output feature vector of shape (B, output_dim).
        """
        for features in self.features_list:
            if self.use_residual:
                x = ResidualBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
            else:
                x = ConvBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
                x = ConvBlock(
                    features=features,
                    norm_fn=self.norm_fn,
                )(x)
                x = nn.max_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
                )

        # Global average pooling over spatial dims.
        x = x.mean(axis=(1, 2))

        # Final dense layer projecting to the requested feature dimension.
        x = nn.Dense(features=self.output_dim)(x)
        return x


def build_cnn_encoder_from_config(
    config: dict[str, Any],
) -> CNNEncoderModel:
    """Build a CNNEncoderModel from a configuration dictionary.

    Config format:
        features_list: List of feature counts per layer (required)
        output_dim: Output feature dimension (default: 64)
        use_residual: Whether to use residual blocks (default: False)
        norm_fn: Normalization type from NORM_REGISTRY (default: "none")

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CNNEncoderModel.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "features_list" not in config:
        raise ValueError(
            "CNNEncoderModel config must contain 'features_list' field."
        )

    features_list = config["features_list"]
    if not isinstance(features_list, (list, tuple)):
        raise ValueError(
            "features_list must be a list or tuple, "
            f"got {type(features_list).__name__}"
        )
    if isinstance(features_list, tuple):
        features_list = list(features_list)
    if not features_list:
        raise ValueError("features_list must be non-empty")
    if not all(isinstance(f, int) and f > 0 for f in features_list):
        raise ValueError(
            "features_list must contain only positive integers, "
            f"got {features_list}"
        )

    output_dim = int(config.get("output_dim", 64))
    if output_dim <= 0:
        raise ValueError(
            f"output_dim must be a positive integer, got {output_dim}"
        )

    use_residual = _parse_use_residual(config)

    norm_fn = cast(
        Literal["none", "batch", "instance", "group"],
        get_norm(str(config.get("norm_fn", "none"))),
    )

    return CNNEncoderModel(
        features_list=features_list,
        output_dim=output_dim,
        use_residual=use_residual,
        norm_fn=norm_fn,
    )


class CNNRouterModel(nn.Module):
    """A CNN router with a shared encoder and per-arm MLP heads.

    This architecture separates feature extraction from Q-value estimation:
    a shared CNNEncoderModel processes images into a feature vector and each
    arm (action) has its own MLP head that maps features to a Q-value. This
    allows per-arm specialization while sharing feature learning. When
    ``single_arm`` is True a single shared MLP outputs all Q-values at once.

    Attributes:
        encoder: Shared CNN encoder module.
        num_q_values: Number of Q-values (number of arms/actions).
        head_hidden_dims: Tuple of hidden dimensions for each MLP head.
        head_activation: Activation function for MLP heads.
        single_arm: If True, use a single shared MLP that outputs all
            Q-values at once instead of one independent MLP per arm.
            Prefer this when ``num_q_values`` is large: per-arm heads
            (``single_arm=False``) instantiate ``num_q_values`` separate
            MLP submodules inside ``@nn.compact``, so parameter count
            and trace/compile time scale linearly with the number of
            arms. The single-arm variant keeps both flat.
        postprocess: Tuple of postprocessing functions applied to the output.
    """

    encoder: CNNEncoderModel
    num_q_values: int
    head_hidden_dims: tuple[int, ...]
    head_activation: Callable = nn.relu
    single_arm: bool = False
    postprocess: tuple[Callable[[jnp.ndarray], jnp.ndarray], ...] = ()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the router model to an input batch.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Q-values of shape (B, num_q_values).
        """
        # Match CNNQEstimatorModel normalization.
        x = x / 255.0

        # Shared encoder: (B, H, W, C) -> (B, encoder.output_dim).
        features = self.encoder(x)

        if self.single_arm:
            head = MLP(
                hidden_dims=self.head_hidden_dims,
                output_dim=self.num_q_values,
                activation=self.head_activation,
                use_bias=True,
                name="head",
            )
            x = head(features)
        else:
            q_values = []
            for i in range(self.num_q_values):
                head = MLP(
                    hidden_dims=self.head_hidden_dims,
                    output_dim=1,
                    activation=self.head_activation,
                    use_bias=True,
                    name=f"head_{i}",
                )
                q_values.append(head(features))
            x = jnp.concatenate(q_values, axis=-1)

        for fn in self.postprocess:
            x = fn(x)

        return x


def build_cnn_router_model_from_config(
    config: dict[str, Any],
) -> CNNRouterModel:
    """Build a CNNRouterModel from a configuration dictionary.

    Config format:
        encoder: Dict with encoder configuration (required)
            - features_list: List of feature counts per layer
            - output_dim: Output feature dimension (default: 64)
            - use_residual: Whether to use residual blocks (default: False)
            - norm_fn: Normalization type (default: "none")
        head: Dict with head configuration (required)
            - hidden_dims: List of hidden layer dimensions
            - activation: Activation function name (default: "relu")
            - single_arm: Use one shared MLP instead of per-arm MLPs
              (default: False)
        num_q_values: Number of Q-values to output (required)
        postprocess: List of postprocess operation names (default: [])

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CNNRouterModel.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "encoder" not in config:
        raise ValueError(
            "CNNRouterModel config must contain 'encoder' field with "
            "encoder configuration."
        )
    if "head" not in config:
        raise ValueError(
            "CNNRouterModel config must contain 'head' field with "
            "head configuration."
        )
    if "num_q_values" not in config:
        raise ValueError(
            "CNNRouterModel config must contain 'num_q_values' field."
        )

    encoder_config = config["encoder"]
    if not isinstance(encoder_config, dict):
        raise ValueError(
            "encoder config must be a dict, "
            f"got {type(encoder_config).__name__}"
        )

    head_config = config["head"]
    if not isinstance(head_config, dict):
        raise ValueError(
            f"head config must be a dict, got {type(head_config).__name__}"
        )

    encoder = build_cnn_encoder_from_config(encoder_config)

    if "hidden_dims" not in head_config:
        raise ValueError(
            "head config must contain 'hidden_dims' field. "
            "Example: {'hidden_dims': [32, 16]}"
        )

    hidden_dims = head_config["hidden_dims"]
    if not isinstance(hidden_dims, (list, tuple)):
        raise ValueError(
            "hidden_dims must be a list or tuple, "
            f"got {type(hidden_dims).__name__}"
        )
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must be non-empty")
    if not all(isinstance(d, int) and d > 0 for d in hidden_dims):
        raise ValueError(
            "hidden_dims must contain only positive integers, "
            f"got {hidden_dims}"
        )
    head_hidden_dims = tuple(hidden_dims)

    activation_name = head_config.get("activation", "relu")
    if not isinstance(activation_name, str):
        raise ValueError(
            f"activation must be a string, got {type(activation_name).__name__}"
        )
    head_activation = get_activation(activation_name)

    single_arm = head_config.get("single_arm", False)
    if not isinstance(single_arm, bool):
        raise ValueError(
            "head single_arm must be a boolean, "
            f"got {type(single_arm).__name__}"
        )

    num_q_values = int(config["num_q_values"])
    if num_q_values <= 0:
        raise ValueError(f"num_q_values must be positive, got {num_q_values}")

    postprocess = _resolve_postprocess(config.get("postprocess", []))

    return CNNRouterModel(
        encoder=encoder,
        num_q_values=num_q_values,
        head_hidden_dims=head_hidden_dims,
        head_activation=head_activation,
        single_arm=single_arm,
        postprocess=postprocess,
    )


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "cnn_density": build_cnn_density_model_from_config,
    "cnn_flow_field": build_cnn_flow_field_model_from_config,
    "cnn_q_estimator": build_cnn_q_estimator_model_from_config,
    "cnn_encoder": build_cnn_encoder_from_config,
    "cnn_router": build_cnn_router_model_from_config,
}


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    """Build a neural network model from a configuration dictionary.

    This is a dispatcher that routes to the appropriate model builder
    based on the "type" field in the config.

    Config format:
        type: Model type (e.g., "cnn_density", "cnn_flow_field",
            "cnn_q_estimator")
        ... other fields specific to the model type

    Args:
        config: Configuration dictionary.

    Returns:
        Configured model module.

    Raises:
        ValueError: If type is missing or unknown.
    """
    if "type" not in config:
        raise ValueError("Model config must contain 'type' field.")

    model_type = str(config["type"]).lower()
    builder = MODEL_REGISTRY.get(model_type)

    if builder is None:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {available}"
        )

    return builder(config)
