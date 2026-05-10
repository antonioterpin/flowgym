"""Convolutional blocks for building neural networks."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.image import resize
from jax.typing import ArrayLike

from flowgym.nn.registry import get_activation, get_norm

PostProcess = tuple[Callable[[ArrayLike], jax.Array], ...]


class ConvBlock(nn.Module):
    """A single convolutional block.

    Attributes:
        features: Number of output channels.
        kernel_size: Size of convolutional kernel.
        strides: Stride for convolution.
        norm_fn: Normalization type ("batch", "instance", "group", "none").
        activation: Activation function to apply.
        group_size: Group size for group normalization.
    """

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    norm_fn: str = "none"  # 'batch', 'instance', 'group', 'none'
    activation: Callable | None = nn.relu
    group_size: int = 8  # for group norm

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the convolutional block to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the convolutional block.

        Raises:
            ValueError: If `norm_fn` is not a supported normalization type.
        """
        out_dtype = x.dtype
        if x.dtype == jnp.float32:
            x = x.astype(jnp.float16)
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            dtype=x.dtype,
        )(x)

        if self.norm_fn == "group":
            C = x.shape[-1]
            num_groups = max(1, C // self.group_size)
            x = nn.GroupNorm(num_groups=num_groups, dtype=x.dtype)(x)
        elif self.norm_fn == "batch":
            x = nn.BatchNorm(use_running_average=False, dtype=x.dtype)(x)
        elif self.norm_fn == "instance":
            x = cast(jnp.ndarray, ClampedInstanceNorm()(x))
        # 'none' means no normalization
        elif self.norm_fn == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {self.norm_fn}")

        if self.activation:
            x = self.activation(x)
        if x.dtype != out_dtype:
            x = x.astype(out_dtype)
        return x


class ResidualBlock(nn.Module):
    """A residual block with two ConvBlocks and skip connection.

    Attributes:
        features: Number of output channels.
        kernel_size: Size of convolutional kernel.
        strides: Stride for convolution.
        norm_fn: Normalization type.
    """

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the residual block to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the residual block.
        """
        out_dtype = x.dtype
        if x.dtype == jnp.float32:
            x = x.astype(jnp.float16)
        residual = x
        # First conv
        x = ConvBlock(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            norm_fn=self.norm_fn,
        )(x)
        # Second conv
        x = ConvBlock(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            norm_fn=self.norm_fn,
        )(x)
        # Adjust channels of residual
        residual = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=self.strides,
            padding="SAME",
            dtype=x.dtype,
        )(residual)
        x = x + residual
        if x.dtype != out_dtype:
            x = x.astype(jnp.float32)
        return nn.relu(x)


class ClampedInstanceNorm(nn.Module):
    """Instance normalization with variance clamping for stability.

    Attributes:
        eps: Epsilon for numerical stability.
        var_threshold: Variance threshold for masking.
        use_scale: Whether to use learnable scale parameter.
        use_bias: Whether to use learnable bias parameter.
    """

    eps: float = 1e-5
    var_threshold: float = 1e-8
    use_scale: bool = False
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply clamped instance normalization to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor with clamping applied.
        """
        norm = nn.InstanceNorm(
            epsilon=self.eps,
            use_scale=self.use_scale,
            use_bias=self.use_bias,
            dtype=x.dtype,
        )
        y = norm(x)

        # Recompute variance for masking logic
        # TODO: optimize to avoid double computation
        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)

        # If affine, extract bias so we can mask only normalized part
        if self.use_bias:
            params = cast(Mapping[str, Any], self.variables.get("params", {}))
            try:
                bias = cast(jnp.ndarray, params["bias"])
            except KeyError:
                bias = cast(
                    jnp.ndarray,
                    cast(Mapping[str, Any], params["InstanceNorm_0"])["bias"],
                )
            # y = normalized * scale + bias
            # Mask only the normalized component, keep bias
            y = jnp.where(var < self.var_threshold, bias, y)
        else:
            y = jnp.where(var < self.var_threshold, 0.0, y)
        return cast(jnp.ndarray, y)


class UpsampleBlock(nn.Module):
    """An upsample block with bilinear interpolation and a ConvBlock.

    Attributes:
        features: Number of output channels.
        kernel_size: Size of convolutional kernel.
        scale: Upsampling scale factor.
        use_bn: Whether to use batch normalization.
        activation: Activation function to apply.
    """

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    scale: int = 2
    use_bn: bool = False
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the upsample block to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the upsample block.
        """
        # Bilinear upsample
        bs, h, w, c = x.shape
        new_size = (h * self.scale, w * self.scale)
        x = resize(x, shape=(bs, *new_size, c), method="bilinear")
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding="SAME",
        )(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=False)(x)
        if self.activation:
            x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block for processing inputs into feature maps.

    Attributes:
        output_dim: Number of output channels.
        norm_fn: Normalization type.
        dropout: Dropout rate.
        train: Whether the module is in training mode.
    """

    output_dim: int
    norm_fn: str = "none"
    dropout: float = 0.0
    train: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...]
    ) -> jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...]:
        """Apply the encoder block to the input tensor.

        Args:
            x: Input tensor or list/tuple of tensors.

        Returns:
            Output tensor or split output tensors after applying the encoder.
        """
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = jnp.concatenate(x, axis=0)

        x_array = cast(jnp.ndarray, x)
        x_array = ConvBlock(
            features=64,
            kernel_size=(7, 7),
            strides=(1, 1),
            norm_fn=self.norm_fn,
        )(x_array)
        x_array = ResidualBlock(features=64, norm_fn=self.norm_fn)(x_array)
        x_array = ResidualBlock(features=64, norm_fn=self.norm_fn)(x_array)
        x_array = ResidualBlock(features=96, norm_fn=self.norm_fn)(x_array)
        x_array = ResidualBlock(features=96, norm_fn=self.norm_fn)(x_array)
        x_array = ResidualBlock(features=128, norm_fn=self.norm_fn)(x_array)
        x_array = ResidualBlock(features=128, norm_fn=self.norm_fn)(x_array)
        out = ConvBlock(
            features=self.output_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            norm_fn="none",
            activation=None,
        )(x_array)
        out = nn.Dropout(rate=self.dropout, broadcast_dims=(1, 2))(
            out, deterministic=not self.train
        )
        if is_list:
            return jnp.split(cast(jnp.ndarray, out), [batch_dim], axis=0)

        return out


class FlowHeadBlock(nn.Module):
    """Flow head that predicts optical flow from features.

    Attributes:
        hidden_dim: Hidden dimension size.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the flow head block to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the flow head block.
        """
        x = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            norm_fn="none",
        )(x)
        x = ConvBlock(
            features=2,
            kernel_size=(3, 3),
            strides=(1, 1),
            norm_fn="none",
            activation=None,
        )(x)
        return x


class MotionEncoderBlock(nn.Module):
    """Motion encoder that processes flow and correlation features."""

    @nn.compact
    def __call__(self, flow: jnp.ndarray, corr: jnp.ndarray) -> jnp.ndarray:
        """Apply the motion encoder block to the input tensors.

        Args:
            flow: Flow tensor.
            corr: Correlation tensor.

        Returns:
            Output tensor after applying the motion encoder block.
        """
        cor = ConvBlock(features=256, kernel_size=(1, 1), strides=(1, 1))(corr)
        cor = ConvBlock(features=192, kernel_size=(3, 3), strides=(1, 1))(cor)
        flo = ConvBlock(features=128, kernel_size=(7, 7), strides=(1, 1))(flow)
        flo = ConvBlock(features=64, kernel_size=(3, 3), strides=(1, 1))(flo)

        cor_flo = jnp.concatenate([cor, flo], axis=-1)
        out = ConvBlock(features=126, kernel_size=(3, 3), strides=(1, 1))(
            cor_flo
        )
        return jnp.concatenate([out, flow], axis=-1)


class SepConvGRUBlock(nn.Module):
    """A separable convolutional GRU block.

    Attributes:
        hidden_dim: Hidden dimension size.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the separable convolutional GRU block to inputs.

        Args:
            h: Hidden state tensor.
            x: Input tensor.

        Returns:
            Updated hidden state tensor.
        """
        # horizontal
        hx = jnp.concatenate([h, x], axis=-1)
        z = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(1, 5),
            strides=(1, 1),
            activation=nn.sigmoid,
        )(hx)
        r = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(1, 5),
            strides=(1, 1),
            activation=nn.sigmoid,
        )(hx)
        rhx = jnp.concatenate([r * h, x], axis=-1)
        q = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(1, 5),
            strides=(1, 1),
            activation=nn.tanh,
        )(rhx)
        h = (1 - z) * h + z * q

        # vertical
        hx = jnp.concatenate([h, x], axis=-1)
        z = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(5, 1),
            strides=(1, 1),
            activation=nn.sigmoid,
        )(hx)
        r = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(5, 1),
            strides=(1, 1),
            activation=nn.sigmoid,
        )(hx)
        rhx = jnp.concatenate([r * h, x], axis=-1)
        q = ConvBlock(
            features=self.hidden_dim,
            kernel_size=(5, 1),
            strides=(1, 1),
            activation=nn.tanh,
        )(rhx)
        h = (1 - z) * h + z * q

        return h


class UpdateBlock(nn.Module):
    """Update block that updates the hidden state and predicts flow.

    Attributes:
        hidden_dim: Hidden dimension size.
        corr_levels: Number of correlation pyramid levels.
        corr_radius: Correlation radius for lookup.
    """

    hidden_dim: int
    corr_levels: int
    corr_radius: int

    @nn.compact
    def __call__(
        self,
        net: jnp.ndarray,
        inp: jnp.ndarray,
        corr: jnp.ndarray,
        flow: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply the update block to the input tensors.

        Args:
            net: Hidden state tensor.
            inp: Input tensor.
            corr: Correlation tensor.
            flow: Flow tensor.

        Returns:
            A tuple of `(updated hidden state, mask, predicted flow)`.
        """
        encoder = MotionEncoderBlock()
        gru = SepConvGRUBlock(hidden_dim=self.hidden_dim)
        flow_head = FlowHeadBlock(hidden_dim=256)

        motion_features = encoder(flow, corr)
        inp = jnp.concatenate([inp, motion_features], axis=-1)
        net = gru(net, inp)
        delta_flow = flow_head(net)

        # scale mask to balance gradients
        mask = ConvBlock(features=256, kernel_size=(3, 3), strides=(1, 1))(net)
        mask = 0.25 * ConvBlock(
            features=64 * 9, kernel_size=(1, 1), strides=(1, 1), activation=None
        )(mask)

        return net, mask, delta_flow


class ScanBodyBlock(nn.Module):
    """Scan body for iterative flow refinement.

    Attributes:
        update_block: Update block module for flow refinement.
        coords0: Initial coordinate grid.
        corr_radius: Correlation radius for lookup.
        inp: Input features.
        corr_pyramid: Correlation pyramid.
    """

    update_block: nn.Module
    coords0: jnp.ndarray
    corr_radius: int
    inp: jnp.ndarray
    corr_pyramid: list

    @nn.compact
    def __call__(
        self,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Apply the scan body to iteratively refine the flow.

        Args:
            carry: Tuple containing hidden state and coordinates tensors.

        Returns:
            A tuple containing `(updated carry, current flow)`.
        """
        # Import here to avoid circular dependency
        from flowgym.flow.raft.process import correlation_block  # noqa: PLC0415

        net, coords1 = carry

        # Detach gradients to prevent backprop through time
        coords1 = jax.lax.stop_gradient(coords1)

        # Compute correlation features
        corr = correlation_block(self.corr_pyramid, coords1, self.corr_radius)
        # Compute flow as difference between coordinates
        flow = coords1 - self.coords0
        # Update hidden state, mask, and delta flow
        net, _, delta_flow = self.update_block(net, self.inp, corr, flow)

        # Update coordinates with predicted flow
        coords1 = coords1 + delta_flow

        # Compute current flow
        flow = coords1 - self.coords0

        return (net, coords1), flow


POSTPROCESS_REGISTRY: dict[str, Callable[[ArrayLike], jax.Array]] = {
    "none": jax.nn.identity,
    "neg": jnp.negative,
    "softplus": nn.softplus,
    "abs": jnp.abs,
    "exp": jnp.exp,
    "symlog": lambda x: jnp.sign(x) * jnp.log1p(jnp.abs(x)),
    "log": jnp.log,
}


def build_postprocess_pipeline(names: Sequence[str]) -> PostProcess:
    """Build a postprocessing pipeline from a list of operation names.

    Args:
        names: List or tuple of operation names.

    Returns:
        Tuple of postprocessing functions.

    Raises:
        ValueError: If an unknown postprocess operation name is provided.
    """
    try:
        return tuple(POSTPROCESS_REGISTRY[name] for name in names)
    except KeyError as e:
        raise ValueError(
            f"Unknown postprocess operation '{e.args[0]}'. "
            f"Valid options: {list(POSTPROCESS_REGISTRY.keys())}"
        ) from None


# Block builders


def build_conv_block_from_config(config: dict[str, Any]) -> ConvBlock:
    """Build a ConvBlock from a configuration dictionary.

    Config format:
        features: Number of output channels (required)
        kernel_size: Kernel size as int or [H, W] (default: 3)
        strides: Stride as int or [H, W] (default: 1)
        norm_fn: Normalization type from NORM_REGISTRY (default: "none")
        activation: Activation function name from ACTIVATION_REGISTRY
            (default: "relu")
        group_size: Group size for group normalization (default: 8)

    Args:
        config: Configuration dictionary.

    Returns:
        Configured ConvBlock module.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "features" not in config:
        raise ValueError("ConvBlock config must contain 'features' field.")

    features = int(config["features"])
    if features <= 0:
        raise ValueError(f"features must be positive, got {features}")

    # Handle kernel_size: int or [H, W]
    kernel_size_cfg = config.get("kernel_size", 3)
    if isinstance(kernel_size_cfg, (list, tuple)):
        if len(kernel_size_cfg) != 2:
            raise ValueError(
                f"kernel_size must be int or [H, W], got {kernel_size_cfg}"
            )
        kernel_size = cast(
            tuple[int, int], tuple(int(k) for k in kernel_size_cfg)
        )
    else:
        k = int(kernel_size_cfg)
        kernel_size = (k, k)

    # Handle strides: int or [H, W]
    strides_cfg = config.get("strides", 1)
    if isinstance(strides_cfg, (list, tuple)):
        if len(strides_cfg) != 2:
            raise ValueError(
                f"strides must be int or [H, W], got {strides_cfg}"
            )
        strides = cast(tuple[int, int], tuple(int(s) for s in strides_cfg))
    else:
        s = int(strides_cfg)
        strides = (s, s)

    # Validate and get normalization
    norm_fn = get_norm(str(config.get("norm_fn", "none")))

    # Get activation function
    activation_name = str(config.get("activation", "relu"))
    activation = get_activation(activation_name)
    # Handle "none" activation (identity) by setting to None for ConvBlock
    if activation_name.lower() == "none":
        activation = None

    group_size = int(config.get("group_size", 8))
    if norm_fn == "group" and group_size <= 0:
        raise ValueError(
            "group_size must be positive when norm_fn='group', got "
            f"{group_size}"
        )

    return ConvBlock(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        norm_fn=norm_fn,
        activation=activation,
        group_size=group_size,
    )


def build_residual_block_from_config(config: dict[str, Any]) -> ResidualBlock:
    """Build a ResidualBlock from a configuration dictionary.

    Config format:
        features: Number of output channels (required)
        kernel_size: Kernel size as int or [H, W] (default: 3)
        strides: Stride as int or [H, W] (default: 1)
        norm_fn: Normalization type from NORM_REGISTRY (default: "none")

    Args:
        config: Configuration dictionary.

    Returns:
        Configured ResidualBlock module.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if "features" not in config:
        raise ValueError("ResidualBlock config must contain 'features' field.")

    features = int(config["features"])
    if features <= 0:
        raise ValueError(f"features must be positive, got {features}")

    # Handle kernel_size: int or [H, W]
    kernel_size_cfg = config.get("kernel_size", 3)
    if isinstance(kernel_size_cfg, (list, tuple)):
        if len(kernel_size_cfg) != 2:
            raise ValueError(
                f"kernel_size must be int or [H, W], got {kernel_size_cfg}"
            )
        kernel_size = cast(
            tuple[int, int], tuple(int(k) for k in kernel_size_cfg)
        )
    else:
        k = int(kernel_size_cfg)
        kernel_size = (k, k)

    # Handle strides: int or [H, W]
    strides_cfg = config.get("strides", 1)
    if isinstance(strides_cfg, (list, tuple)):
        if len(strides_cfg) != 2:
            raise ValueError(
                f"strides must be int or [H, W], got {strides_cfg}"
            )
        strides = cast(tuple[int, int], tuple(int(s) for s in strides_cfg))
    else:
        s = int(strides_cfg)
        strides = (s, s)

    # Validate and get normalization
    norm_fn = get_norm(str(config.get("norm_fn", "none")))

    return ResidualBlock(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        norm_fn=norm_fn,
    )


BLOCK_REGISTRY: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "conv": build_conv_block_from_config,
    "residual": build_residual_block_from_config,
}


def build_block_from_config(config: dict[str, Any]) -> nn.Module:
    """Build a neural network block from a configuration dictionary.

    This is a dispatcher that routes to the appropriate block builder
    based on the "type" field in the config.

    Config format:
        type: Block type ("conv" or "residual")
        ... other fields specific to the block type

    Args:
        config: Configuration dictionary.

    Returns:
        Configured block module.

    Raises:
        ValueError: If type is missing or unknown.
    """
    if "type" not in config:
        raise ValueError("Block config must contain 'type' field.")

    block_type = str(config["type"]).lower()
    builder = BLOCK_REGISTRY.get(block_type)

    if builder is None:
        available = ", ".join(sorted(BLOCK_REGISTRY.keys()))
        raise ValueError(
            f"Unknown block type '{block_type}'. Available: {available}"
        )

    return builder(config)
