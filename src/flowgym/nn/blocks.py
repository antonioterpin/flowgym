"""Convolutional blocks for building neural networks."""

from collections.abc import Callable
import jax.numpy as jnp
import jax
from jax.image import resize
from flax import linen as nn


class ConvBlock(nn.Module):
    """A single convolutional block."""

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    norm_fn: str = "none"  # 'batch', 'instance', 'group', 'none'
    activation: Callable | None = nn.relu
    group_size: int = 8  # for group norm

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Apply the convolutional block to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the convolutional block.
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
            x = ClampedInstanceNorm()(x)
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
    """A residual block with two ConvBlocks and skip connection."""

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
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
    """Instance Normalization with variance clamping for stability."""

    eps: float = 1e-5
    var_threshold: float = 1e-8
    use_scale: bool = False
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        """Apply clamped instance normalization to the input tensor."""
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
            bias = self.variables["params"]["bias"]
            # y = normalized * scale + bias
            # Mask only the normalized component, keep bias
            y = jnp.where(var < self.var_threshold, bias, y)
        else:
            y = jnp.where(var < self.var_threshold, 0.0, y)
        return y


class UpsampleBlock(nn.Module):
    """An upsample block with bilinear interpolation and a ConvBlock."""

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
    """Encoder block that processes input images into feature representations."""

    output_dim: int
    norm_fn: str = "none"
    dropout: float = 0.0
    train: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the encoder block to the input tensor.

        Args:
            x (jnp.ndarray or list of jnp.ndarray): Input tensor or list of tensors.

        Returns:
            x (jnp.ndarray or list of jnp.ndarray):
                Output tensor or list of tensors after applying the encoder block.
        """
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = jnp.concatenate(x, axis=0)

        x = ConvBlock(
            features=64, kernel_size=(7, 7), strides=(1, 1), norm_fn=self.norm_fn
        )(x)
        x = ResidualBlock(features=64, norm_fn=self.norm_fn)(x)
        x = ResidualBlock(features=64, norm_fn=self.norm_fn)(x)
        x = ResidualBlock(features=96, norm_fn=self.norm_fn)(x)
        x = ResidualBlock(features=96, norm_fn=self.norm_fn)(x)
        x = ResidualBlock(features=128, norm_fn=self.norm_fn)(x)
        x = ResidualBlock(features=128, norm_fn=self.norm_fn)(x)
        x = ConvBlock(
            features=self.output_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            norm_fn="none",
            activation=None,
        )(x)
        x = nn.Dropout(rate=self.dropout, broadcast_dims=(1, 2))(
            x, deterministic=not self.train
        )
        if is_list:
            x = jnp.split(x, [batch_dim], axis=0)

        return x


class FlowHeadBlock(nn.Module):
    """Flow head that predicts optical flow from features."""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the flow head block to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            x (jnp.ndarray): Output tensor after applying the flow head block.
        """
        x = ConvBlock(
            features=self.hidden_dim, kernel_size=(3, 3), strides=(1, 1), norm_fn="none"
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
            flow (jnp.ndarray): Flow tensor.
            corr (jnp.ndarray): Correlation tensor.

        Returns:
            jnp.ndarray: Output tensor after applying the motion encoder block.
        """
        cor = ConvBlock(features=256, kernel_size=(1, 1), strides=(1, 1))(corr)
        cor = ConvBlock(features=192, kernel_size=(3, 3), strides=(1, 1))(cor)
        flo = ConvBlock(features=128, kernel_size=(7, 7), strides=(1, 1))(flow)
        flo = ConvBlock(features=64, kernel_size=(3, 3), strides=(1, 1))(flo)

        cor_flo = jnp.concatenate([cor, flo], axis=-1)
        out = ConvBlock(features=126, kernel_size=(3, 3), strides=(1, 1))(cor_flo)
        return jnp.concatenate([out, flow], axis=-1)


class SepConvGRUBlock(nn.Module):
    """A separable convolutional GRU block."""

    hidden_dim: int

    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the separable convolutional GRU block to the input tensors.

        Args:
            h (jnp.ndarray): Hidden state tensor.
            x (jnp.ndarray): Input tensor.

        Returns:
            h (jnp.ndarray):
                Output tensor after applying the separable convolutional GRU block.
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
    """Update block that updates the hidden state and predicts flow."""

    hidden_dim: int
    corr_levels: int
    corr_radius: int

    @nn.compact
    def __call__(
        self, net: jnp.ndarray, inp: jnp.ndarray, corr: jnp.ndarray, flow: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply the update block to the input tensors.

        Args:
            net (jnp.ndarray): Hidden state tensor.
            inp (jnp.ndarray): Input tensor.
            corr (jnp.ndarray): Correlation tensor.
            flow (jnp.ndarray): Flow tensor.

        Returns:
            net (jnp.ndarray): Updated hidden state tensor.
            mask (jnp.ndarray): Mask tensor.
            delta_flow (jnp.ndarray): Predicted flow tensor.
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
    """Scan body for iterative flow refinement."""

    update_block: nn.Module
    coords0: jnp.ndarray
    corr_radius: int
    inp: jnp.ndarray
    corr_pyramid: list

    @nn.compact
    def __call__(
        self,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the scan body to iteratively refine the flow.

        Args:
            carry (tuple[jnp.ndarray, jnp.ndarray]):
                A tuple containing the hidden state tensor and the coordinates tensor.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                A tuple containing the updated hidden state tensor
                and the updated coordinates tensor.
        """
        from flowgym.flow.raft.process import correlation_block
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
