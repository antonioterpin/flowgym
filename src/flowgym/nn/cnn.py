"""Module implementing a CNN model using Flax."""

import jax.numpy as jnp
from flax import linen as nn
from flowgym.nn.blocks import ConvBlock, ResidualBlock


class CNNDensityModel(nn.Module):
    """A configurable CNN that outputs a single scalar per example."""

    features_list: list
    use_residual: bool = False
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x):
        """Apply the CNN model to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            x: Output tensor after applying the CNN model.
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
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        # Global average pooling over spatial dims
        x = x.mean(axis=(1, 2))
        # Final dense layer to single scalar
        x = nn.Dense(features=1)(x)
        # Squeeze channel dim
        return jnp.squeeze(x, axis=-1)


class CNNFlowFieldModel(nn.Module):
    """A configurable CNN that outputs a single scalar per example."""

    features_list: list
    use_residual: bool = False
    norm_fn: str = "none"

    @nn.compact
    def __call__(self, x):
        """Apply the CNN model to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            x: Output tensor after applying the CNN model.
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
        