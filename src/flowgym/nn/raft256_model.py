"""Module implementing the RAFT256 model using Flax.

This is the 1/8-resolution variant of :class:`RaftEstimatorModel`. The feature
and context encoders downsample the input by a factor of eight (stride-2 in
each of the three residual stages), the recurrent refinement runs at that
1/8 resolution, and a learned convex upsampling head reconstructs the
full-resolution flow at every iteration. Every shared building block is reused
from :mod:`flowgym.nn.blocks`, so the parameter tree is identical to RAFT32 and
the same PyTorch -> Flax converter applies.
"""

from typing import cast

import jax
import jax.numpy as jnp
from flax import linen as nn

from flowgym.flow.raft.process import build_corr_pyramid
from flowgym.nn.blocks import EncoderBlock, ScanBody256Block, UpdateBlock

# Encoder strides that downsample the input by 1/8 across the three residual
# stages, plus the explicit padding that reproduces PyTorch's symmetric
# ``padding=1`` for the strided 3x3 convolutions.
_DOWNSAMPLE_STRIDES: tuple[int, ...] = (2, 1, 2, 1, 2, 1)
_CONV_PADDING: tuple[tuple[int, int], tuple[int, int]] = ((1, 1), (1, 1))


class RaftEstimatorModel256(nn.Module):
    """RAFT256 flow estimator model.

    Attributes:
        hidden_dim: Hidden dimension size.
        context_dim: Context dimension size.
        corr_levels: Number of correlation pyramid levels.
        corr_radius: Correlation radius for lookup.
        iters: Number of refinement iterations.
        norm_fn: Normalization type.
        dropout: Dropout rate.
        train: Whether in training mode.
    """

    hidden_dim: int
    context_dim: int
    corr_levels: int
    corr_radius: int
    iters: int
    norm_fn: str
    dropout: float = 0.0
    train: bool = False

    @nn.compact
    def __call__(
        self, images: jnp.ndarray, flow_init: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply the RAFT256 estimator model to the input images.

        Args:
            images: Input image tensor of shape (B, H, W, 2). H and W must be
                divisible by 8.
            flow_init: Initial flow tensor of shape (B, H, W, 2), at full
                resolution.

        Returns:
            Estimated optical flow of shape (iters, B, H, W, 2).
        """
        # Normalize images to [0, 1]
        images = images / 256.0

        img1, img2 = jnp.split(images, 2, axis=-1)
        B, H, W, _ = img1.shape
        Hd, Wd = H // 8, W // 8

        # Coordinate grid at 1/8 resolution.
        coords0 = self._coords_grid(B, Hd, Wd)

        # Downsample the (full-resolution) flow init to 1/8 and add it. For the
        # common zero initialisation this is a no-op; it matches the PyTorch
        # ``F.interpolate`` branch used for tiled/temporal inference.
        flow_init_lr = jax.image.resize(
            flow_init,
            (B, Hd, Wd, 2),
            method="bilinear",
            antialias=False,
        )
        coords1 = coords0 + flow_init_lr

        fmap1, fmap2 = EncoderBlock(
            output_dim=256,
            norm_fn=self.norm_fn,
            dropout=self.dropout,
            train=self.train,
            residual_strides=_DOWNSAMPLE_STRIDES,
            residual_padding=_CONV_PADDING,
        )([img1, img2])

        cnet = EncoderBlock(
            output_dim=self.hidden_dim + self.context_dim,
            norm_fn=self.norm_fn,
            dropout=self.dropout,
            train=self.train,
            residual_strides=_DOWNSAMPLE_STRIDES,
            residual_padding=_CONV_PADDING,
        )(img1)
        net, inp = jnp.split(
            cast(jnp.ndarray, cnet), [self.hidden_dim], axis=-1
        )
        net = nn.tanh(net)
        inp = nn.relu(inp)  # context

        corr_pyramid = build_corr_pyramid(fmap1, fmap2, self.corr_levels)

        update_block = UpdateBlock(
            hidden_dim=self.hidden_dim,
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
        )

        ScanBlock = nn.scan(
            nn.remat(ScanBody256Block),
            variable_broadcast="params",
            split_rngs={"params": False},
            length=self.iters,
            out_axes=0,
        )

        (_, coords1), flows = ScanBlock(
            update_block=update_block,
            coords0=coords0,
            corr_radius=self.corr_radius,
            inp=inp,
            corr_pyramid=corr_pyramid,
        )((net, coords1))

        return flows

    def _coords_grid(self, batch: int, ht: int, wd: int) -> jnp.ndarray:
        """Generate a coordinate grid.

        Args:
            batch: Batch size.
            ht: Height of the grid.
            wd: Width of the grid.

        Returns:
            Coordinate grid tensor of shape (batch, ht, wd, 2) in (x, y) order.
        """
        coords = jnp.meshgrid(jnp.arange(ht), jnp.arange(wd), indexing="ij")
        coords = jnp.stack(coords[::-1], axis=-1).astype(jnp.float32)
        coords = jnp.tile(coords[None, ...], (batch, 1, 1, 1))  # (B, H, W, 2)
        return coords
