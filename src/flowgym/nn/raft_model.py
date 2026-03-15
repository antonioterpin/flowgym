"""Module implementing the RAFT model using Flax."""

from typing import cast

import jax.numpy as jnp
from flax import linen as nn

from flowgym.flow.raft.process import build_corr_pyramid
from flowgym.nn.blocks import EncoderBlock, ScanBodyBlock, UpdateBlock


class RaftEstimatorModel(nn.Module):
    """RAFT flow estimator model.

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
        """Apply the RAFT estimator model to the input images.

        Args:
            images: Input image tensor of shape (B, H, W, 2).
            flow_init: Initial flow tensor of shape (B, H, W, 2).

        Returns:
            Estimated optical flow of shape (B, H, W, 2).
        """
        # Normalize images to [0, 1]
        images = images / 256.0

        # TODO: Check order of img1 and img2 (flipped better performance)
        img1, img2 = jnp.split(images, 2, axis=-1)
        B, H, W, _ = img1.shape
        coords0 = self._coords_grid(B, H, W)
        coords1 = coords0 + flow_init

        fmap1, fmap2 = EncoderBlock(
            output_dim=256,
            norm_fn=self.norm_fn,
            dropout=self.dropout,
            train=self.train,
        )([img1, img2])

        cnet = EncoderBlock(
            output_dim=self.hidden_dim + self.context_dim,
            norm_fn=self.norm_fn,
            dropout=self.dropout,
            train=self.train,
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
            # TODO: Check if remat helps with memory and speed (training)
            nn.remat(ScanBodyBlock),
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
            Coordinate grid tensor.
        """
        coords = jnp.meshgrid(jnp.arange(ht), jnp.arange(wd), indexing="ij")
        coords = jnp.stack(coords[::-1], axis=-1).astype(jnp.float32)
        coords = jnp.tile(coords[None, ...], (batch, 1, 1, 1))  # (B, H, W, 2)
        return coords
