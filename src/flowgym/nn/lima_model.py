"""Flax implementation of the LIMA optical-flow model for PIV.

LIMA (lightweight image matching architecture) is a lean PWC-Net / IRR-PWC
style coarse-to-fine network specialised for particle image velocimetry:

- a six-level convolutional **encoder** (shared between the two frames) builds
  a feature pyramid (Table I of Mucignat, Zdybał & Lunati 2025);
- at each pyramid level the features are **symmetrically warped** toward the
  temporal midpoint and matched with a **local correlation** cost volume;
- a single **weight-shared decoder** of dilated convolutions (Table III)
  predicts a residual displacement that is added to the upsampled estimate
  from the coarser level (iterative residual refinement, Hur & Roth 2019).

Displacements are kept in input-image pixel units throughout; the warp
converts them to each level's resolution via the cumulative stride. This keeps
the decoder's flow input on a consistent scale across levels, which is what
makes a single shared decoder work for every pyramid level.

The decoder follows the layer table given in the paper (corr + flow -> dilated
convs -> 2-channel flow head). The paper reports ~1.6M parameters for the
original PyTorch LIMA; the tabulated architecture implemented here is ~0.93M
(search range 2). The difference is most likely an additional, non-tabulated
flow-estimator stack in the original; the padding/search-range contributions of
the 2025 paper concern exactly the dilated decoder reproduced here.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from flowgym.flow.lima.process import (
    local_correlation,
    pad_for_conv,
    symmetric_warp,
)

# Encoder channels per level (Table I). Length defines the pyramid depth.
DEFAULT_ENCODER_CHANNELS: tuple[int, ...] = (16, 32, 64, 96, 128, 196)
# Decoder channels and dilations (Table III).
DEFAULT_DECODER_CHANNELS: tuple[int, ...] = (128, 128, 128, 96, 64, 32)
DEFAULT_DECODER_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 1)


class LimaEncoder(nn.Module):
    """Convolutional feature pyramid encoder.

    Each level is a single 3x3 stride-2 convolution followed by a LeakyReLU
    activation, halving the spatial resolution. The same encoder is applied to
    both input frames (shared weights).

    Attributes:
        channels: Output channels per pyramid level (fine to coarse).
        padding_mode: Padding scheme for the convolutions.
        activation_slope: Negative slope of the LeakyReLU activation.
    """

    channels: tuple[int, ...]
    padding_mode: str = "replicate"
    activation_slope: float = 0.1

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> list[jnp.ndarray]:
        """Encode an image into a multi-level feature pyramid.

        Args:
            image: Input image of shape (B, H, W, 1).

        Returns:
            Feature maps ordered from finest (level 1, stride 2) to coarsest
            (level L, stride 2**L).
        """
        feats = []
        x = image
        for channels in self.channels:
            x = pad_for_conv(x, dilation=1, kernel=3, mode=self.padding_mode)
            x = nn.Conv(channels, (3, 3), strides=(2, 2), padding="VALID")(x)
            x = jax.nn.leaky_relu(x, self.activation_slope)
            feats.append(x)
        return feats


class LimaDecoder(nn.Module):
    """Weight-shared dilated-convolution flow decoder (Table III).

    Maps a ``[cost_volume, upsampled_flow]`` tensor to a residual displacement
    field at the same resolution. Dilations grow the receptive field while
    padding keeps the output size constant. A final linear 3x3 head outputs the
    two displacement components.

    Attributes:
        channels: Output channels per decoder layer.
        dilations: Dilation rate per decoder layer.
        padding_mode: Padding scheme for the convolutions.
        activation_slope: Negative slope of the LeakyReLU activation.
    """

    channels: tuple[int, ...]
    dilations: tuple[int, ...]
    padding_mode: str = "replicate"
    activation_slope: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Decode a cost-volume/flow tensor into a residual displacement.

        Args:
            x: Tensor of shape (B, H, W, (2R+1)**2 + 2).

        Returns:
            Residual displacement field of shape (B, H, W, 2).
        """
        for channels, dilation in zip(
            self.channels, self.dilations, strict=True
        ):
            x = pad_for_conv(x, dilation, kernel=3, mode=self.padding_mode)
            x = nn.Conv(
                channels,
                (3, 3),
                kernel_dilation=(dilation, dilation),
                padding="VALID",
            )(x)
            x = jax.nn.leaky_relu(x, self.activation_slope)
        x = pad_for_conv(x, dilation=1, kernel=3, mode=self.padding_mode)
        return nn.Conv(2, (3, 3), padding="VALID")(x)


class LimaModel(nn.Module):
    """LIMA flow estimator model.

    Attributes:
        encoder_channels: Output channels per encoder level (Table I).
        decoder_channels: Output channels per decoder layer (Table III).
        decoder_dilations: Dilation per decoder layer (Table III).
        search_range: Local-correlation search range R (``(2R+1)**2`` channels).
        refine_levels: Number of coarsest pyramid levels to refine over
            (``len(encoder_channels)`` = LIMA-6; fewer = faster, coarser).
        padding_mode: Padding scheme (``zeros`` = LIMA0, ``replicate`` = LIMAR).
        activation_slope: Negative slope of the LeakyReLU activations.
    """

    encoder_channels: tuple[int, ...] = DEFAULT_ENCODER_CHANNELS
    decoder_channels: tuple[int, ...] = DEFAULT_DECODER_CHANNELS
    decoder_dilations: tuple[int, ...] = DEFAULT_DECODER_DILATIONS
    search_range: int = 2
    refine_levels: int = 6
    padding_mode: str = "replicate"
    activation_slope: float = 0.1

    @nn.compact
    def __call__(
        self, images: jnp.ndarray, flow_init: jnp.ndarray | None = None
    ) -> list[jnp.ndarray]:
        """Estimate the displacement field by iterative residual refinement.

        Args:
            images: Image pair of shape (B, H, W, 2) (frames stacked on the
                last axis).
            flow_init: Optional initial full-resolution displacement of shape
                (B, H, W, 2) in input-pixel units (temporal warm start).

        Returns:
            Per-level displacement fields in input-pixel units, ordered from
            the coarsest refined level to the finest. The last element is the
            finest estimate.
        """
        images = images / 256.0
        img1, img2 = jnp.split(images, 2, axis=-1)

        encoder = LimaEncoder(
            channels=self.encoder_channels,
            padding_mode=self.padding_mode,
            activation_slope=self.activation_slope,
        )
        feats1 = encoder(img1)
        feats2 = encoder(img2)

        decoder = LimaDecoder(
            channels=self.decoder_channels,
            dilations=self.decoder_dilations,
            padding_mode=self.padding_mode,
            activation_slope=self.activation_slope,
        )

        num_levels = len(self.encoder_channels)
        coarsest = num_levels - 1
        finest = num_levels - self.refine_levels

        flows: list[jnp.ndarray] = []
        flow: jnp.ndarray | None = None
        for idx in range(coarsest, finest - 1, -1):
            stride = 2 ** (idx + 1)
            feat1, feat2 = feats1[idx], feats2[idx]
            B, level_h, level_w, _ = feat1.shape

            if flow is None:
                if flow_init is not None:
                    flow = jax.image.resize(
                        flow_init, (B, level_h, level_w, 2), method="bilinear"
                    )
                else:
                    flow = jnp.zeros((B, level_h, level_w, 2), feat1.dtype)
            else:
                flow = jax.image.resize(
                    flow, (B, level_h, level_w, 2), method="bilinear"
                )
            flow = jnp.asarray(flow)

            warped1, warped2 = symmetric_warp(feat1, feat2, flow, stride)
            cost = local_correlation(
                warped1, warped2, self.search_range, self.padding_mode
            )
            flow = flow + decoder(jnp.concatenate([cost, flow], axis=-1))
            flows.append(flow)

        return flows
