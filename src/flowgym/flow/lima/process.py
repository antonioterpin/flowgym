"""Processing ops for the LIMA flow estimator (JAX).

These are the parameter-free building blocks used by the LIMA model:

- ``coords_grid`` / ``bilinear_warp`` / ``symmetric_warp`` implement the
  symmetric feature warping toward the temporal midpoint (second-order
  accurate matching, Wereley & Meinhart 2001).
- ``local_correlation`` builds the local cost volume with an integer search
  range ``R`` (``(2R+1)**2`` channels), independent of the feature channel
  count.
- ``pad_for_conv`` applies the explicit padding required to keep dilated
  convolutions size-preserving, supporting the padding schemes studied in
  Mucignat, Zdybał & Lunati (Phys. Fluids 2025).

References:
    Mucignat, Zdybał & Lunati, Phys. Fluids 37, 105112 (2025),
    https://doi.org/10.1063/5.0283779 (padding Eq. 1 and search range);
    Wereley & Meinhart, "Second-order accurate particle image velocimetry",
    Exp. Fluids 31, 258-268 (2001), https://doi.org/10.1007/s003480100281
    (symmetric midpoint warping). See the ``flowgym.flow.lima`` package
    docstring for the full reference list.
"""

import jax.numpy as jnp

# Map LIMA padding-mode names to ``jnp.pad`` modes.
_PAD_MODES: dict[str, str] = {
    "zeros": "constant",
    "replicate": "edge",
    "reflect": "reflect",
    "circular": "wrap",
}


def coords_grid(batch: int, height: int, width: int) -> jnp.ndarray:
    """Build a pixel coordinate grid in (x, y) order.

    Args:
        batch: Batch size.
        height: Grid height (number of rows).
        width: Grid width (number of columns).

    Returns:
        Coordinate grid of shape (batch, height, width, 2) where the last
        axis holds (x, y) pixel coordinates.
    """
    ys, xs = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    grid = jnp.stack([xs, ys], axis=-1).astype(jnp.float32)
    return jnp.broadcast_to(grid[None], (batch, height, width, 2))


def bilinear_warp(feat: jnp.ndarray, flow: jnp.ndarray) -> jnp.ndarray:
    """Warp a feature map by a displacement field with bilinear sampling.

    Pixel ``(x, y)`` of the output samples ``feat`` at ``(x + flow_x,
    y + flow_y)``. Out-of-bounds locations use replicate (border-clamp)
    sampling.

    Args:
        feat: Feature map of shape (B, H, W, C).
        flow: Displacement field of shape (B, H, W, 2) in (x, y) pixel units,
            expressed at the resolution of ``feat``.

    Returns:
        Warped feature map of shape (B, H, W, C).
    """
    B, H, W, _ = feat.shape
    coords = coords_grid(B, H, W) + flow
    x = coords[..., 0]
    y = coords[..., 1]

    x0 = jnp.floor(x)
    y0 = jnp.floor(y)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    x0c = jnp.clip(x0, 0, W - 1).astype(jnp.int32)
    x1c = jnp.clip(x0 + 1, 0, W - 1).astype(jnp.int32)
    y0c = jnp.clip(y0, 0, H - 1).astype(jnp.int32)
    y1c = jnp.clip(y0 + 1, 0, H - 1).astype(jnp.int32)

    b = jnp.arange(B)[:, None, None]
    top_left = feat[b, y0c, x0c, :]
    bot_left = feat[b, y1c, x0c, :]
    top_right = feat[b, y0c, x1c, :]
    bot_right = feat[b, y1c, x1c, :]

    return (
        (1 - wx) * (1 - wy) * top_left
        + (1 - wx) * wy * bot_left
        + wx * (1 - wy) * top_right
        + wx * wy * bot_right
    )


def symmetric_warp(
    feat1: jnp.ndarray,
    feat2: jnp.ndarray,
    flow: jnp.ndarray,
    stride: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Warp both feature maps toward the temporal midpoint.

    With the repo's flow convention (a particle at ``p`` in frame 1 appears at
    ``p + flow`` in frame 2, so ``feat2[p] = feat1[p - flow]``) and the
    ``bilinear_warp`` sampling rule (output ``p`` samples the input at
    ``p + offset``), bringing both frames to the temporal midpoint requires
    sampling ``feat1`` at ``-flow/2`` and ``feat2`` at ``+flow/2``. When the
    running estimate equals the true displacement both warped maps coincide
    (residual correlation peaks at zero shift), giving the central-difference,
    second-order-accurate match of Wereley & Meinhart (2001). The flow is
    supplied in input-image pixel units and converted to the feature-map
    resolution by dividing by ``stride``.

    Args:
        feat1: Feature map of the first image, shape (B, H, W, C).
        feat2: Feature map of the second image, shape (B, H, W, C).
        flow: Displacement field of shape (B, H, W, 2) in input-pixel units.
        stride: Cumulative stride of the feature map (input px per feature px).

    Returns:
        Tuple ``(warped_feat1, warped_feat2)``.
    """
    half = 0.5 * flow / stride
    return bilinear_warp(feat1, -half), bilinear_warp(feat2, half)


def local_correlation(
    feat1: jnp.ndarray,
    feat2: jnp.ndarray,
    search_range: int,
    padding_mode: str = "replicate",
) -> jnp.ndarray:
    """Build a local correlation cost volume.

    For every pixel ``(x, y)`` and every integer shift ``(dx, dy)`` with
    ``dx, dy in [-R, R]``, the channel response is the mean over feature
    channels of ``feat1(x, y) * feat2(x + dx, y + dy)``. The channels are
    ordered row-major over ``(dy, dx)``.

    Args:
        feat1: First (warped) feature map, shape (B, H, W, C).
        feat2: Second (warped) feature map, shape (B, H, W, C).
        search_range: Maximum integer shift ``R`` in each direction.
        padding_mode: Border handling for shifts that leave the image
            (one of ``zeros``, ``replicate``, ``reflect``, ``circular``).

    Returns:
        Cost volume of shape (B, H, W, (2R+1)**2).
    """
    _, H, W, _ = feat1.shape
    R = search_range
    padded = jnp.pad(
        feat2,
        ((0, 0), (R, R), (R, R), (0, 0)),
        mode=_PAD_MODES[padding_mode],
    )

    responses = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            shifted = padded[:, R + dy : R + dy + H, R + dx : R + dx + W, :]
            responses.append(jnp.mean(feat1 * shifted, axis=-1))
    return jnp.stack(responses, axis=-1)


def pad_for_conv(
    x: jnp.ndarray,
    dilation: int,
    kernel: int = 3,
    mode: str = "replicate",
) -> jnp.ndarray:
    """Pad spatial dims so a dilated VALID convolution preserves size.

    The padding width is ``floor(dilation * (kernel - 1) / 2)`` per side
    (Eq. 1 of Mucignat, Zdybał & Lunati 2025), which for a 3x3 kernel equals
    ``dilation``.

    Args:
        x: Input tensor of shape (B, H, W, C).
        dilation: Convolution dilation rate.
        kernel: Convolution kernel size (assumed square).
        mode: Padding scheme (one of ``zeros``, ``replicate``, ``reflect``,
            ``circular``).

    Returns:
        Padded tensor of shape (B, H + 2p, W + 2p, C).
    """
    p = dilation * (kernel - 1) // 2
    return jnp.pad(
        x,
        ((0, 0), (p, p), (p, p), (0, 0)),
        mode=_PAD_MODES[mode],
    )
