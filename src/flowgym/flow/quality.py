"""Flow quality metrics that need no ground truth."""

import jax
import jax.numpy as jnp

from flowgym.flow.process import bilinear_interpolate


def _peak_quality_single(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    flow: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Correlation-peak quality of one image pair.

    Args:
        image1: First frame, shape (H, W).
        image2: Second frame, shape (H, W).
        flow: Flow from `image1` to `image2`, shape (H, W, 2).
        window_size: Side of the square interrogation windows.

    Returns:
        One minus the mean normalised peak, a scalar.
    """
    H, W = image1.shape
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    hx, hy = 0.5 * flow[..., 0], 0.5 * flow[..., 1]
    a = bilinear_interpolate(image1, xs - hx, ys - hy)
    b = bilinear_interpolate(image2, xs + hx, ys + hy)

    def in_bounds(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)

    valid = in_bounds(xs - hx, ys - hy) & in_bounds(xs + hx, ys + hy)

    # Interrogation windows with 50% overlap: (ny, nx, w, w).
    w = window_size
    rows = jnp.arange(0, H - w + 1, w // 2)[:, None] + jnp.arange(w)
    cols = jnp.arange(0, W - w + 1, w // 2)[:, None] + jnp.arange(w)

    def windows(img: jnp.ndarray) -> jnp.ndarray:
        return img[rows[:, None, :, None], cols[None, :, None, :]]

    A, B = windows(a), windows(b)
    A = A - A.mean((-1, -2), keepdims=True)
    B = B - B.mean((-1, -2), keepdims=True)
    energy = jnp.sqrt(jnp.sum(A**2, (-1, -2)) * jnp.sum(B**2, (-1, -2)))
    ok = (windows(valid.astype(jnp.float32)).mean((-1, -2)) > 0.9) & (
        energy > 1e-6
    )

    # Normalised circular cross-correlation, searched within +-w/4 pixels.
    corr = jnp.fft.irfft2(
        jnp.conj(jnp.fft.rfft2(A)) * jnp.fft.rfft2(B), s=(w, w)
    )
    corr = jnp.fft.fftshift(corr, axes=(-2, -1))
    r, c = w // 4, w // 2
    corr = corr[..., c - r : c + r + 1, c - r : c + r + 1]
    peak = corr.max((-1, -2)) / jnp.maximum(energy, 1e-6)

    m = ok.astype(jnp.float32)
    return 1.0 - jnp.sum(peak * m) / jnp.maximum(jnp.sum(m), 1.0)


def correlation_peak_quality(
    images1: jnp.ndarray,
    images2: jnp.ndarray,
    flow: jnp.ndarray,
    window_size: int = 16,
) -> jnp.ndarray:
    """Mean correlation-peak deficit after deforming both frames with `flow`.

    Both frames are deformed halfway toward each other along `flow`; if the
    flow is right, windows of the deformed frames match, so their normalised
    cross-correlation peak is close to 1. Windows that fall outside the image
    or contain no texture are ignored. Lower is better.

    Args:
        images1: First frames, shape (B, H, W).
        images2: Second frames, shape (B, H, W).
        flow: Flow from `images1` to `images2` in pixels, shape (B, H, W, 2),
            with channel 0 the x (column) displacement.
        window_size: Side of the square interrogation windows.

    Returns:
        One minus the mean normalised peak per sample, shape (B,).
    """
    return jax.vmap(_peak_quality_single, in_axes=(0, 0, 0, None))(
        images1.astype(jnp.float32),
        images2.astype(jnp.float32),
        flow,
        window_size,
    )
