"""Image-support and candidate-flow diagnostics on the flow grid."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from flowgym.flow.process import bilinear_interpolate


# Bilinear resize at cell centers without antialiasing, like cv2.INTER_LINEAR.
def _resize(field: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    return jax.image.resize(
        field.astype(jnp.float32), shape, "linear", antialias=False
    )


@partial(jax.jit, static_argnames=("target_shape", "sigma", "border"))
def particle_support_mask(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    target_shape: tuple[int, int] | None = None,
    threshold: float = 12.0,
    sigma: float = 5.0,
    border: int = 0,
) -> jnp.ndarray:
    """Fraction of supported pixels around each flow-grid cell, in [0, 1].

    A pixel is supported if the Gaussian-blurred mean of the pair exceeds
    `threshold` and it is at least `border` pixels from the image edge. This
    matches cv2.GaussianBlur followed by cv2.resize.

    Args:
        image1: First frame, shape (H, W).
        image2: Second frame, shape (H, W).
        target_shape: Output (height, width); defaults to (H, W).
        threshold: Intensity threshold, in the images' units.
        sigma: Gaussian standard deviation in pixels; 0 disables blurring.
        border: Pixels excluded on every side.

    Returns:
        Support of shape `target_shape`.
    """
    mean = (image1.astype(jnp.float32) + image2.astype(jnp.float32)) / 2
    if sigma:
        radius = int(4 * sigma + 0.5)
        x = jnp.arange(-radius, radius + 1)
        kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        mean = jnp.pad(mean, radius, mode="reflect")
        mean = convolve2d(mean, kernel[None], "valid", precision="highest")
        mean = convolve2d(mean, kernel[:, None], "valid", precision="highest")
    H, W = image1.shape
    rows, cols = jnp.arange(H)[:, None], jnp.arange(W)[None]
    inside = (
        (rows >= border)
        & (rows < H - border)
        & (cols >= border)
        & (cols < W - border)
    )
    return _resize((mean > threshold) & inside, target_shape or (H, W))


@partial(jax.jit, static_argnames=("window_size",))
def flow_pair_diagnostics(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    flow: jnp.ndarray,
    window_size: int = 15,
    min_variance: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """Local normalized correlation between `image1` and `image2` warped back.

    The flow is upsampled to the image grid, `image2` is sampled at
    `(x + dx, y + dy)`, and zero-mean normalized correlation is computed in
    `window_size` windows, then sampled back on the flow grid. Unlike
    `correlation_peak_quality`, there is no peak search: a correct flow gives
    correlation 1 everywhere.

    Args:
        image1: First frame, shape (H, W).
        image2: Second frame, shape (H, W).
        flow: Flow from `image1` to `image2` in image pixels, shape
            (Hf, Wf, 2), with channel 0 the x displacement.
        window_size: Side of the square correlation windows.
        min_variance: Minimum intensity variance of a valid window.

    Returns:
        `correlation` in [-1, 1] (0 where invalid) and boolean `valid`, both
        of shape (Hf, Wf). A window is invalid if it lacks texture or any of
        its warped samples falls outside the image.
    """
    H, W = image1.shape
    dx = _resize(flow[..., 0], (H, W))
    dy = _resize(flow[..., 1], (H, W))
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    # Removing the global mean keeps the float32 moments below well-conditioned.
    a = image1.astype(jnp.float32)
    b = bilinear_interpolate(image2.astype(jnp.float32), xs + dx, ys + dy)
    a, b = a - a.mean(), b - b.mean()
    inside = (
        (xs + dx >= 0)
        & (xs + dx <= W - 1)
        & (ys + dy >= 0)
        & (ys + dy <= H - 1)
    )

    box = jnp.ones((window_size, window_size))

    def window_sum(x: jnp.ndarray) -> jnp.ndarray:
        return convolve2d(x, box, "same", precision="highest")

    def mean(x: jnp.ndarray) -> jnp.ndarray:
        return window_sum(x) / window_size**2

    mean_a, mean_b = mean(a), mean(b)
    var_a = mean(a * a) - mean_a**2
    var_b = mean(b * b) - mean_b**2
    cov = mean(a * b) - mean_a * mean_b
    valid = (
        (window_sum(inside.astype(jnp.float32)) > window_size**2 - 0.5)
        & (var_a > min_variance)
        & (var_b > min_variance)
    )
    corr = jnp.where(
        valid, cov / jnp.sqrt(jnp.where(valid, var_a * var_b, 1)), 0
    )

    grid = flow.shape[:2]
    valid = _resize(valid, grid) > 1 - 1e-6
    corr = jnp.where(valid, jnp.clip(_resize(corr, grid), -1, 1), 0)
    return {"correlation": corr, "valid": valid}
