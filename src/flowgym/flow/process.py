"""Module for PIV images processing in JAX."""

import jax
import jax.numpy as jnp


def img_resize(image: jnp.ndarray, new_shape: tuple[int, int]) -> jnp.ndarray:
    """Resize an image to a new shape using bilinear interpolation.

    Args:
        image: (H, W) or (H, W, C) image. The channel must be the last dimension, resize will happen on H and W.
        new_shape: (new_H, new_W) target shape.

    Returns:
        Resized image of shape (new_H, new_W) or (new_H, new_W, C).
    """
    if image.shape == new_shape:
        return image

    H, W = image.shape[-2:]
    new_H, new_W = new_shape

    # Center-based coordinate mapping
    scale_y = H / new_H
    scale_x = W / new_W

    y = (jnp.arange(new_H) + 0.5) * scale_y - 0.5
    x = (jnp.arange(new_W) + 0.5) * scale_x - 0.5

    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)

    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)

    def gather_channel(image):
        """Gather pixels from the image at the specified coordinates.

        Note: This function works for both single-channel and multi-channel images.
        """
        top_left = image[y0[:, None], x0]
        top_right = image[y0[:, None], x1]
        bot_left = image[y1[:, None], x0]
        bot_right = image[y1[:, None], x1]

        top = (1.0 - wx) * top_left + wx * top_right
        bottom = (1.0 - wx) * bot_left + wx * bot_right
        return (1.0 - wy) * top + wy * bottom

    return gather_channel(image)


def bilinear_interpolate(
    image: jnp.ndarray, x_f: jnp.ndarray, y_f: jnp.ndarray
) -> jnp.ndarray:
    """Perform bilinear interpolation at floating-point pixel coordinates.

    NOTE: this is the same implementation as in
    synthpix.utils.bilinear_interpolate, but drops the dependency on
    synthpix.

    Args:
        image: 2D image to sample from, of shape (H, W).
        x_f: 2D array of floating-point x-coordinates
        y_f: 2D array of floating-point y-coordinates

    Returns:
        Interpolated intensities at each (y, x) location, of shape (H, W).
    """
    H, W = image.shape

    # Clamp x_f and y_f to be within the image bounds
    x_f_clamped = jnp.clip(x_f, 0.0, W - 1.0)
    y_f_clamped = jnp.clip(y_f, 0.0, H - 1.0)

    # Integer neighbors & clamping
    x0 = jnp.clip(jnp.floor(x_f).astype(jnp.int32), 0, W - 1)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.clip(jnp.floor(y_f).astype(jnp.int32), 0, H - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    # Fractional weights
    wx = x_f_clamped - x0
    wy = y_f_clamped - y0

    # Gather neighboring pixels
    I00 = image[y0, x0]
    I10 = image[y0, x1]
    I01 = image[y1, x0]
    I11 = image[y1, x1]

    # Bilinear
    return (
        (1 - wx) * (1 - wy) * I00
        + wx * (1 - wy) * I10
        + (1 - wx) * wy * I01
        + wx * wy * I11
    )

def apply_flow_to_image_backward(
    image: jnp.ndarray,
    flow_field: jnp.ndarray,
    dt: float = 1.0,
) -> jnp.ndarray:
    """Warp a 2D image of particles according to a given flow field.

    For each pixel (y, x) in the output image, we compute a velocity (u, v)
    from `flow_field[y, x]`, then sample from the input image at
    (y_s, x_s) = (y - v * dt, x - u * dt) via bilinear interpolation.

    NOTE: this is the same implementation as in
    synthpix.apply.apply_flow_to_image_backward, but drops the dependency on
    synthpix.

    Args:
        image: 2D array (H, W) representing the input particle image.
        flow_field: 3D array (H, W, 2) representing the velocity field.
        dt: Time step for the backward mapping.

    Returns:
        A new 2D array of shape (H, W) with the particles displaced.
    """
    H, W = image.shape

    # Meshgrid of pixel coordinates
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

    # Real sample locations
    dx = flow_field[..., 0]
    dy = flow_field[..., 1]
    x_f = xs - dx * dt
    y_f = ys - dy * dt

    # Bilinear interpolation to sample the image at (y_f, x_f)
    return bilinear_interpolate(
        image,
        x_f,
        y_f,
    )

def apply_flow_to_image_forward(
    image: jnp.ndarray,
    flow_field: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Warp a 2D image of particles according to a given flow field using forward mapping.

    For each pixel (y, x) in the input image, we compute a velocity (u, v)
    from `flow_field[y, x]`, then deposit the pixel value at the displaced
    location (y + v * dt, x + u * dt) in the output image using bilinear splatting.

    NOTE: this is the same implementation as in
    synthpix.apply.apply_flow_to_image_forward, but drops the dependency on
    synthpix.

    Args:
        image: 2D array (H, W) representing the input particle image.
        flow_field: 3D array (H, W, 2) representing the velocity field.
        dt: Time step for the forward mapping.

    Returns:
        A new 2D array of shape (H, W) with the particles displaced using forward mapping.
    """
    H, W = image.shape
    y_grid, x_grid = jnp.indices((H, W))

    u = flow_field[..., 0]
    v = flow_field[..., 1]

    # Forward mapping: (x_d, y_d) = (x + u * dt, y + v * dt)
    x_d = x_grid + u * dt
    y_d = y_grid + v * dt

    new_image = jnp.zeros_like(image)

    def deposit_pixel(new_image, x_src, y_src, val):
        x0 = jnp.floor(x_src).astype(int)
        y0 = jnp.floor(y_src).astype(int)

        wx = x_src - x0
        wy = y_src - y0

        def in_bounds(x, y):
            return (x >= 0) & (x < W) & (y >= 0) & (y < H)

        for dx, dy, weight in [
            (0, 0, (1 - wx) * (1 - wy)),
            (1, 0, wx * (1 - wy)),
            (0, 1, (1 - wx) * wy),
            (1, 1, wx * wy),
        ]:
            xi = x0 + dx
            yi = y0 + dy
            cond = in_bounds(xi, yi)
            new_image = jax.lax.cond(
                cond,
                lambda img: img.at[yi, xi].add(val * weight),
                lambda img: img,
                operand=new_image,
            )
        return new_image

    def body_fn(i, new_image):
        y, x = divmod(i, W)
        return deposit_pixel(new_image, x_d[y, x], y_d[y, x], image[y, x])

    new_image = jax.lax.fori_loop(0, H * W, body_fn, new_image)
    return new_image