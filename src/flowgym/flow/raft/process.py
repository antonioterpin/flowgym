"""Module for RAFT flow processing in JAX."""

import jax
import jax.numpy as jnp
from flax.linen import avg_pool
from jax import lax


def build_corr_volume(fmap1: jnp.ndarray, fmap2: jnp.ndarray) -> jnp.ndarray:
    """Compute the correlation volume between two feature maps.

    Args:
        fmap1: (B, H, W, C) feature map of image 1
        fmap2: (B, H, W, C) feature map of image 2

    Returns:
        corr: (B, H, W, 1, H, W) correlation volume
    """
    B, H, W, C = fmap1.shape
    x = fmap1.reshape(B, H * W, C)  # (B, N, C)
    y = fmap2.reshape(B, H * W, C)  # (B, N, C)
    corr = jnp.matmul(x, jnp.swapaxes(y, 1, 2))  # (B, N, N)
    corr = corr.reshape(B, H, W, 1, H, W)
    corr = corr / jnp.sqrt(jnp.array(C, dtype=corr.dtype))
    return corr  # (B, H, W, 1, H, W)


def build_corr_pyramid(
    fmap1: jnp.ndarray, fmap2: jnp.ndarray, num_levels: int = 4
) -> list[jnp.ndarray]:
    """Build a correlation pyramid from two feature maps."""
    corr_pyramid = []

    # all pairs correlation
    corr_vol = build_corr_volume(fmap1, fmap2)  # (B, H1, W1, 1, H2, W2)
    B, H1, W1, _, H2, W2 = corr_vol.shape
    corr = corr_vol.reshape(B * H1 * W1, H2, W2, 1)  # (B*H1*W1, H2, W2, 1)

    corr_pyramid.append(corr)

    for _ in range(num_levels - 1):
        # average pool over spatial dimensions
        corr = avg_pool(corr, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        # TODO check if this is needed
        # corr = avg_pool2(corr)

        corr_pyramid.append(corr)

    return corr_pyramid


def bilinear_sampler(
    img: jnp.ndarray, coords: jnp.ndarray, mask: bool = False
) -> jnp.ndarray:
    """Wrapper for grid_sample, uses pixel coordinates.

    Args:
        img: (B, H, W, C) image to sample from
        coords: (B, H_out, W_out, 2) coordinates of pixels to sample
                in (x, y) order and in pixel space
        mask: if True, also returns a mask of which pixels were sampled
              inside the image

    Returns:
        sampled: (B, H_out, W_out, C) sampled image
        mask (optional): (B, H_out, W_out, 1) mask of which pixels were
                         sampled inside the image
    """
    _, H, W, _ = img.shape
    xgrid, ygrid = jnp.split(coords, [1], axis=-1)

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = jnp.concatenate([xgrid, ygrid], axis=-1)
    img = grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.astype(img.dtype)

    return img


def grid_sample(
    img: jnp.ndarray,
    grid: jnp.ndarray,
    align_corners: bool = True,
    padding_mode: str = "zeros",
):
    """Bilinear sampling of img at grid locations.

    Args:
        img: (B, H, W, C) image to sample from
        grid: (B, H_out, W_out, 2) in range [-1, 1]
        align_corners: whether to align corners
        padding_mode: 'zeros' or 'border'

    Returns:
        sampled: (B, H_out, W_out, C) sampled image
    """
    B, H, W, C = img.shape
    x = grid[..., 0]
    y = grid[..., 1]

    # scale grid to image size
    if align_corners:
        u = 0.5 * (x + 1.0) * (W - 1)
        v = 0.5 * (y + 1.0) * (H - 1)
    else:
        u = ((x + 1.0) * W - 1) / 2
        v = ((y + 1.0) * H - 1) / 2

    u0 = jnp.floor(u).astype(jnp.int16)
    v0 = jnp.floor(v).astype(jnp.int16)
    u1 = u0 + 1
    v1 = v0 + 1

    # clip for border mode
    if padding_mode == "border":
        u0 = jnp.clip(u0, 0, W - 1)
        u1 = jnp.clip(u1, 0, W - 1)
        v0 = jnp.clip(v0, 0, H - 1)
        v1 = jnp.clip(v1, 0, H - 1)

    # get pixel values with safe indexing
    def safe_get(yy, xx):
        yy = yy[..., None]  # (B,Hout,Wout,1)
        xx = xx[..., None]  # (B,Hout,Wout,1)
        in_bounds = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        vals = img[b, yy.clip(0, H - 1), xx.clip(0, W - 1), c]
        return vals * in_bounds.astype(img.dtype)

    b = jnp.arange(B)[:, None, None, None]  # (B,1,1,1)
    c = jnp.arange(C)[None, None, None, :]  # (1,1,1,C)

    Ia = safe_get(v0, u0)  # top-left
    Ib = safe_get(v1, u0)  # bottom-left
    Ic = safe_get(v0, u1)  # top-right
    Id = safe_get(v1, u1)  # bottom-right

    # bilinear weights
    wa = (u1 - u) * (v1 - v)
    wb = (u1 - u) * (v - v0)
    wc = (u - u0) * (v1 - v)
    wd = (u - u0) * (v - v0)

    # add channel axis so weights broadcast over C
    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def correlation_block(
    corr_pyramid: list[jnp.ndarray],
    coords: jnp.ndarray,
    radius: int = 4,
):
    """Correlation block for RAFT.

    Args:
        corr_pyramid: list of correlation volumes at different levels
        coords: (B, H1, W1, 2) coordinates of the pixels to sample
        radius: radius of the correlation block

    Returns:
        out: (B, H1, W1, (2r+1)^2 * num_levels) correlation features
    """
    B, H1, W1, _ = coords.shape
    num_levels = len(corr_pyramid)

    dx = jnp.arange(-radius, radius + 1, dtype=jnp.float16)
    dy = jnp.arange(-radius, radius + 1, dtype=jnp.float16)
    dX, dY = jnp.meshgrid(dx, dy, indexing="ij")
    delta = jnp.stack([dX, dY], axis=-1)  # (K,K,2) in (x,y)
    delta_lvl = delta[None, ...]  # (1,K,K,2)

    out_pyramid = []
    # TODO: use scan instead of for loop
    for i in range(num_levels):
        corr = corr_pyramid[i]
        centroid_lvl = coords.reshape(B * H1 * W1, 1, 1, 2) / (2**i)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        corr = corr.reshape(B, H1, W1, -1)
        out_pyramid.append(corr)

    out = jnp.concatenate(out_pyramid, axis=-1)
    return out


# TODO: finish this and test, verify if extract_patches is equivalent
# try to see if lax.lax.conv_general_dilated_patches it's faster
def unfold_patches(images: jnp.ndarray, K: int, S: int):
    """Extract overlapping KxK patches (K=size, S=stride) from images.

    Args:
      images: input images of shape (B, H, W, C)
      K: tile/window size K
      S: stride/shift S

    Returns:
      patches: extracted patches of shape (B, Ny, Nx, K, K, C) where
               Ny = (H - K) // S + 1, Nx = (W - K) // S + 1
    """
    _, H, W, C = images.shape

    ys = jnp.arange(0, H - K + 1, S, dtype=jnp.int32)  # (Ny,)
    xs = jnp.arange(0, W - K + 1, S, dtype=jnp.int32)  # (Nx,)

    # Extract patches from a single image (H,W,C) -> (Ny,Nx,K,K,C)
    def extract_one(img):
        def extract_at(y, x):
            return jax.lax.dynamic_slice(img, (y, x, 0), (K, K, C))  # (K,K,C)

        # vmap over x, then over y -> (Ny,Nx,K,K,C)
        patches = jax.vmap(lambda y: jax.vmap(lambda x: extract_at(y, x))(xs))(ys)
        return patches

    # vmap over batch -> (B,Ny,Nx,K,K,C)
    patches = jax.vmap(extract_one)(images)
    return patches


def triang(M: int) -> jnp.ndarray:
    """Compute a 1D triangular window.

    Args:
        M (int): Size of the window.

    Returns:
        jnp.ndarray: Triangular window of shape (M,).
    """
    n = jnp.arange(0, M, dtype=jnp.float16)
    # Formula from scipy.signal.windows.triang
    if M % 2 == 0:
        w = 1.0 - jnp.abs((n - (M - 1) / 2) / (M / 2))
    else:
        w = 1.0 - jnp.abs((n - (M - 1) / 2) / ((M + 1) / 2))
    return w  # shape (M,)


def spline_window(window_size, power=2):
    """Compute a 1D spline window.

    Args:
        window_size (int): Size of the window.
        power (int): Power for the spline.

    Returns:
        jnp.ndarray: Spline window of shape (window_size,).
    """
    intersection = int(window_size / 4)
    wind_outer = (jnp.abs(2 * (triang(window_size))) ** power) / 2
    wind_outer = wind_outer.at[intersection:-intersection].set(0)

    wind_inner = 1 - (abs(2 * (triang(window_size) - 1)) ** power) / 2
    wind_inner = wind_inner.at[:intersection].set(0)
    wind_inner = wind_inner.at[-intersection:].set(0)

    wind = wind_inner + wind_outer
    wind = wind / jnp.average(wind)
    return wind


def window_2D(window_size, power=2):
    """Compute a 2D spline window.

    Args:
        window_size (int): Size of the window.
        power (int): Power for the spline.

    Returns:
        jnp.ndarray: 2D spline window of shape (window_size, window_size).
    """
    wind = spline_window(window_size, power)
    wind = jnp.expand_dims(jnp.expand_dims(wind, -1), -1)  # shape (window_size, 1, 1)
    wind = wind * wind.transpose(1, 0, 2)  # shape (window_size, window_size, 1)
    return jnp.squeeze(wind, axis=-1)


def fold_patches(patches, H, W, S):
    """Recompose patches into full images by summing overlaps.

    Args:
        patches: (Ny, Nx, K, K, C) patches to recompose
        H: height of the full image
        W: width of the full image
        S: stride/shift S

    Returns:
      full: recomposed full image of shape (H, W, C)
    """
    Ny, Nx, K, _, C = patches.shape
    patches = patches.reshape((Ny * Nx, K, K, C))  # (Ny*Nx, K, K, C)

    # Prepare empty canvas
    full = jnp.zeros((H, W, C), patches.dtype)

    # Generate top-left anchor positions for all patches
    y_idxs = jnp.arange(Ny) * S
    x_idxs = jnp.arange(Nx) * S
    ys, xs = jnp.meshgrid(y_idxs, x_idxs, indexing="ij")
    ys = ys.reshape(-1)  # (Ny*Nx,)
    xs = xs.reshape(-1)

    # Get internal patch coordinates (0 to K-1)
    ky, kx = jnp.meshgrid(jnp.arange(K), jnp.arange(K), indexing="ij")  # (K, K)
    ky = ky.reshape(-1)
    kx = kx.reshape(-1)

    # Compute absolute coordinates for all patch pixels
    y_coords = (ys[:, None] + ky[None, :]).reshape(-1)  # (N*K*K,)
    x_coords = (xs[:, None] + kx[None, :]).reshape(-1)  # (N*K*K,)

    # Flatten all patch pixel values
    pixel_values = patches.reshape(-1, C)  # (N*K*K, C)

    # Scatter-add all pixels into their positions
    full = full.at[y_coords, x_coords].add(pixel_values)

    return full


# alternative to avg_pool from jax.lax
# TODO: benchmark
def avg_pool2(x: jnp.ndarray) -> jnp.ndarray:
    """2x2 average pooling using reduce_window.

    Args:
        x: (N, H, W, C) input tensor

    Returns:
        y: (N, H//2, W//2, C) pooled tensor
    """
    # x: (N, H, W, C) -> (N, H//2, W//2, C)
    y = lax.reduce_window(
        x,
        0.0,
        lax.add,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )
    return y / 4.0
