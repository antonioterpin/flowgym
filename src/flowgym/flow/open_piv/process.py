"""Module for OpenPIV processing in JAX."""

import jax
import jax.numpy as jnp
from jax import lax

from flowgym.utils import DEBUG
from flowgym.flow.process import img_resize


def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int,
    search_area_size: int,
    overlap: int,
) -> jnp.ndarray:
    """Batched PIV cross-correlation algorithm.

    JAX implementation of the openpiv extended search area PIV algorithm.
    See https://github.com/OpenPIV/openpiv-python

    Args:
        img1: First image.
        img2: Second image.
        window_size: Size of the interrogation window.
        overlap: Overlap between interrogation windows.
        search_area_size: Size of the search area.

    Returns:
        Displacement field of shape (batch_size, n_rows, n_cols, 2).
    """
    # Validate inputs
    if DEBUG:
        assert (
            img1.ndim == 3
        ), f"Image must be (batch_size, height, width), instead {img1.shape}"
        assert (
            img2.ndim == 3
        ), f"Image must be (batch_size, height, width), instead {img2.shape}"
        assert isinstance(window_size, int), "Window size must be an integer"
        assert isinstance(overlap, int), "Overlap must be an integer"
        assert isinstance(search_area_size, int)
        # TODO: allow <=
        assert (
            search_area_size >= window_size
        ), "Search area size must be greater than window size"

    # TODO: extend to handle non-square windows
    window_size_tuple = (window_size, window_size)
    overlap_tuple = (overlap, overlap)
    search_area_size_tuple = (search_area_size, search_area_size)

    # Extract windows
    aa = sliding_window_array(img1, search_area_size_tuple, overlap_tuple)
    bb = sliding_window_array(img2, search_area_size_tuple, overlap_tuple)

    n_rows, n_cols = get_field_shape(
        (img1.shape[1], img1.shape[2]), search_area_size_tuple, overlap_tuple
    )
    if DEBUG:
        assert (
            aa.shape == (img1.shape[0], n_rows * n_cols) + search_area_size_tuple
        ), f"Sliding window wrong dimensions: {aa.shape}"
        assert (
            bb.shape == (img2.shape[0], n_rows * n_cols) + search_area_size_tuple
        ), f"Sliding window wrong dimensions: {bb.shape}"

    # Extended search area masking
    # TODO: do this optionally, only if search_area_size > window_size
    mask = jnp.zeros(search_area_size_tuple, dtype=aa.dtype)
    pady = (search_area_size_tuple[0] - window_size_tuple[0]) // 2
    padx = (search_area_size_tuple[1] - window_size_tuple[1]) // 2
    mask = mask.at[
        pady : search_area_size_tuple[0] - pady, padx : search_area_size_tuple[1] - padx
    ].set(1)
    mask = mask[None, None, :, :]
    aa = aa * mask
    aa = normalize_intensity(aa)
    bb = normalize_intensity(bb)

    # Compute correlation
    corr = fft_correlate_images(aa, bb)

    # Find peaks and compute displacements
    peaks_i, peaks_j = find_all_first_peaks(corr)
    disp_vx, disp_vy = jax.vmap(subpixel_displacement)(corr, peaks_i, peaks_j)

    # Reshape displacements
    disp_vx = disp_vx.reshape(img1.shape[0], n_rows, n_cols)
    disp_vy = disp_vy.reshape(img1.shape[0], n_rows, n_cols)

    # final displacement field of shape (batch, n_rows, n_cols, 2)
    return jnp.stack((disp_vx, disp_vy), axis=-1)


def get_field_shape(
    image_size: tuple[int, int],
    search_area_size: tuple[int, int],
    overlap: tuple[int, int],
) -> tuple[int, int]:
    """Compute the shape of the resulting flow field.

    Args:
        image_size: Size of the image (height, width).
        search_area_size: Size of the search area (height, width).
        overlap: Overlap between windows (height, width).

    Returns:
        Shape of the resulting flow field (num_rows, num_cols).
    """
    if DEBUG:
        assert (
            len(image_size) == 2
        ), f"Image size must be a tuple of (height, width), instead {image_size}"
        assert (
            len(search_area_size) == 2
        ), "Search area size must be a tuple of (height, width)"
        assert len(overlap) == 2, "Overlap must be a tuple of (height, width)"

    return (
        (image_size[0] - search_area_size[0]) // (search_area_size[0] - overlap[0]) + 1,
        (image_size[1] - search_area_size[1]) // (search_area_size[1] - overlap[1]) + 1,
    )


def fft_correlate_images(aa: jnp.ndarray, bb: jnp.ndarray):
    """Perform FFT-based cross-correlation on batched windows.

    Args:
        aa: First image batch (..., height, width).
        bb: Second image batch (..., height, width).

    Returns:
        Cross-correlation result (..., height, width).
    """
    aa = normalize_intensity(aa)
    bb = normalize_intensity(bb)

    s2 = aa.shape[-2:]

    f2a = jnp.conj(jnp.fft.rfft2(aa, axes=(-2, -1)))
    f2b = jnp.fft.rfft2(bb, axes=(-2, -1))
    corr = jnp.fft.irfft2(f2a * f2b, axes=(-2, -1))
    corr = jnp.fft.fftshift(corr, axes=(-2, -1))

    corr = corr / (s2[0] * s2[1])
    corr = jnp.clip(corr, 0, 1)
    return corr


def normalize_intensity(windows: jnp.ndarray) -> jnp.ndarray:
    """Normalize intensity of windows by subtracting mean and dividing by std.

    Args:
        windows: Batched windows (..., height, width).

    Returns:
        Normalized windows (..., height, width).
    """
    mean = jnp.mean(windows, axis=(-2, -1), keepdims=True)
    std = jnp.std(windows, axis=(-2, -1), keepdims=True)
    normalized = jnp.where(std == 0, jnp.zeros_like(windows), (windows - mean) / std)
    return jnp.clip(normalized, 0, normalized.max())


def find_all_first_peaks(corr):
    """Find first peaks in batched correlation maps.

    Args:
        corr: Batched correlation maps (..., height, width).

    Returns:
        Indices of peaks (peaks_i, peaks_j).
    """
    batch_size, num_windows, _, corr_width = corr.shape
    flat_corr = corr.reshape(batch_size, num_windows, -1)
    ind = jnp.argmax(flat_corr, axis=-1)
    peaks_i = ind // corr_width
    peaks_j = ind % corr_width
    return peaks_i, peaks_j


def subpixel_displacement(
    corr: jnp.ndarray, peaks_i: jnp.ndarray, peaks_j: jnp.ndarray, mask_width: int = 1
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute subpixel displacements from correlation maps.

    Args:
        corr: Correlation maps (batch_size * n_windows, height, width).
        peaks_i: Peak indices in the i direction.
        peaks_j: Peak indices in the j direction.
        mask_width: Width of the mask for invalid peaks.

    Returns:
        Subpixel displacements (disp_vx, disp_vy).
    """
    if DEBUG:
        assert corr.ndim == 3, (
            "Correlation must be (batch_size * n_windows, height, width), "
            + f"instead {corr.shape}"
        )
        assert (
            peaks_i.ndim == 1 and peaks_j.ndim == 1
        ), f"Peaks must be 1D arrays, instead {peaks_i.shape} and {peaks_j.shape}"
        assert len(peaks_i) == len(peaks_j), "Peaks i and j must have the same length"
        assert len(peaks_i) == corr.shape[0], (
            "Peaks must match the number of windows, "
            f"instead {len(peaks_i)} and {corr.shape[0]}",
        )

    K, H, W = corr.shape
    idx = jnp.arange(K)

    # 1) Identify out-of-bounds (“invalid”) peaks
    invalid = (
        (peaks_i < mask_width)
        | (peaks_i > (H - mask_width - 1))
        | (peaks_j < mask_width)
        | (peaks_j > (W - mask_width - 1))
    )

    # 2) “Safe” indices (so we never index outside) — we’ll mask these later
    safe_i = jnp.where(invalid, H // 2, peaks_i)
    safe_j = jnp.where(invalid, W // 2, peaks_j)

    if DEBUG:
        assert safe_i.shape == (K,) and safe_j.shape == (
            K,
        ), f"Safe indices must be 1D, instead {safe_i.shape} and {safe_j.shape}"

    # 3) Gather the 5-point stencil around each peak
    c = corr[idx, safe_i, safe_j]
    cl = corr[idx, safe_i - 1, safe_j]
    cr = corr[idx, safe_i + 1, safe_j]
    cd = corr[idx, safe_i, safe_j - 1]
    cu = corr[idx, safe_i, safe_j + 1]

    if DEBUG:
        assert c.shape == (K,), f"Peak correlation must be 1D, instead {c.shape}"
        assert cl.shape == (K,) and cr.shape == (
            K,
        ), f"Left and right correlations must be 1D, instead {cl.shape} and {cr.shape}"
        assert cd.shape == (K,) and cu.shape == (
            K,
        ), f"Down and up correlations must be 1D, instead {cd.shape} and {cu.shape}"

    # 4) Detect any non-positive values → fallback to 3‑point parabolic
    inv = (c <= 0) | (cl <= 0) | (cr <= 0) | (cd <= 0) | (cu <= 0)

    # 5) Log‑parabolic interpolation
    lcl, lcr, lc = jnp.log(cl), jnp.log(cr), jnp.log(c)
    lcd, lcu = jnp.log(cd), jnp.log(cu)
    nom1 = lcl - lcr
    den1 = 2 * lcl - 4 * lc + 2 * lcr
    nom2 = lcd - lcu
    den2 = 2 * lcd - 4 * lc + 2 * lcu

    shift_i_log = jnp.where(den1 != 0, nom1 / den1, 0.0)
    shift_j_log = jnp.where(den2 != 0, nom2 / den2, 0.0)

    # 6) 3‑point parabolic fallback
    shift_i_fallback = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
    shift_j_fallback = (cd - cu) / (2 * cd - 4 * c + 2 * cu)

    # 7) Combine & shift relative to center
    shift_i = jnp.where(inv, shift_i_fallback, shift_i_log)
    shift_j = jnp.where(inv, shift_j_fallback, shift_j_log)

    disp_vy = shift_i + safe_i - jnp.floor(H / 2)
    disp_vx = shift_j + safe_j - jnp.floor(W / 2)

    # 8) Mask out the originally invalid peaks → NaN
    disp_vx = jnp.where(invalid, jnp.nan, disp_vx)
    disp_vy = jnp.where(invalid, jnp.nan, disp_vy)

    return disp_vx, disp_vy


def sliding_window_array(
    image: jnp.ndarray, window_size: tuple[int, int], overlap: tuple[int, int]
) -> jnp.ndarray:
    """Extract sliding windows from a batch of images.

    Args:
        image: Batch of images (batch_size, height, width).
        window_size: Size of the window (height, width).
        overlap: Overlap between windows (height, width).

    Returns:
        Extracted windows (batch_size, num_windows, win_height, win_width).
    """
    if DEBUG:
        assert image.ndim == 3, "Image batch must be 3D (batch_size, height, width)"
        assert len(window_size) == 2, "Window size must be a tuple of (height, width)"
        assert len(overlap) == 2, "Overlap must be a tuple of (height, width)"

    xs, ys = get_rect_coordinates(
        (image.shape[1], image.shape[2]), window_size, overlap
    )
    half_h, half_w = window_size[0] // 2, window_size[1] // 2
    coords = jnp.stack([ys - half_h, xs - half_w], axis=-1).astype(jnp.int32)

    def extract_windows_single(img):
        def slice_one(coord):
            return lax.dynamic_slice(img, (coord[0], coord[1]), window_size)

        return jax.vmap(slice_one)(coords)

    return jax.vmap(extract_windows_single)(image)


def get_rect_coordinates(
    image_size: tuple[int, int], window_size: tuple[int, int], overlap: tuple[int, int]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute coordinates of interrogation window centers.

    Args:
        image_size: Size of the image (height, width).
        window_size: Size of the window (height, width).
        overlap: Overlap between windows (height, width).

    Returns:
        Coordinates of the window centers (X, Y).
    """
    h, w = image_size
    wh, ww = window_size
    oh, ow = overlap
    sh, sw = wh - oh, ww - ow
    ny, nx = (h - wh) // sh + 1, (w - ww) // sw + 1
    ys = jnp.arange(ny) * sh + wh * 0.5
    xs = jnp.arange(nx) * sw + ww * 0.5
    X, Y = jnp.meshgrid(xs, ys)
    return X.flatten(), Y.flatten()


def upsample_flow(flows: jnp.ndarray, image_shape: tuple[int, int]) -> jnp.ndarray:
    """Upsample flow field to match the target image shape.

    Args:
        flows: (B, height, width, 2) flow field to resize
        image_shape: (height, width) target image shape

    Returns:
        full_flows: (B, height, width, 2)
    """
    if DEBUG:
        assert flows.ndim == 4, (
            "Flow field must be 4D (batch_size, height, width, channels), "
            + f"instead {flows.shape}"
        )
        assert flows.shape[-1] == 2, "Flow field must have 2 channels"
        assert (
            len(image_shape) == 2
        ), f"Image shape must be a tuple of (height, width), instead {image_shape}"
        assert isinstance(
            image_shape[0], int
        ), f"Image height must be an integer, instead {image_shape[0]}"
        assert isinstance(
            image_shape[1], int
        ), f"Image width must be an integer, instead {image_shape[1]}"
        assert image_shape[0] >= flows.shape[1], (
            "Image height must be greater or equal to flow field height, "
            + f"instead {image_shape[0]} and {flows.shape[1]}"
        )
        assert image_shape[1] >= flows.shape[2], (
            "Image width must be greater or equal to flow field width, "
            + f"instead {image_shape[1]} and {flows.shape[2]}"
        )
    flows_x = flows[..., 0]
    flows_y = flows[..., 1]
    images_resize = jax.vmap(img_resize, in_axes=(0, None))
    flows_x_resized = images_resize(flows_x, image_shape)
    flows_y_resized = images_resize(flows_y, image_shape)
    return jnp.stack((flows_x_resized, flows_y_resized), axis=-1)
