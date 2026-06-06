"""Module for OpenPIV processing in JAX."""

from typing import overload

import jax
import jax.numpy as jnp
from jax import lax

from flowgym.flow.process import img_resize
from flowgym.utils import DEBUG


@overload
def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int,
    search_area_size: int,
    overlap: int,
    sig2noise_method: None = None,
    width: int = 2,
    subpixel_method: str = "gaussian",
) -> jnp.ndarray: ...


@overload
def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int,
    search_area_size: int,
    overlap: int,
    sig2noise_method: str,
    width: int = 2,
    subpixel_method: str = "gaussian",
) -> tuple[jnp.ndarray, jnp.ndarray]: ...


def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int,
    search_area_size: int,
    overlap: int,
    sig2noise_method: str | None = None,
    width: int = 2,
    subpixel_method: str = "gaussian",
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Batched PIV cross-correlation algorithm.

    JAX implementation of the openpiv extended search area PIV algorithm.
    See https://github.com/OpenPIV/openpiv-python

    Args:
        img1: First image.
        img2: Second image.
        window_size: Size of the interrogation window.
        search_area_size: Size of the search area.
        overlap: Overlap between interrogation windows.
        sig2noise_method: Optional signal-to-noise method ("peak2peak" or
            "peak2mean"). When set, the per-window signal-to-noise ratio is
            returned alongside the displacement field, mirroring the third
            output of the openpiv reference pipeline.
        width: Half-size of the exclusion box around the first correlation
            peak; only used when ``sig2noise_method == "peak2peak"``.
        subpixel_method: Sub-pixel peak estimator passed to
            :func:`subpixel_displacement`, one of ``"gaussian"``,
            ``"parabolic"`` or ``"centroid"``.

    Returns:
        Displacement field of shape (batch_size, n_rows, n_cols, 2). If
        ``sig2noise_method`` is set, a tuple of the displacement field and
        the signal-to-noise ratios of shape (batch_size, n_rows, n_cols).
    """
    # Validate inputs
    if DEBUG:
        assert img1.ndim == 3, (
            f"Image must be (batch_size, height, width), instead {img1.shape}"
        )
        assert img2.ndim == 3, (
            f"Image must be (batch_size, height, width), instead {img2.shape}"
        )
        assert isinstance(window_size, int), "Window size must be an integer"
        assert isinstance(overlap, int), "Overlap must be an integer"
        assert isinstance(search_area_size, int)
        # TODO: allow <=
        assert search_area_size >= window_size, (
            "Search area size must be greater than window size"
        )

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
        assert aa.shape == (
            img1.shape[0],
            n_rows * n_cols,
            *search_area_size_tuple,
        ), f"Sliding window wrong dimensions: {aa.shape}"
        assert bb.shape == (
            img2.shape[0],
            n_rows * n_cols,
            *search_area_size_tuple,
        ), f"Sliding window wrong dimensions: {bb.shape}"

    # Extended search area masking. This must mirror the reference openpiv
    # pipeline (pyprocess.extended_search_area_piv) exactly, otherwise the
    # correlation maps differ and the argmax peak can land on a different
    # pixel in competitive windows. The reference only normalizes-then-masks
    # when the search area is strictly larger than the interrogation window,
    # and normalizes BEFORE masking so the zeroed border does not pollute the
    # per-window mean/std. fft_correlate_images then normalizes once more, so
    # the extended-search branch is normalized twice exactly as in openpiv.
    if search_area_size > window_size:
        aa = normalize_intensity(aa)
        bb = normalize_intensity(bb)
        mask = jnp.zeros(search_area_size_tuple, dtype=aa.dtype)
        pady = (search_area_size_tuple[0] - window_size_tuple[0]) // 2
        padx = (search_area_size_tuple[1] - window_size_tuple[1]) // 2
        mask = mask.at[
            pady : search_area_size_tuple[0] - pady,
            padx : search_area_size_tuple[1] - padx,
        ].set(1)
        aa = aa * mask[None, None, :, :]

    # Compute correlation (normalizes the windows internally, matching the
    # reference's normalized_correlation=True path).
    corr = fft_correlate_images(aa, bb)

    # Find peaks and compute displacements. subpixel_method is static, so it
    # is closed over rather than vmapped.
    peaks_i, peaks_j = find_all_first_peaks(corr)
    disp_vx, disp_vy = jax.vmap(
        lambda c, pi, pj: subpixel_displacement(
            c, pi, pj, subpixel_method=subpixel_method
        )
    )(corr, peaks_i, peaks_j)

    # Reshape displacements
    disp_vx = disp_vx.reshape(img1.shape[0], n_rows, n_cols)
    disp_vy = disp_vy.reshape(img1.shape[0], n_rows, n_cols)

    # final displacement field of shape (batch, n_rows, n_cols, 2)
    flow = jnp.stack((disp_vx, disp_vy), axis=-1)

    if sig2noise_method is not None:
        s2n = sig2noise_ratio(corr, sig2noise_method, width)
        return flow, s2n.reshape(img1.shape[0], n_rows, n_cols)
    return flow


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
        assert len(image_size) == 2, (
            f"Image size must be a tuple of (height, width), "
            f"instead {image_size}"
        )
        assert len(search_area_size) == 2, (
            "Search area size must be a tuple of (height, width)"
        )
        assert len(overlap) == 2, "Overlap must be a tuple of (height, width)"

    return (
        (image_size[0] - search_area_size[0])
        // (search_area_size[0] - overlap[0])
        + 1,
        (image_size[1] - search_area_size[1])
        // (search_area_size[1] - overlap[1])
        + 1,
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
    normalized = jnp.where(
        std == 0, jnp.zeros_like(windows), (windows - mean) / std
    )
    return jnp.clip(normalized, 0, normalized.max())


def find_all_first_peaks(corr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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


def find_all_second_peaks(
    corr: jnp.ndarray, width: int = 2
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find the second-highest peak outside a box around the first peak.

    Mirrors ``openpiv.pyprocess.find_all_second_peaks``: a square region of
    half-size ``width`` centred on the first peak (clipped at the map
    borders, exactly like the reference's clamped slices) is excluded, and
    the highest remaining value is the second peak.

    Args:
        corr: Batched correlation maps (..., height, width).
        width: Half-size of the exclusion box around the first peak.

    Returns:
        Tuple ``(peaks_i, peaks_j, peaks_value)`` of the row indices, column
        indices and heights of the second peaks, each of shape
        ``corr.shape[:-2]``.
    """
    if DEBUG:
        assert corr.ndim >= 2, (
            f"Correlation must be (..., height, width), instead {corr.shape}"
        )
        assert width > 0, "Width must be positive."

    H, W = corr.shape[-2:]
    flat = corr.reshape(*corr.shape[:-2], -1)
    ind1 = jnp.argmax(flat, axis=-1)
    peaks1_i, peaks1_j = ind1 // W, ind1 % W

    # Exclude the (2 * width + 1)^2 box around the first peak. The reference
    # clamps the box slices at the map borders, which is exactly what the
    # |index - peak| <= width mask reproduces.
    rows = jnp.arange(H)
    cols = jnp.arange(W)
    box = (jnp.abs(rows[:, None] - peaks1_i[..., None, None]) <= width) & (
        jnp.abs(cols[None, :] - peaks1_j[..., None, None]) <= width
    )
    masked = jnp.where(box, -jnp.inf, corr)

    flat2 = masked.reshape(*corr.shape[:-2], -1)
    ind2 = jnp.argmax(flat2, axis=-1)
    peaks2 = jnp.max(flat2, axis=-1)
    return ind2 // W, ind2 % W, peaks2


def sig2noise_ratio(
    corr: jnp.ndarray,
    sig2noise_method: str = "peak2peak",
    width: int = 2,
) -> jnp.ndarray:
    """Compute the signal-to-noise ratio of batched correlation maps.

    JAX port of ``openpiv.pyprocess.vectorized_sig2noise_ratio``. The ratio
    is the first-peak height over the second-peak height ("peak2peak") or
    over the absolute mean of the correlation map ("peak2mean"), and is a
    per-window measure of the matching quality.

    Windows with a weak first peak (< 1e-3), a first peak on the map border
    and — for "peak2peak" — a weak or border second peak are flagged and
    their ratio is set to 0. Note that the reference builds this exact flag
    but never applies it (``flag is True`` indexes with a constant ``False``
    and selects nothing in openpiv 0.25.4); this port applies it as
    intended, matching the loop-based ``pyprocess.sig2noise_ratio``
    semantics on the shared rules.

    Args:
        corr: Batched correlation maps (..., height, width).
        sig2noise_method: Either "peak2peak" or "peak2mean".
        width: Half-size of the exclusion box around the first peak; only
            used when ``sig2noise_method == "peak2peak"``.

    Returns:
        Signal-to-noise ratios of shape ``corr.shape[:-2]``, with flagged
        windows set to 0.

    Raises:
        ValueError: If ``sig2noise_method`` is not supported.
    """
    if sig2noise_method not in ("peak2peak", "peak2mean"):
        raise ValueError(f"sig2noise_method not supported: {sig2noise_method}")
    if DEBUG:
        assert corr.ndim >= 2, (
            f"Correlation must be (..., height, width), instead {corr.shape}"
        )

    H, W = corr.shape[-2:]
    flat = corr.reshape(*corr.shape[:-2], -1)
    ind1 = jnp.argmax(flat, axis=-1)
    peaks1_i, peaks1_j = ind1 // W, ind1 % W
    peaks1 = jnp.max(flat, axis=-1)

    flag = (
        (peaks1 < 1e-3)
        | (peaks1_i == 0)
        | (peaks1_i == H - 1)
        | (peaks1_j == 0)
        | (peaks1_j == W - 1)
    )

    if sig2noise_method == "peak2peak":
        peaks2_i, peaks2_j, peaks2 = find_all_second_peaks(corr, width)
        flag = (
            flag
            | (peaks2 < 1e-3)
            | (peaks2_i == 0)
            | (peaks2_i == H - 1)
            | (peaks2_j == 0)
            | (peaks2_j == W - 1)
        )
        noise = peaks2
    else:
        noise = jnp.abs(jnp.nanmean(corr, axis=(-2, -1)))

    # Reference: np.divide(..., out=zeros, where=noise > 0). The guarded
    # denominator keeps the division NaN-free under jit.
    ratio = jnp.where(noise > 0, peaks1 / jnp.where(noise > 0, noise, 1.0), 0.0)
    return jnp.where(flag, 0.0, ratio)


def subpixel_displacement(
    corr: jnp.ndarray,
    peaks_i: jnp.ndarray,
    peaks_j: jnp.ndarray,
    mask_width: int = 1,
    eps: float = 1e-7,
    subpixel_method: str = "gaussian",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute subpixel displacements from correlation maps.

    Mirrors the three estimators of openpiv's
    ``vectorized_correlation_to_displacements`` over the 3-point stencil
    around each peak:

    - ``"gaussian"``: log-parabolic fit, with a 3-point parabolic fallback on
      any non-positive stencil value (the reference behaviour).
    - ``"parabolic"``: plain 3-point parabolic fit.
    - ``"centroid"``: intensity-weighted centroid of the stencil; unlike the
      other two the fitted value is an absolute position rather than a
      sub-pixel offset from the peak.

    Args:
        corr: Correlation maps (batch_size * n_windows, height, width).
        peaks_i: Peak indices in the i direction.
        peaks_j: Peak indices in the j direction.
        mask_width: Width of the mask for invalid peaks.
        eps: Small constant added to the correlation map before the fit,
            matching openpiv's vectorized_correlation_to_displacements. It
            both prevents log(0) and keeps every stencil value strictly
            positive, so the gaussian branch is always taken exactly as in
            the reference (no parabolic fallback on clipped zeros).
        subpixel_method: Peak estimator, one of ``"gaussian"``,
            ``"parabolic"`` or ``"centroid"``.

    Returns:
        Subpixel displacements (disp_vx, disp_vy).

    Raises:
        ValueError: If ``subpixel_method`` is not implemented.
    """
    if subpixel_method not in ("gaussian", "parabolic", "centroid"):
        raise ValueError(
            f"Unknown subpixel_method {subpixel_method!r}; expected one of "
            "'gaussian', 'parabolic', 'centroid'."
        )
    if DEBUG:
        assert corr.ndim == 3, (
            "Correlation must be (batch_size * n_windows, height, width), "
            + f"instead {corr.shape}"
        )
        assert peaks_i.ndim == 1 and peaks_j.ndim == 1, (
            f"Peaks must be 1D arrays, instead {peaks_i.shape} and "
            f"{peaks_j.shape}"
        )
        assert len(peaks_i) == len(peaks_j), (
            "Peaks i and j must have the same length"
        )
        assert len(peaks_i) == corr.shape[0], (
            "Peaks must match the number of windows, "
            f"instead {len(peaks_i)} and {corr.shape[0]}",
        )

    K, H, W = corr.shape
    idx = jnp.arange(K)

    # Match the reference: stabilize the correlation map so the gaussian fit
    # never sees a non-positive stencil value. argmax is invariant to a
    # uniform shift, so the supplied peak indices remain valid.
    corr = corr + eps

    # 1) Identify out-of-bounds ("invalid") peaks
    invalid = (
        (peaks_i < mask_width)
        | (peaks_i > (H - mask_width - 1))
        | (peaks_j < mask_width)
        | (peaks_j > (W - mask_width - 1))
    )

    # 2) "Safe" indices (so we never index outside) -- we'll mask these later
    safe_i = jnp.where(invalid, H // 2, peaks_i)
    safe_j = jnp.where(invalid, W // 2, peaks_j)

    if DEBUG:
        assert safe_i.shape == (K,) and safe_j.shape == (K,), (
            f"Safe indices must be 1D, instead {safe_i.shape} and "
            f"{safe_j.shape}"
        )

    # 3) Gather the 5-point stencil around each peak
    c = corr[idx, safe_i, safe_j]
    cl = corr[idx, safe_i - 1, safe_j]
    cr = corr[idx, safe_i + 1, safe_j]
    cd = corr[idx, safe_i, safe_j - 1]
    cu = corr[idx, safe_i, safe_j + 1]

    if DEBUG:
        assert c.shape == (K,), (
            f"Peak correlation must be 1D, instead {c.shape}"
        )
        assert cl.shape == (K,) and cr.shape == (K,), (
            f"Left and right correlations must be 1D, instead "
            f"{cl.shape} and {cr.shape}"
        )
        assert cd.shape == (K,) and cu.shape == (K,), (
            f"Down and up correlations must be 1D, instead "
            f"{cd.shape} and {cu.shape}"
        )

    # 4) Estimate the sub-pixel peak with the requested method. The branch is
    # on a static Python string, so only one path is traced.
    if subpixel_method == "centroid":
        # Intensity-weighted centroid: yields an absolute position, so the
        # peak index is already folded in (no `+ safe_i` below).
        fi, fj = safe_i.astype(corr.dtype), safe_j.astype(corr.dtype)
        shift_i = ((fi - 1) * cl + fi * c + (fi + 1) * cr) / (cl + c + cr)
        shift_j = ((fj - 1) * cd + fj * c + (fj + 1) * cu) / (cd + c + cu)
        disp_vy = shift_i - jnp.floor(H / 2)
        disp_vx = shift_j - jnp.floor(W / 2)
    elif subpixel_method == "parabolic":
        shift_i = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
        shift_j = (cd - cu) / (2 * cd - 4 * c + 2 * cu)
        disp_vy = shift_i + safe_i - jnp.floor(H / 2)
        disp_vx = shift_j + safe_j - jnp.floor(W / 2)
    else:  # gaussian
        # Detect any non-positive values -> fallback to 3-point parabolic.
        inv = (c <= 0) | (cl <= 0) | (cr <= 0) | (cd <= 0) | (cu <= 0)

        # Log-parabolic interpolation.
        lcl, lcr, lc = jnp.log(cl), jnp.log(cr), jnp.log(c)
        lcd, lcu = jnp.log(cd), jnp.log(cu)
        nom1 = lcl - lcr
        den1 = 2 * lcl - 4 * lc + 2 * lcr
        nom2 = lcd - lcu
        den2 = 2 * lcd - 4 * lc + 2 * lcu

        shift_i_log = jnp.where(den1 != 0, nom1 / den1, 0.0)
        shift_j_log = jnp.where(den2 != 0, nom2 / den2, 0.0)

        # 3-point parabolic fallback.
        shift_i_fallback = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
        shift_j_fallback = (cd - cu) / (2 * cd - 4 * c + 2 * cu)

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
        assert image.ndim == 3, (
            "Image batch must be 3D (batch_size, height, width)"
        )
        assert len(window_size) == 2, (
            "Window size must be a tuple of (height, width)"
        )
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
    image_size: tuple[int, int],
    window_size: tuple[int, int],
    overlap: tuple[int, int],
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


def upsample_flow(
    flows: jnp.ndarray, image_shape: tuple[int, int]
) -> jnp.ndarray:
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
        assert len(image_shape) == 2, (
            f"Image shape must be a tuple of (height, width), "
            f"instead {image_shape}"
        )
        assert isinstance(image_shape[0], int), (
            f"Image height must be an integer, instead {image_shape[0]}"
        )
        assert isinstance(image_shape[1], int), (
            f"Image width must be an integer, instead {image_shape[1]}"
        )
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
