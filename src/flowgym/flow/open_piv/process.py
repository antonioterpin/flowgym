"""Module for OpenPIV processing in JAX."""

from typing import overload

import jax
import jax.numpy as jnp
from jax import lax

from flowgym.flow.process import img_resize
from flowgym.utils import DEBUG


def _as_int(value: int) -> int:
    """Coerce an integer-like scalar to ``int``, rejecting non-integers.

    Window/overlap normalization runs host-side (not in the traced path),
    so this guard is always on rather than ``DEBUG``-gated: a non-integer
    such as ``32.5`` would otherwise silently truncate to ``32`` and yield
    a wrong-geometry PIV field instead of a clear error. Integer-valued
    floats (``32.0``) and ``numpy`` integers pass through unchanged.

    Args:
        value: An integer-like scalar.

    Returns:
        The value as a Python ``int``.

    Raises:
        ValueError: If the value is not integer-like.
    """
    ivalue = int(value)
    if ivalue != value:
        raise ValueError(
            f"Expected an integer-like window/overlap size, got {value!r}."
        )
    return ivalue


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an int or (height, width) pair to a 2-tuple of ints.

    Mirrors openpiv's scalar-to-tuple reshaping so square windows can be
    given as a single int while rectangular windows use an explicit pair.
    A tuple/list is taken as the pair; any other value is treated as a
    scalar and duplicated, so integer-like scalars such as ``numpy.int32``
    work too. Each element is coerced via :func:`_as_int`, so non-integer
    sizes raise rather than silently truncating.

    Args:
        value: Either an integer-like scalar (square) or a ``(height,
            width)`` pair.

    Returns:
        The value as a ``(height, width)`` tuple of ints.
    """
    if isinstance(value, (tuple, list)):
        if DEBUG:
            assert len(value) == 2, (
                "Expected an int or a (height, width) pair, got "
                f"length-{len(value)} {value!r}."
            )
        return (_as_int(value[0]), _as_int(value[1]))
    return (_as_int(value), _as_int(value))


@overload
def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int | tuple[int, int],
    search_area_size: int | tuple[int, int],
    overlap: int | tuple[int, int],
    sig2noise_method: None = None,
    width: int = 2,
    subpixel_method: str = "gaussian",
    correlation_method: str = "circular",
) -> jnp.ndarray: ...


@overload
def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int | tuple[int, int],
    search_area_size: int | tuple[int, int],
    overlap: int | tuple[int, int],
    sig2noise_method: str,
    width: int = 2,
    subpixel_method: str = "gaussian",
    correlation_method: str = "circular",
) -> tuple[jnp.ndarray, jnp.ndarray]: ...


def extended_search_area_piv(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int | tuple[int, int],
    search_area_size: int | tuple[int, int],
    overlap: int | tuple[int, int],
    sig2noise_method: str | None = None,
    width: int = 2,
    subpixel_method: str = "gaussian",
    correlation_method: str = "circular",
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
        correlation_method: Cross-correlation method passed to
            :func:`fft_correlate_images`, either ``"circular"`` or
            ``"linear"``.

    Note:
        ``sig2noise_method``, ``subpixel_method`` and ``correlation_method``
        are Python strings that select code paths, so under ``jax.jit`` they
        must be marked static (e.g. ``jax.jit(extended_search_area_piv,
        static_argnames=("window_size", "search_area_size", "overlap",
        "sig2noise_method", "subpixel_method", "correlation_method"))``)
        alongside the window-geometry arguments.

        Input validation is opt-in: the geometry constraints
        (``search_area_size >= window_size`` and
        ``overlap < search_area_size`` per axis) and the ``(height, width)``
        shape of tuple arguments are checked only under the module ``DEBUG``
        flag. With ``DEBUG`` disabled (the default) malformed geometry is not
        rejected here.

    Returns:
        Displacement field of shape (batch_size, n_rows, n_cols, 2). If
        ``sig2noise_method`` is set, a tuple of the displacement field and
        the signal-to-noise ratios of shape (batch_size, n_rows, n_cols).
    """
    # Accept square (int) or rectangular ((height, width)) windows, mirroring
    # openpiv's scalar-to-tuple reshaping.
    window_size_tuple = _as_pair(window_size)
    overlap_tuple = _as_pair(overlap)
    search_area_size_tuple = _as_pair(search_area_size)

    # Validate inputs
    if DEBUG:
        assert img1.ndim == 3, (
            f"Image must be (batch_size, height, width), instead {img1.shape}"
        )
        assert img2.ndim == 3, (
            f"Image must be (batch_size, height, width), instead {img2.shape}"
        )
        assert (
            search_area_size_tuple[0] >= window_size_tuple[0]
            and search_area_size_tuple[1] >= window_size_tuple[1]
        ), "Search area size must be >= window size on both axes"
        assert (
            overlap_tuple[0] < search_area_size_tuple[0]
            and overlap_tuple[1] < search_area_size_tuple[1]
        ), "Overlap must be smaller than the search area size on both axes"

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
    # Lexicographic tuple comparison, matching openpiv exactly. Under the
    # per-axis `search >= window` precondition (asserted above under DEBUG)
    # this is equivalent to "search is larger on at least one axis": given
    # s[0] >= w[0] and s[1] >= w[1], if s != w then either s[0] > w[0] (lex
    # true via the first element) or s[0] == w[0] and s[1] > w[1] (lex true
    # via the second). Without it the equivalence breaks the other way:
    # search=(16, 64) vs window=(32, 16) is larger on axis 1 but lex-False
    # (axis 0 decides), so this branch is wrongly skipped.
    if search_area_size_tuple > window_size_tuple:
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
    corr = fft_correlate_images(aa, bb, correlation_method=correlation_method)

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


def fft_correlate_images(
    aa: jnp.ndarray, bb: jnp.ndarray, correlation_method: str = "circular"
):
    """Perform FFT-based cross-correlation on batched windows.

    Mirrors openpiv's ``fft_correlate_images`` with normalized correlation.
    The ``"circular"`` method correlates without zero-padding (fast, but the
    correlation wraps around for large displacements). The ``"linear"``
    method zero-pads each window to the next power-of-two-minus-one before
    the transform so the cross-correlation is acyclic, then crops back to the
    window size; this avoids the wraparound at the cost of larger transforms.

    Args:
        aa: First image batch (..., height, width).
        bb: Second image batch (..., height, width).
        correlation_method: Either ``"circular"`` or ``"linear"``.

    Note:
        ``correlation_method`` is a Python string that selects a code path,
        so under ``jax.jit`` it must be marked static (e.g.
        ``static_argnames="correlation_method"``). The ``ValueError`` below
        only fires when the value is concrete; passing it as a traced
        argument instead raises an opaque tracer ``TypeError``.

    Returns:
        Cross-correlation result (..., height, width).

    Raises:
        ValueError: If ``correlation_method`` is not implemented (only when
            the argument is a concrete Python string, not a tracer).
    """
    if correlation_method not in ("circular", "linear"):
        raise ValueError(
            f"Unknown correlation_method {correlation_method!r}; expected "
            "one of 'circular', 'linear'."
        )

    aa = normalize_intensity(aa)
    bb = normalize_intensity(bb)

    s1 = aa.shape[-2:]
    s2 = bb.shape[-2:]

    if correlation_method == "linear":
        # Zero-pad to the reference's fsize = 2**ceil(log2(s1+s2-1)) - 1 and
        # crop the centred s1-sized region after the inverse transform. The
        # bit_length form computes the same fsize exactly without float log2.
        fsize = tuple(
            (1 << (s1[d] + s2[d] - 2).bit_length()) - 1 for d in (0, 1)
        )
        f2a = jnp.conj(jnp.fft.rfft2(aa, s=fsize, axes=(-2, -1)))
        f2b = jnp.fft.rfft2(bb, s=fsize, axes=(-2, -1))
        # No `s=fsize` on the inverse, deliberately reproducing openpiv: with
        # odd fsize (e.g. 63) rfft2 produces a last-axis length of 32, and
        # irfft2 without `s` infers 2*(32-1) = 62, so corr's last axis is
        # fsize-1 -- asymmetric and one short of fsize. The centred crop below
        # still lands the right pixels because the s1-sized window fits inside
        # that shorter axis. Do NOT add `s=fsize` to "match" the forward
        # transform; it would shift the crop and desync from the reference.
        corr = jnp.fft.irfft2(f2a * f2b, axes=(-2, -1)).real
        corr = jnp.fft.fftshift(corr, axes=(-2, -1))
        corr = corr[
            ...,
            (fsize[0] - s1[0]) // 2 : (fsize[0] + s1[0]) // 2,
            (fsize[1] - s1[1]) // 2 : (fsize[1] + s1[1]) // 2,
        ]
    else:  # circular
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


def sig2noise_val(s2n: jnp.ndarray, threshold: float = 1.0) -> jnp.ndarray:
    """Flag vectors whose signal-to-noise ratio is below a threshold.

    JAX port of ``openpiv.validation.sig2noise_val``: vectors whose
    signal-to-noise ratio (e.g. from :func:`sig2noise_ratio`) is below
    ``threshold`` are marked as outliers. This completes the standard
    signal-to-noise outlier-rejection path.

    NaN entries in ``s2n`` yield ``False`` (not flagged), since ``nan <
    threshold`` is ``False``, matching the reference.

    Note:
        This returns openpiv's convention, where ``True`` marks an *outlier*.
        That is the **inverse** of flowgym's ``postprocess`` mask convention
        (``True``/``1`` means *valid*; see ``postprocess.data_validation``).
        Invert the result (``~mask``) before composing it with ``postprocess``
        masks, otherwise it rejects exactly the vectors it should keep. The
        openpiv name is kept for port parity.

    Args:
        s2n: Signal-to-noise ratios of any shape.
        threshold: Vectors with ``s2n < threshold`` are flagged.

    Returns:
        Boolean array of the same shape as ``s2n``; ``True`` marks outliers
        (openpiv convention -- see the Note above).
    """
    return s2n < threshold


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
        #
        # The index is cast to corr.dtype (float32), so the whole centroid is
        # computed in single precision exactly as openpiv does (its
        # (peak-1)*cl etc. promote int*float32 -> float32). Keep it
        # single-precision: a well-meant float64 "fix" here would silently
        # desync from the reference.
        fi, fj = safe_i.astype(corr.dtype), safe_j.astype(corr.dtype)
        # Divisor left unguarded to match openpiv. `corr + eps` keeps every
        # stencil value positive only while the raw correlation is
        # non-negative; after normalized_correlation an entry below -eps can
        # drive cl + c + cr toward zero or flip its sign, yielding a garbage
        # absolute position. openpiv has the identical gap (locked by the
        # degenerate-stencil parity test).
        shift_i = ((fi - 1) * cl + fi * c + (fi + 1) * cr) / (cl + c + cr)
        shift_j = ((fj - 1) * cd + fj * c + (fj + 1) * cu) / (cd + c + cu)
        disp_vy = shift_i - jnp.floor(H / 2)
        disp_vx = shift_j - jnp.floor(W / 2)
    elif subpixel_method == "parabolic":
        # Denominator left unguarded to preserve openpiv parity: openpiv's
        # parabolic fit divides identically, so a degenerate (flat) stencil
        # yields 0/0 -> NaN (or +/-Inf) in both. Adding a jnp.where guard here
        # would desync from the reference (locked by the degenerate-stencil
        # parity test); the `invalid` border mask is what protects production.
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

        # 3-point parabolic fallback, also unguarded for parity (same as the
        # "parabolic" branch above); only selected where a stencil value is
        # non-positive, matching the reference's fallback path.
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


def deform_windows(
    frame: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Deform an image by the displacement field of a previous PIV pass.

    JAX port of openpiv's ``windef.deform_windows`` at linear interpolation
    (``interpolation_order = interpolation_order2 = 1``). The coarse
    displacement field defined on the interrogation-window centre grid is
    upsampled to every pixel and used to resample the image onto the
    deformed grid, which is the core operation of iterative window
    deformation.

    Two linear resamplings are performed, both matching the reference:

    - the window-centre field ``(u, v)`` is bilinearly interpolated onto the
      pixel grid; outside the window-centre grid the values are clamped to
      the border, exactly reproducing ``RectBivariateSpline`` with degree 1
      (which extrapolates as a constant there); and
    - the image is resampled at ``(y - vt, x + ut)`` with the same
      ``map_coordinates`` linear interpolation and ``"nearest"`` border mode
      as the reference.

    Only the linear case is reproduced: openpiv's default cubic field
    interpolation (``RectBivariateSpline`` degree 3) has no exact JAX
    equivalent.

    Args:
        frame: Single image of shape (height, width).
        x: Window-centre x coordinates as a (n_rows, n_cols) meshgrid.
        y: Window-centre y coordinates as a (n_rows, n_cols) meshgrid.
        u: u displacement component on the window-centre grid (n_rows, n_cols).
        v: v displacement component on the window-centre grid (n_rows, n_cols).

    Returns:
        The deformed image of shape (height, width).
    """
    if DEBUG:
        assert frame.ndim == 2, (
            f"Frame must be 2D (height, width), instead {frame.shape}"
        )
        assert x.ndim == 2 and y.ndim == 2, (
            "x and y must be 2D window-centre meshgrids."
        )
        assert u.shape == x.shape and v.shape == x.shape, (
            "u and v must match the window-centre grid shape."
        )

    frame = frame.astype(jnp.float32)
    height, width = frame.shape

    # Window-centre grid is uniformly spaced (overlap-defined), so a pixel
    # coordinate maps to a fractional field index by an affine transform.
    y1 = y[:, 0]
    x1 = x[0, :]
    dy = y1[1] - y1[0]
    dx = x1[1] - x1[0]

    side_y = jnp.arange(height, dtype=jnp.float32)
    side_x = jnp.arange(width, dtype=jnp.float32)
    idx_y = (side_y - y1[0]) / dy
    idx_x = (side_x - x1[0]) / dx
    grid_iy, grid_ix = jnp.meshgrid(idx_y, idx_x, indexing="ij")

    # Bilinear field upsampling with border clamping (== RectBivariateSpline
    # degree 1, which extrapolates as a constant outside the grid).
    ut = jax.scipy.ndimage.map_coordinates(
        u.astype(jnp.float32), [grid_iy, grid_ix], order=1, mode="nearest"
    )
    vt = jax.scipy.ndimage.map_coordinates(
        v.astype(jnp.float32), [grid_iy, grid_ix], order=1, mode="nearest"
    )

    # Resample the image onto the deformed grid.
    pixel_x, pixel_y = jnp.meshgrid(side_x, side_y)
    return jax.scipy.ndimage.map_coordinates(
        frame, [pixel_y - vt, pixel_x + ut], order=1, mode="nearest"
    )
