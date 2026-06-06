"""Numerical equivalence tests for the JAX OpenPIV reimplementation.

Each test pins a JAX function against the original ``openpiv`` reference
implementation it mirrors, so the two stay numerically aligned:

- ``subpixel_displacement`` vs
  ``pyprocess.vectorized_correlation_to_displacements``
- ``extended_search_area_piv`` (full pipeline) vs
  ``pyprocess.extended_search_area_piv``
- ``replace_outliers`` vs ``openpiv.filters.replace_outliers``
- ``find_all_second_peaks`` vs ``pyprocess.find_all_second_peaks``
- ``sig2noise_ratio`` vs ``pyprocess.vectorized_sig2noise_ratio``

The per-component tests (FFT correlation, peak finding, normalization,
sliding windows, coordinate grids) live in ``test_openpiv_jax``; this
module covers the sub-pixel stage, the end-to-end displacement field,
outlier replacement, and the signal-to-noise ratio.

Sig2noise parity note: the JAX :func:`sig2noise_ratio` mirrors
``vectorized_sig2noise_ratio`` with one deliberate divergence. The
reference computes a validity ``flag`` per window (weak or border first
peak, and for ``peak2peak`` a weak or border second peak) but never
applies it — ``peak2peak[flag is True] = 0`` indexes with the Python
expression ``flag is True`` (always ``False``), which numpy treats as an
empty boolean mask, so the assignment is a no-op in openpiv 0.25.4. The
JAX port applies the flag as evidently intended (the loop-based
``pyprocess.sig2noise_ratio`` does zero out failed windows). Parity is
therefore pinned in two parts: unflagged windows must match the
vectorized reference exactly, flagged windows must be zero.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from openpiv import filters, pyprocess

from flowgym.flow.open_piv.openpiv_jax import replace_outliers
from flowgym.flow.open_piv.process import (
    extended_search_area_piv,
    fft_correlate_images,
    find_all_first_peaks,
    find_all_second_peaks,
    get_field_shape,
    sig2noise_ratio,
    sliding_window_array,
    subpixel_displacement,
)


def _shifted_pair(height, width, shift_y, shift_x, pad=8, seed=0):
    """Build a particle-like image pair with a known uniform integer shift.

    The second frame is a copy of the first translated by ``(shift_y,
    shift_x)``, so every interrogation window shares the same displacement
    and the recovered field can be checked against the ground truth.
    """
    rng = np.random.RandomState(seed)
    base = rng.rand(height + 2 * pad, width + 2 * pad).astype(np.float32)
    frame_a = base[pad : pad + height, pad : pad + width]
    frame_b = base[
        pad - shift_y : pad - shift_y + height,
        pad - shift_x : pad - shift_x + width,
    ]
    return frame_a, frame_b


# ---------------------------------------------------------------------------
# subpixel_displacement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "subpixel_method", ["gaussian", "parabolic", "centroid"]
)
@pytest.mark.parametrize(
    "search_area_size, overlap",
    [(24, 12), (16, 8), (32, 16)],
)
def test_subpixel_displacement_matches_vectorized_reference(
    search_area_size, overlap, subpixel_method
):
    """JAX sub-pixel peak fitting matches openpiv across all methods."""
    height = width = 96
    frame_a, frame_b = _shifted_pair(height, width, shift_y=2, shift_x=4)

    aa = sliding_window_array(
        jnp.asarray(frame_a)[None],
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    bb = sliding_window_array(
        jnp.asarray(frame_b)[None],
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    corr = fft_correlate_images(aa, bb)
    peaks_i, peaks_j = find_all_first_peaks(corr)

    # JAX path: subpixel_displacement is already batched over windows.
    corr_windows = corr[0]
    disp_vx, disp_vy = subpixel_displacement(
        corr_windows,
        peaks_i[0],
        peaks_j[0],
        subpixel_method=subpixel_method,
    )
    disp_vx = np.asarray(disp_vx)
    disp_vy = np.asarray(disp_vy)

    # Reference path: same sub-pixel estimator.
    corr_ref = np.asarray(corr_windows).astype(np.float32)
    u_ref, v_ref = pyprocess.vectorized_correlation_to_displacements(
        corr_ref, subpixel_method=subpixel_method
    )

    # Invalid (NaN) windows must agree exactly between implementations.
    np.testing.assert_array_equal(
        ~np.isfinite(disp_vx), ~np.isfinite(np.asarray(u_ref))
    )
    np.testing.assert_array_equal(
        ~np.isfinite(disp_vy), ~np.isfinite(np.asarray(v_ref))
    )

    finite = np.isfinite(disp_vx) & np.isfinite(np.asarray(u_ref))
    np.testing.assert_allclose(
        disp_vx[finite], np.asarray(u_ref)[finite], atol=1e-4
    )
    np.testing.assert_allclose(
        disp_vy[finite], np.asarray(v_ref)[finite], atol=1e-4
    )


def test_subpixel_displacement_invalid_method_raises():
    """An unknown subpixel method raises a clear ValueError."""
    corr = jnp.ones((1, 8, 8))
    peaks_i = jnp.array([4])
    peaks_j = jnp.array([4])
    with pytest.raises(ValueError, match="Unknown subpixel_method"):
        subpixel_displacement(corr, peaks_i, peaks_j, subpixel_method="quartic")


def test_pipeline_invalid_subpixel_method_raises():
    """The full pipeline surfaces the ValueError, not an opaque tracer error.

    ``extended_search_area_piv`` routes ``subpixel_method`` through to
    ``subpixel_displacement``; this pins that the validation still fires
    end-to-end, so a refactor hiding the inner call behind another
    jit/vmap wrapper cannot silently drop it.
    """
    frame_a = jnp.zeros((1, 32, 32))
    frame_b = jnp.zeros((1, 32, 32))
    with pytest.raises(ValueError, match="Unknown subpixel_method"):
        extended_search_area_piv(
            frame_a,
            frame_b,
            window_size=16,
            overlap=8,
            search_area_size=16,
            subpixel_method="quartic",
        )


def test_subpixel_parabolic_centroid_unguarded_match_reference():
    """Degenerate stencils reproduce openpiv's unguarded Inf/NaN exactly.

    The parabolic and centroid divisors are intentionally left unguarded to
    preserve 1-to-1 parity with openpiv (whose
    ``vectorized_correlation_to_displacements`` divides with the identical
    expressions). This pins that contract at the stencil level so a future
    ``jnp.where`` "fix" that silently desyncs from the reference is caught.
    Peaks are supplied explicitly, so the result depends only on the
    division, not on argmax tie-breaking.
    """
    eps = 1e-7
    H = W = 8
    # window 0: flat plus-shape -> parabolic den == 0, nom == 0 -> 0/0 -> NaN
    # window 1: cl + cr == 2*c (asymmetric) -> parabolic den == 0 -> +/-Inf
    # window 2: centroid divisor cl + c + cr ~= 0 (post-normalization
    #           negatives) -> finite but garbage absolute position
    corr = np.zeros((3, H, W), dtype=np.float32)
    corr[0, 3:6, 4] = 1.0
    corr[0, 4, 3:6] = 1.0
    corr[1, 3, 4], corr[1, 4, 4], corr[1, 5, 4] = 0.5, 1.0, 1.5
    corr[1, 4, 3], corr[1, 4, 5] = 0.5, 1.5
    corr[2, 3, 4], corr[2, 4, 4], corr[2, 5, 4] = -1.0, 1.5, -0.5
    corr[2, 4, 3], corr[2, 4, 5] = -1.0, -0.5
    peaks_i = jnp.array([4, 4, 4])
    peaks_j = jnp.array([4, 4, 4])

    def _reference(method):
        # openpiv's arithmetic on the same eps-stabilised float32 stencil.
        cc = corr + eps
        k = np.arange(3)
        c = cc[k, 4, 4]
        cl, cr = cc[k, 3, 4], cc[k, 5, 4]
        cd, cu = cc[k, 4, 3], cc[k, 4, 5]
        with np.errstate(divide="ignore", invalid="ignore"):
            if method == "parabolic":
                si = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
                sj = (cd - cu) / (2 * cd - 4 * c + 2 * cu)
                return sj + 4 - np.floor(W / 2), si + 4 - np.floor(H / 2)
            fi = np.float32(4)
            si = ((fi - 1) * cl + fi * c + (fi + 1) * cr) / (cl + c + cr)
            sj = ((fi - 1) * cd + fi * c + (fi + 1) * cu) / (cd + c + cu)
            return sj - np.floor(W / 2), si - np.floor(H / 2)

    for method in ("parabolic", "centroid"):
        vx, vy = subpixel_displacement(
            jnp.asarray(corr), peaks_i, peaks_j, subpixel_method=method
        )
        vx, vy = np.asarray(vx), np.asarray(vy)
        rx, ry = _reference(method)
        # Non-finite (NaN/Inf) positions agree exactly with the reference ...
        np.testing.assert_array_equal(np.isfinite(vx), np.isfinite(rx))
        np.testing.assert_array_equal(np.isfinite(vy), np.isfinite(ry))
        # ... and finite values match, even where the divisor collapses.
        fx, fy = np.isfinite(vx), np.isfinite(vy)
        np.testing.assert_allclose(vx[fx], rx[fx], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(vy[fy], ry[fy], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "subpixel_method", ["gaussian", "parabolic", "centroid"]
)
@pytest.mark.parametrize(
    "window_size, search_area_size, overlap",
    [(32, 32, 16), (16, 32, 8)],
)
def test_pipeline_subpixel_method_matches_reference(
    subpixel_method, window_size, search_area_size, overlap
):
    """End-to-end field matches the reference pipeline for each method."""
    height = width = 96
    frame_a, frame_b = _shifted_pair(
        height, width, shift_y=3, shift_x=-2, seed=4
    )

    u_ref, v_ref, _ = pyprocess.extended_search_area_piv(
        frame_a.copy(),
        frame_b.copy(),
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="circular",
        subpixel_method=subpixel_method,
        sig2noise_method="peak2peak",
        normalized_correlation=True,
        use_vectorized=True,
    )

    flow = np.asarray(
        extended_search_area_piv(
            jnp.asarray(frame_a)[None],
            jnp.asarray(frame_b)[None],
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_area_size,
            subpixel_method=subpixel_method,
        )
    )[0]
    u_jax, v_jax = flow[..., 0], flow[..., 1]

    np.testing.assert_array_equal(
        ~np.isfinite(u_jax), ~np.isfinite(np.asarray(u_ref))
    )
    finite = np.isfinite(u_jax) & np.isfinite(np.asarray(u_ref))
    assert finite.any()
    np.testing.assert_allclose(
        u_jax[finite], np.asarray(u_ref)[finite], atol=1e-2
    )
    np.testing.assert_allclose(
        v_jax[finite], np.asarray(v_ref)[finite], atol=1e-2
    )


# ---------------------------------------------------------------------------
# fft_correlate_images: linear correlation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window_size, overlap", [(32, 16), (24, 12), (16, 8)])
def test_fft_correlate_linear_matches_reference(window_size, overlap):
    """Linear (zero-padded) correlation matches the openpiv reference."""
    frame_a, frame_b = _shifted_pair(96, 96, shift_y=3, shift_x=-2, seed=0)
    aa = sliding_window_array(
        jnp.asarray(frame_a)[None],
        (window_size, window_size),
        (overlap, overlap),
    )[0]
    bb = sliding_window_array(
        jnp.asarray(frame_b)[None],
        (window_size, window_size),
        (overlap, overlap),
    )[0]

    got = np.asarray(fft_correlate_images(aa, bb, correlation_method="linear"))
    ref = pyprocess.fft_correlate_images(
        np.asarray(aa),
        np.asarray(bb),
        correlation_method="linear",
        normalized_correlation=True,
    )

    assert got.shape == np.asarray(ref).shape
    np.testing.assert_allclose(got, np.asarray(ref), atol=1e-5)


def test_fft_correlate_invalid_method_raises():
    """An unknown correlation method raises a clear ValueError."""
    win = jnp.ones((2, 16, 16))
    with pytest.raises(ValueError, match="Unknown correlation_method"):
        fft_correlate_images(win, win, correlation_method="quadratic")


@pytest.mark.parametrize(
    "window_size, search_area_size, overlap, shift_y, shift_x",
    [
        (32, 32, 16, 3, -2),
        (16, 32, 8, 2, 4),
        (24, 32, 12, -3, 1),
    ],
)
def test_pipeline_linear_correlation_matches_reference(
    window_size, search_area_size, overlap, shift_y, shift_x
):
    """End-to-end field with linear correlation matches the reference."""
    height = width = 96
    frame_a, frame_b = _shifted_pair(
        height, width, shift_y=shift_y, shift_x=shift_x, seed=6
    )
    u_ref, v_ref, _ = pyprocess.extended_search_area_piv(
        frame_a.copy(),
        frame_b.copy(),
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="linear",
        subpixel_method="gaussian",
        sig2noise_method="peak2peak",
        normalized_correlation=True,
        use_vectorized=True,
    )
    flow = np.asarray(
        extended_search_area_piv(
            jnp.asarray(frame_a)[None],
            jnp.asarray(frame_b)[None],
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_area_size,
            correlation_method="linear",
        )
    )[0]
    u_jax, v_jax = flow[..., 0], flow[..., 1]

    np.testing.assert_array_equal(
        ~np.isfinite(u_jax), ~np.isfinite(np.asarray(u_ref))
    )
    finite = np.isfinite(u_jax) & np.isfinite(np.asarray(u_ref))
    assert finite.any()
    np.testing.assert_allclose(
        u_jax[finite], np.asarray(u_ref)[finite], atol=1e-2
    )
    np.testing.assert_allclose(
        v_jax[finite], np.asarray(v_ref)[finite], atol=1e-2
    )


# ---------------------------------------------------------------------------
# extended_search_area_piv (full pipeline)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "window_size, search_area_size, overlap, shift_y, shift_x",
    [
        # standard FFT PIV: search area equals window
        (32, 32, 16, 3, -2),
        # extended search area: window smaller than search area
        (16, 32, 8, 2, 4),
        (24, 32, 12, -3, 1),
    ],
)
def test_extended_search_area_piv_matches_reference(
    window_size, search_area_size, overlap, shift_y, shift_x
):
    """End-to-end displacement field matches the openpiv reference pipeline."""
    height = width = 96
    frame_a, frame_b = _shifted_pair(
        height, width, shift_y=shift_y, shift_x=shift_x, pad=8, seed=3
    )

    # Reference: vectorized gaussian path with normalized correlation, which
    # is exactly what the JAX implementation reproduces.
    u_ref, v_ref, _ = pyprocess.extended_search_area_piv(
        frame_a.copy(),
        frame_b.copy(),
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="circular",
        subpixel_method="gaussian",
        sig2noise_method="peak2peak",
        normalized_correlation=True,
        use_vectorized=True,
    )

    flow = extended_search_area_piv(
        jnp.asarray(frame_a)[None],
        jnp.asarray(frame_b)[None],
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
    )
    flow = np.asarray(flow)[0]

    n_rows, n_cols = get_field_shape(
        (height, width),
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    assert flow.shape == (n_rows, n_cols, 2)
    assert u_ref.shape == (n_rows, n_cols)

    u_jax, v_jax = flow[..., 0], flow[..., 1]

    # Sub-pixel agreement with the reference on every window.
    finite = (
        np.isfinite(u_jax)
        & np.isfinite(v_jax)
        & np.isfinite(np.asarray(u_ref))
        & np.isfinite(np.asarray(v_ref))
    )
    assert finite.any()
    np.testing.assert_allclose(
        u_jax[finite], np.asarray(u_ref)[finite], atol=0.1
    )
    np.testing.assert_allclose(
        v_jax[finite], np.asarray(v_ref)[finite], atol=0.1
    )

    # The recovered field should reproduce the imposed uniform translation.
    np.testing.assert_allclose(np.median(u_jax[finite]), shift_x, atol=0.2)
    np.testing.assert_allclose(np.median(v_jax[finite]), shift_y, atol=0.2)


def test_extended_search_area_piv_batched_independent():
    """Batched estimation matches per-sample reference for distinct shifts."""
    height = width = 64
    window_size = search_area_size = 32
    overlap = 16
    shifts = [(2, 3), (-1, 4), (3, -2)]
    frames_a, frames_b, refs = [], [], []
    for idx, (sy, sx) in enumerate(shifts):
        fa, fb = _shifted_pair(height, width, sy, sx, pad=6, seed=10 + idx)
        frames_a.append(fa)
        frames_b.append(fb)
        u_ref, v_ref, _ = pyprocess.extended_search_area_piv(
            fa.copy(),
            fb.copy(),
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_area_size,
            correlation_method="circular",
            subpixel_method="gaussian",
            sig2noise_method="peak2peak",
            normalized_correlation=True,
            use_vectorized=True,
        )
        refs.append((u_ref, v_ref))

    flow = np.asarray(
        extended_search_area_piv(
            jnp.asarray(np.stack(frames_a)),
            jnp.asarray(np.stack(frames_b)),
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_area_size,
        )
    )

    for idx, (u_ref, v_ref) in enumerate(refs):
        u_jax, v_jax = flow[idx, ..., 0], flow[idx, ..., 1]
        finite = np.isfinite(u_jax) & np.isfinite(np.asarray(u_ref))
        np.testing.assert_allclose(
            u_jax[finite], np.asarray(u_ref)[finite], atol=0.1
        )
        np.testing.assert_allclose(
            v_jax[finite], np.asarray(v_ref)[finite], atol=0.1
        )


# ---------------------------------------------------------------------------
# replace_outliers
# ---------------------------------------------------------------------------
def test_replace_outliers_matches_openpiv_isolated():
    """Isolated-outlier replacement matches openpiv localmean on both channels.

    For a single invalid vector surrounded by valid neighbours, both the
    JAX local-mean iteration and openpiv's ``replace_outliers`` reduce to the
    mean of the eight neighbouring vectors, so they must agree numerically on
    *both* the u and v channels. This guards against the channel-mixing bug
    where a single ND convolution leaks values between components.

    Parity is limited to isolated *interior* outliers on purpose: the JAX
    kernel divides by a fixed ``(2 * kernel_size + 1)**2 - 1`` while the
    reference ``openpiv.lib.replace_nans`` divides by the number of valid,
    in-bounds neighbours, so outliers at the field border or in clusters
    (NaN neighbours within an iteration) intentionally diverge.
    """
    rng = np.random.RandomState(1)
    height = width = 12
    u = (rng.rand(height, width) * 5).astype(np.float64)
    v = (rng.rand(height, width) * 5).astype(np.float64)

    flags = np.zeros((height, width), dtype=bool)
    for i, j in [(2, 2), (2, 7), (6, 3), (8, 8), (4, 9)]:
        flags[i, j] = True

    u_ref, v_ref = filters.replace_outliers(
        u.copy(),
        v.copy(),
        flags.copy(),
        method="localmean",
        max_iter=30,
        kernel_size=1,
    )

    field = np.stack([u, v], axis=-1).astype(np.float32)
    field[flags] = np.nan
    out = np.asarray(
        replace_outliers(
            jnp.asarray(field)[None],
            jnp.asarray(flags)[None],
            30,
            1,
        )
    )[0]

    assert not np.isnan(out).any()
    # Valid vectors are left untouched.
    np.testing.assert_array_equal(out[~flags, 0], u[~flags].astype(np.float32))
    np.testing.assert_array_equal(out[~flags, 1], v[~flags].astype(np.float32))
    # Replaced vectors match openpiv on both channels.
    np.testing.assert_allclose(
        out[flags, 0], np.asarray(u_ref)[flags], atol=1e-4
    )
    np.testing.assert_allclose(
        out[flags, 1], np.asarray(v_ref)[flags], atol=1e-4
    )


def test_replace_outliers_no_channel_leak():
    """A constant u channel stays constant when only v has outliers.

    If the convolution mixed channels, replacing v outliers would perturb the
    u channel; with per-channel convolution the constant u field is exact.
    """
    height = width = 10
    u = np.full((height, width), 3.0, dtype=np.float32)
    rng = np.random.RandomState(5)
    v = (rng.rand(height, width) * 4).astype(np.float32)

    flags = np.zeros((height, width), dtype=bool)
    for i, j in [(3, 3), (5, 6), (7, 2)]:
        flags[i, j] = True

    field = np.stack([u, v], axis=-1)
    field[flags] = np.nan
    out = np.asarray(
        replace_outliers(
            jnp.asarray(field)[None], jnp.asarray(flags)[None], 30, 1
        )
    )[0]

    # u is constant everywhere, so a pure local mean must keep it at 3.0.
    np.testing.assert_allclose(out[..., 0], 3.0, atol=1e-5)


def test_replace_outliers_vmap_over_batch():
    """Each sample in a batch is filled independently of the others."""
    height = width = 8
    rng = np.random.RandomState(2)
    field = rng.rand(3, height, width, 2).astype(np.float32)
    flags = np.zeros((3, height, width), dtype=bool)
    flags[0, 4, 4] = True
    flags[2, 2, 5] = True

    nan_field = field.copy()
    nan_field[flags] = np.nan
    out = np.asarray(
        replace_outliers(jnp.asarray(nan_field), jnp.asarray(flags), 30, 1)
    )

    assert not np.isnan(out).any()
    # Sample 1 has no outliers and must be returned unchanged.
    np.testing.assert_allclose(out[1], field[1], atol=1e-6)
    # Valid positions in the other samples are unchanged too.
    valid = ~flags
    np.testing.assert_allclose(out[valid], field[valid], atol=1e-6)


# ---------------------------------------------------------------------------
# find_all_second_peaks / sig2noise_ratio helpers
# ---------------------------------------------------------------------------
def _pipeline_correlation(window_size=32, overlap=16, seed=0):
    """Correlation maps as produced by the JAX PIV pipeline, flattened.

    Returns a ``(n_windows, window_size, window_size)`` float32 array so the
    same maps can be fed verbatim to both implementations.
    """
    frame_a, frame_b = _shifted_pair(96, 96, shift_y=2, shift_x=-3, seed=seed)
    aa = sliding_window_array(
        jnp.asarray(frame_a)[None],
        (window_size, window_size),
        (overlap, overlap),
    )
    bb = sliding_window_array(
        jnp.asarray(frame_b)[None],
        (window_size, window_size),
        (overlap, overlap),
    )
    return fft_correlate_images(aa, bb)[0]


def _reference_flags(corr, sig2noise_method, width):
    """Recompute the validity flag the reference intends (but never applies).

    This mirrors the flag construction inside
    ``pyprocess.vectorized_sig2noise_ratio`` using the reference's own peak
    finders, so the parity tests can compare flag-free windows exactly and
    assert zeroing on the flagged ones.
    """
    ind1, peaks1 = pyprocess.find_all_first_peaks(corr)
    p1i, p1j = ind1[:, 1], ind1[:, 2]
    flag = (
        (peaks1 < 1e-3)
        | (p1i == 0)
        | (p1i == corr.shape[1] - 1)
        | (p1j == 0)
        | (p1j == corr.shape[2] - 1)
    )
    if sig2noise_method == "peak2peak":
        ind2, peaks2 = pyprocess.find_all_second_peaks(corr, width=width)
        p2i, p2j = ind2[:, 1], ind2[:, 2]
        flag = (
            flag
            | (peaks2 < 1e-3)
            | (p2i == 0)
            | (p2i == corr.shape[1] - 1)
            | (p2j == 0)
            | (p2j == corr.shape[2] - 1)
        )
    return np.asarray(flag, dtype=bool)


# ---------------------------------------------------------------------------
# find_all_second_peaks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1, 2, 3])
def test_find_all_second_peaks_matches_reference(width):
    """Second-peak indices and heights match the reference exactly."""
    corr = np.asarray(_pipeline_correlation(seed=1))

    ind_ref, peaks_ref = pyprocess.find_all_second_peaks(corr, width=width)
    p2i, p2j, peaks2 = find_all_second_peaks(jnp.asarray(corr), width=width)

    np.testing.assert_array_equal(np.asarray(p2i), ind_ref[:, 1])
    np.testing.assert_array_equal(np.asarray(p2j), ind_ref[:, 2])
    np.testing.assert_allclose(
        np.asarray(peaks2), np.asarray(peaks_ref), rtol=1e-6
    )


@pytest.mark.parametrize("width", [1, 2])
def test_find_all_second_peaks_random_maps(width):
    """Reference parity also holds on unstructured random maps."""
    rng = np.random.RandomState(7)
    corr = rng.rand(40, 24, 24).astype(np.float32)

    ind_ref, peaks_ref = pyprocess.find_all_second_peaks(corr, width=width)
    p2i, p2j, peaks2 = find_all_second_peaks(jnp.asarray(corr), width=width)

    np.testing.assert_array_equal(np.asarray(p2i), ind_ref[:, 1])
    np.testing.assert_array_equal(np.asarray(p2j), ind_ref[:, 2])
    np.testing.assert_allclose(
        np.asarray(peaks2), np.asarray(peaks_ref), rtol=1e-6
    )


def test_find_all_second_peaks_border_peak_box_is_clipped():
    """A first peak at the map border clips the exclusion box, as in openpiv."""
    corr = np.full((1, 16, 16), 0.1, dtype=np.float32)
    corr[0, 0, 0] = 1.0  # first peak in the corner
    corr[0, 8, 8] = 0.5  # second peak well outside the box

    ind_ref, peaks_ref = pyprocess.find_all_second_peaks(corr, width=2)
    p2i, p2j, peaks2 = find_all_second_peaks(jnp.asarray(corr), width=2)

    np.testing.assert_array_equal(np.asarray(p2i), ind_ref[:, 1])
    np.testing.assert_array_equal(np.asarray(p2j), ind_ref[:, 2])
    np.testing.assert_allclose(np.asarray(peaks2), np.asarray(peaks_ref))


# ---------------------------------------------------------------------------
# sig2noise_ratio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sig2noise_method", ["peak2peak", "peak2mean"])
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("window_size, overlap", [(32, 16), (16, 8), (64, 32)])
def test_sig2noise_matches_reference_on_same_corr(
    sig2noise_method, width, window_size, overlap
):
    """JAX s2n equals the vectorized reference on identical correlation maps.

    Both implementations consume the exact same maps, so unflagged windows
    must agree to float32 round-off; flagged windows must be zero on the JAX
    side (the reference leaves them untouched due to its no-op flag bug).
    """
    corr = np.asarray(
        _pipeline_correlation(window_size=window_size, overlap=overlap)
    )

    s2n_ref = pyprocess.vectorized_sig2noise_ratio(
        corr, sig2noise_method=sig2noise_method, width=width
    )
    s2n_jax = np.asarray(
        sig2noise_ratio(
            jnp.asarray(corr), sig2noise_method=sig2noise_method, width=width
        )
    )
    flags = _reference_flags(corr, sig2noise_method, width)

    assert s2n_jax.shape == np.asarray(s2n_ref).shape
    # The test must keep teeth: most windows are healthy on this data.
    assert (~flags).mean() > 0.5
    np.testing.assert_allclose(
        s2n_jax[~flags], np.asarray(s2n_ref)[~flags], rtol=1e-5
    )
    np.testing.assert_array_equal(s2n_jax[flags], 0.0)


def test_sig2noise_weak_first_peak_is_zeroed():
    """A near-flat map (peak < 1e-3) yields zero, as both references intend."""
    corr = np.full((3, 16, 16), 1e-5, dtype=np.float32)
    corr[:, 8, 8] = 5e-4  # below the 1e-3 signal threshold

    for method in ("peak2peak", "peak2mean"):
        s2n = np.asarray(
            sig2noise_ratio(jnp.asarray(corr), sig2noise_method=method)
        )
        np.testing.assert_array_equal(s2n, 0.0)
        # The loop-based reference applies the same rule.
        s2n_loop = pyprocess.sig2noise_ratio(
            corr.astype(np.float64), sig2noise_method=method
        )
        np.testing.assert_array_equal(np.asarray(s2n_loop), 0.0)


def test_sig2noise_border_first_peak_is_zeroed():
    """A first peak on the map border is flagged and zeroed."""
    corr = np.full((1, 16, 16), 0.1, dtype=np.float32)
    corr[0, 0, 5] = 1.0

    for method in ("peak2peak", "peak2mean"):
        s2n = np.asarray(
            sig2noise_ratio(jnp.asarray(corr), sig2noise_method=method)
        )
        np.testing.assert_array_equal(s2n, 0.0)
        s2n_loop = pyprocess.sig2noise_ratio(
            corr.astype(np.float64), sig2noise_method=method
        )
        np.testing.assert_array_equal(np.asarray(s2n_loop), 0.0)


def test_sig2noise_border_second_peak_is_zeroed():
    """peak2peak flags a second peak on the border (intended reference rule).

    The vectorized reference builds exactly this flag and then drops it on
    the floor (``flag is True`` no-op); the JAX port applies it.
    """
    corr = np.full((1, 16, 16), 0.1, dtype=np.float32)
    corr[0, 8, 8] = 1.0  # healthy first peak
    corr[0, 0, 3] = 0.9  # second peak on the border

    s2n = np.asarray(
        sig2noise_ratio(jnp.asarray(corr), sig2noise_method="peak2peak")
    )
    np.testing.assert_array_equal(s2n, 0.0)


# ---------------------------------------------------------------------------
# extended_search_area_piv with sig2noise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sig2noise_method", ["peak2peak", "peak2mean"])
@pytest.mark.parametrize(
    "window_size, search_area_size, overlap",
    [(32, 32, 16), (16, 32, 8)],
)
def test_pipeline_sig2noise_matches_reference(
    sig2noise_method, window_size, search_area_size, overlap
):
    """End-to-end s2n matches the openpiv pipeline on unflagged windows."""
    height = width = 96
    frame_a, frame_b = _shifted_pair(
        height, width, shift_y=2, shift_x=3, seed=5
    )

    _u_ref, _v_ref, s2n_ref = pyprocess.extended_search_area_piv(
        frame_a.copy(),
        frame_b.copy(),
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="circular",
        subpixel_method="gaussian",
        sig2noise_method=sig2noise_method,
        normalized_correlation=True,
        use_vectorized=True,
    )

    flow, s2n = extended_search_area_piv(
        jnp.asarray(frame_a)[None],
        jnp.asarray(frame_b)[None],
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        sig2noise_method=sig2noise_method,
    )
    s2n = np.asarray(s2n)[0]

    n_rows, n_cols = get_field_shape(
        (height, width),
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    assert flow.shape == (1, n_rows, n_cols, 2)
    assert s2n.shape == (n_rows, n_cols)
    assert np.asarray(s2n_ref).shape == (n_rows, n_cols)

    # Recompute the reference's intended flags on its own correlation maps to
    # exclude windows the reference fails to zero (no-op flag bug).
    aa = pyprocess.sliding_window_array(
        frame_a.astype(np.float32),
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    bb = pyprocess.sliding_window_array(
        frame_b.astype(np.float32),
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    if search_area_size > window_size:
        aa = pyprocess.normalize_intensity(aa)
        bb = pyprocess.normalize_intensity(bb)
        mask = np.zeros((search_area_size, search_area_size), dtype=aa.dtype)
        pad = (search_area_size - window_size) // 2
        mask[pad : search_area_size - pad, pad : search_area_size - pad] = 1
        aa = aa * np.broadcast_to(mask, aa.shape)
    corr_ref = pyprocess.fft_correlate_images(aa, bb)
    flags = _reference_flags(corr_ref, sig2noise_method, width=2).reshape(
        n_rows, n_cols
    )

    assert (~flags).mean() > 0.5
    np.testing.assert_allclose(
        s2n[~flags], np.asarray(s2n_ref)[~flags], rtol=1e-3
    )
    np.testing.assert_array_equal(s2n[flags], 0.0)
