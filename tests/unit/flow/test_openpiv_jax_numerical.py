"""Numerical equivalence tests for the JAX OpenPIV reimplementation.

Each test pins a JAX function against the original ``openpiv`` reference
implementation it mirrors, so the two stay numerically aligned:

- ``subpixel_displacement`` vs
  ``pyprocess.vectorized_correlation_to_displacements``
- ``extended_search_area_piv`` (full pipeline) vs
  ``pyprocess.extended_search_area_piv``
- ``replace_outliers`` vs ``openpiv.filters.replace_outliers``

The per-component tests (FFT correlation, peak finding, normalization,
sliding windows, coordinate grids) live in ``test_openpiv_jax``; this
module covers the sub-pixel stage, the end-to-end displacement field, and
outlier replacement.
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
    get_field_shape,
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
    "search_area_size, overlap",
    [(24, 12), (16, 8), (32, 16)],
)
def test_subpixel_displacement_matches_vectorized_reference(
    search_area_size, overlap
):
    """JAX sub-pixel peak fitting matches openpiv's vectorized routine."""
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
        corr_windows, peaks_i[0], peaks_j[0]
    )
    disp_vx = np.asarray(disp_vx)
    disp_vy = np.asarray(disp_vy)

    # Reference path: identical gaussian sub-pixel fit.
    corr_ref = np.asarray(corr_windows).astype(np.float32)
    u_ref, v_ref = pyprocess.vectorized_correlation_to_displacements(
        corr_ref, subpixel_method="gaussian"
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
