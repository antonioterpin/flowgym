"""JIT-compilation tests for the JAX OpenPIV reimplementation.

Every public building block of the JAX PIV pipeline is exercised under
``jax.jit`` to guarantee it traces cleanly, with window/overlap sizes
passed as static arguments where they determine output shapes. Each test
also checks that the compiled result is identical to the eager result, so
jitting never silently changes the numerics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.open_piv.openpiv_jax import (
    replace_invalid_single,
    replace_outliers,
)
from flowgym.flow.open_piv.process import (
    extended_search_area_piv,
    fft_correlate_images,
    find_all_first_peaks,
    get_field_shape,
    get_rect_coordinates,
    normalize_intensity,
    sig2noise_ratio,
    sig2noise_val,
    sliding_window_array,
    subpixel_displacement,
    upsample_flow,
)


def _assert_same(eager, compiled):
    """Assert two arrays match, treating NaNs as equal in matching positions."""
    eager = np.asarray(eager)
    compiled = np.asarray(compiled)
    assert eager.shape == compiled.shape
    np.testing.assert_array_equal(np.isnan(eager), np.isnan(compiled))
    mask = ~np.isnan(eager)
    np.testing.assert_allclose(
        eager[mask], compiled[mask], rtol=1e-6, atol=1e-6
    )


@pytest.fixture
def windows():
    """A small batch of correlation-sized windows."""
    rng = np.random.RandomState(0)
    return jnp.asarray(rng.rand(2, 9, 16, 16).astype(np.float32))


@pytest.fixture
def image_pair():
    rng = np.random.RandomState(1)
    img1 = jnp.asarray(rng.rand(2, 64, 64).astype(np.float32))
    img2 = jnp.asarray(rng.rand(2, 64, 64).astype(np.float32))
    return img1, img2


def test_normalize_intensity_jit(windows):
    """normalize_intensity traces and matches its eager output."""
    jitted = jax.jit(normalize_intensity)
    _assert_same(normalize_intensity(windows), jitted(windows))


@pytest.mark.parametrize("correlation_method", ["circular", "linear"])
def test_fft_correlate_images_jit(windows, correlation_method):
    """fft_correlate_images traces and matches eager for both methods."""
    jitted = jax.jit(fft_correlate_images, static_argnames="correlation_method")
    _assert_same(
        fft_correlate_images(
            windows, windows, correlation_method=correlation_method
        ),
        jitted(windows, windows, correlation_method=correlation_method),
    )


def test_find_all_first_peaks_jit(windows):
    """find_all_first_peaks traces and returns identical integer peaks."""
    corr = fft_correlate_images(windows, windows)
    jitted = jax.jit(find_all_first_peaks)
    pi_e, pj_e = find_all_first_peaks(corr)
    pi_c, pj_c = jitted(corr)
    _assert_same(pi_e, pi_c)
    _assert_same(pj_e, pj_c)


@pytest.mark.parametrize(
    "subpixel_method", ["gaussian", "parabolic", "centroid"]
)
def test_subpixel_displacement_jit(windows, subpixel_method):
    """subpixel_displacement traces and matches eager for every method."""
    corr = fft_correlate_images(windows, windows)[0]
    peaks_i, peaks_j = find_all_first_peaks(windows)
    jitted = jax.jit(subpixel_displacement, static_argnames="subpixel_method")
    vx_e, vy_e = subpixel_displacement(
        corr, peaks_i[0], peaks_j[0], subpixel_method=subpixel_method
    )
    vx_c, vy_c = jitted(
        corr, peaks_i[0], peaks_j[0], subpixel_method=subpixel_method
    )
    _assert_same(vx_e, vx_c)
    _assert_same(vy_e, vy_c)


def test_sliding_window_array_jit(image_pair):
    """sliding_window_array traces with static window/overlap sizes."""
    img1, _ = image_pair
    window_size, overlap = (16, 16), (8, 8)
    jitted = jax.jit(sliding_window_array, static_argnums=(1, 2))
    _assert_same(
        sliding_window_array(img1, window_size, overlap),
        jitted(img1, window_size, overlap),
    )


def test_get_rect_coordinates_jit():
    """get_rect_coordinates traces with fully static geometry."""
    image_size, window_size, overlap = (64, 64), (32, 32), (16, 16)
    jitted = jax.jit(get_rect_coordinates, static_argnums=(0, 1, 2))
    xs_e, ys_e = get_rect_coordinates(image_size, window_size, overlap)
    xs_c, ys_c = jitted(image_size, window_size, overlap)
    _assert_same(xs_e, xs_c)
    _assert_same(ys_e, ys_c)


def test_upsample_flow_jit():
    """upsample_flow traces with a static target shape."""
    rng = np.random.RandomState(0)
    flow = jnp.asarray(rng.rand(2, 8, 8, 2).astype(np.float32))
    jitted = jax.jit(upsample_flow, static_argnums=(1,))
    _assert_same(upsample_flow(flow, (32, 32)), jitted(flow, (32, 32)))


def test_replace_invalid_single_jit():
    """replace_invalid_single traces with a static iteration count."""
    from flowgym.common.filters import uniform_kernel

    rng = np.random.RandomState(0)
    field = rng.rand(12, 12, 2).astype(np.float32)
    flags = np.zeros((12, 12), dtype=bool)
    flags[4, 4] = flags[7, 8] = True
    field[flags] = np.nan
    kernel = uniform_kernel(kernel_size=1, n_channels=2)

    jitted = jax.jit(replace_invalid_single, static_argnums=(3,))
    _assert_same(
        replace_invalid_single(
            jnp.asarray(field), jnp.asarray(flags), kernel, 20
        ),
        jitted(jnp.asarray(field), jnp.asarray(flags), kernel, 20),
    )


def test_replace_outliers_jit():
    """replace_outliers traces with static iteration count and kernel size."""
    rng = np.random.RandomState(0)
    field = rng.rand(3, 12, 12, 2).astype(np.float32)
    flags = np.zeros((3, 12, 12), dtype=bool)
    flags[0, 4, 4] = flags[2, 6, 7] = True
    field[flags] = np.nan

    jitted = jax.jit(replace_outliers, static_argnums=(2, 3))
    _assert_same(
        replace_outliers(jnp.asarray(field), jnp.asarray(flags), 20, 1),
        jitted(jnp.asarray(field), jnp.asarray(flags), 20, 1),
    )


@pytest.mark.parametrize(
    "window_size, search_area_size, overlap",
    [(32, 32, 16), (16, 32, 8)],
)
def test_extended_search_area_piv_jit(
    image_pair, window_size, search_area_size, overlap
):
    """The full PIV pipeline traces with static window geometry."""
    img1, img2 = image_pair
    jitted = jax.jit(
        extended_search_area_piv,
        static_argnames=("window_size", "search_area_size", "overlap"),
    )
    eager = extended_search_area_piv(
        img1,
        img2,
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap,
    )
    compiled = jitted(
        img1,
        img2,
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap,
    )
    n_rows, n_cols = get_field_shape(
        (img1.shape[1], img1.shape[2]),
        (search_area_size, search_area_size),
        (overlap, overlap),
    )
    assert eager.shape == (img1.shape[0], n_rows, n_cols, 2)
    _assert_same(eager, compiled)


def test_extended_search_area_piv_rectangular_jit(image_pair):
    """The pipeline traces with static rectangular (tuple) window geometry."""
    img1, img2 = image_pair
    window_size, search_area_size, overlap = (16, 32), (24, 32), (8, 16)
    jitted = jax.jit(
        extended_search_area_piv,
        static_argnames=("window_size", "search_area_size", "overlap"),
    )
    eager = extended_search_area_piv(
        img1,
        img2,
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap,
    )
    compiled = jitted(
        img1,
        img2,
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap,
    )
    n_rows, n_cols = get_field_shape(
        (img1.shape[1], img1.shape[2]), search_area_size, overlap
    )
    assert eager.shape == (img1.shape[0], n_rows, n_cols, 2)
    _assert_same(eager, compiled)


def test_extended_search_area_piv_jit_no_recompile(image_pair):
    """Re-calling the jitted pipeline with new data does not retrace."""
    img1, img2 = image_pair
    jitted = jax.jit(
        extended_search_area_piv,
        static_argnames=("window_size", "search_area_size", "overlap"),
    )
    lowered = jitted.lower(
        img1, img2, window_size=32, search_area_size=32, overlap=16
    )
    compiled = lowered.compile()

    # A second batch with the same shapes reuses the compiled executable.
    rng = np.random.RandomState(9)
    img1b = jnp.asarray(rng.rand(*img1.shape).astype(np.float32))
    img2b = jnp.asarray(rng.rand(*img2.shape).astype(np.float32))
    out = compiled(img1b, img2b)
    ref = extended_search_area_piv(
        img1b, img2b, window_size=32, search_area_size=32, overlap=16
    )
    _assert_same(ref, out)


@pytest.mark.parametrize("sig2noise_method", ["peak2peak", "peak2mean"])
def test_sig2noise_ratio_jit(windows, sig2noise_method):
    """sig2noise_ratio traces under jit and matches its eager output."""
    corr = fft_correlate_images(windows, windows)
    jitted = jax.jit(
        sig2noise_ratio, static_argnames=("sig2noise_method", "width")
    )
    _assert_same(
        sig2noise_ratio(corr, sig2noise_method=sig2noise_method, width=2),
        jitted(corr, sig2noise_method=sig2noise_method, width=2),
    )


def test_sig2noise_val_jit(windows):
    """sig2noise_val traces under jit and matches its eager output."""
    s2n = sig2noise_ratio(fft_correlate_images(windows, windows))
    jitted = jax.jit(sig2noise_val, static_argnames="threshold")
    np.testing.assert_array_equal(
        np.asarray(sig2noise_val(s2n, threshold=1.0)),
        np.asarray(jitted(s2n, threshold=1.0)),
    )


def test_extended_search_area_piv_sig2noise_jit(image_pair):
    """The pipeline with sig2noise traces with static method and geometry."""
    img1, img2 = image_pair
    jitted = jax.jit(
        extended_search_area_piv,
        static_argnames=(
            "window_size",
            "search_area_size",
            "overlap",
            "sig2noise_method",
            "width",
        ),
    )
    flow_e, s2n_e = extended_search_area_piv(
        img1,
        img2,
        window_size=32,
        search_area_size=32,
        overlap=16,
        sig2noise_method="peak2peak",
    )
    flow_c, s2n_c = jitted(
        img1,
        img2,
        window_size=32,
        search_area_size=32,
        overlap=16,
        sig2noise_method="peak2peak",
    )
    _assert_same(flow_e, flow_c)
    _assert_same(s2n_e, s2n_c)
