"""Unit tests for flowgym.flow.open_piv.

Covers FFT correlation, peak finding, intensity normalization, sliding
windows, coordinate grids, sub-pixel displacement, and flow upsampling
against the reference openpiv.pyprocess implementation.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from openpiv import pyprocess

from flowgym.flow.open_piv.process import (
    extended_search_area_piv,
    fft_correlate_images,
    find_all_first_peaks,
    get_field_shape,
    get_rect_coordinates,
    normalize_intensity,
    sig2noise_ratio,
    sliding_window_array,
    subpixel_displacement,
    upsample_flow,
)


@pytest.fixture
def random_images():
    rng = np.random.RandomState(0)
    batch_size, H, W = 4, 64, 64
    img1 = rng.randint(0, 256, size=(batch_size, H, W)).astype(np.int32)
    img2 = rng.randint(0, 256, size=(batch_size, H, W)).astype(np.int32)
    return img1, img2


# Test for fft_correlate_images with identical inputs
def test_fft_correlate_images_identical():
    """fft_correlate_images returns zero correlation for constant windows."""
    win = jnp.ones((1, 2, 2))
    corr = fft_correlate_images(win, win)
    assert corr.shape == (1, 2, 2)
    # With constant windows, normalized to zero, correlation yields zeros
    assert jnp.allclose(corr, 0.0)


# Test for find_all_first_peaks
def test_find_all_first_peaks_basic():
    """find_all_first_peaks locates the single injected peak correctly."""
    corr = jnp.zeros((1, 1, 2, 3))
    corr = corr.at[0, 0, 1, 2].set(5)
    peaks_i, peaks_j = find_all_first_peaks(corr)
    assert peaks_i[0, 0] == 1 and peaks_j[0, 0] == 2


# Test for normalize_intensity
def test_normalize_intensity_basic():
    """normalize_intensity zeros out windows with zero standard deviation."""
    windows = jnp.array([[[1, 2], [3, 4]], [[5, 5], [5, 5]]], dtype=jnp.float32)
    normed = normalize_intensity(windows)
    # First window mean=2.5, std~=1.118, values clipped between 0 and max
    assert normed.shape == windows.shape
    # Second window has zero std, should be zeros
    assert jnp.all(normed[1] == 0)


# Test for subpixel_displacement at center peak
def test_subpixel_displacement_center():
    """subpixel_displacement returns zero shift for a centered peak."""
    corr = jnp.zeros((1, 3, 3))
    corr = corr.at[0, 1, 1].set(1.0)
    peaks_i = jnp.array([1])
    peaks_j = jnp.array([1])
    vx, vy = subpixel_displacement(corr, peaks_i, peaks_j)
    assert vx.shape == (1,) and vy.shape == (1,)
    # Expect displacement relative to center (floor(3/2)=1) => shift=0 gives 0
    assert jnp.allclose(vx, 0.0) and jnp.allclose(vy, 0.0)


# Test for upsample_flow
def test_upsample_flow():
    """upsample_flow resizes a zero flow field to the target shape."""
    flow = jnp.zeros((1, 2, 2, 2))
    up = upsample_flow(flow, (4, 4))
    assert up.shape == (1, 4, 4, 2)
    assert jnp.all(up == 0)


@pytest.mark.parametrize(
    "image_size, search_area_size, overlap",
    [
        # square image, square windows
        ((64, 64), (32, 32), (16, 16)),
        # rectangular image, square windows
        ((100, 200), (50, 50), (25, 25)),
        # rectangular image, rectangular windows
        ((120, 80), (30, 20), (10, 5)),
        # no overlap (should tile perfectly)
        ((64, 64), (16, 16), (0, 0)),
        # window == image (just one vector)
        ((64, 64), (64, 64), (0, 0)),
    ],
)
def test_get_field_shape_parametrized(image_size, search_area_size, overlap):
    """get_field_shape matches pyprocess for various window configs."""
    expected = tuple(
        pyprocess.get_field_shape(image_size, search_area_size, overlap)
    )
    got = get_field_shape(image_size, search_area_size, overlap)
    assert got == expected, (
        f"mismatch for image_size={image_size}, "
        f"search_area_size={search_area_size}, overlap={overlap}: "
        f"expected {expected}, got {got}"
    )


def test_get_field_shape_with_random_images(random_images):
    """get_field_shape matches pyprocess reference on random batch images."""
    img1, _ = random_images
    # take the shape of one frame (batch dim ignored)
    H, W = img1.shape[1], img1.shape[2]

    # pick some arbitrary window / overlap
    search_area = (32, 32)
    overlap = (16, 16)

    expected = tuple(pyprocess.get_field_shape((H, W), search_area, overlap))
    got = get_field_shape((H, W), search_area, overlap)
    assert got == expected, f"on random data: expected {expected}, got {got}"


@pytest.mark.parametrize(
    "image_size, window_size, overlap, expected_centers",
    [
        # simple square case: 3x3 grid at [16, 32, 48] in both dims
        (
            (64, 64),
            (32, 32),
            (16, 16),
            {
                "xs": np.array([16, 32, 48]),
                "ys": np.array([16, 32, 48]),
            },
        ),
        # rectangular image, non-square windows
        (
            (100, 80),
            (20, 10),
            (5, 2),
            {
                # sh = 20-5=15 ⇒ ny=(100-20)//15+1 = 6
                # offsets: 10*0.5=10, then +15, …
                "ys": np.arange(6) * 15 + 10,
                # sw = 10-2=8 ⇒ nx=(80 - 10)//8+1 = 9
                # offsets: 10*0.5=5, then +8, …
                "xs": np.arange(9) * 8 + 5,
            },
        ),
        # no overlap → perfectly tiled
        (
            (50, 30),
            (10, 5),
            (0, 0),
            {
                "ys": np.arange((50 - 10) // 10 + 1) * 10 + 5,
                "xs": np.arange((30 - 5) // 5 + 1) * 5 + 2.5,
            },
        ),
    ],
)
def test_get_rect_coordinates_parametrized(
    image_size, window_size, overlap, expected_centers
):
    """get_rect_coordinates returns a uniform grid of patch centre locations."""
    xs, ys = get_rect_coordinates(image_size, window_size, overlap)

    # reshape back to grid for easier comparison
    ny, nx = (
        (image_size[0] - window_size[0]) // (window_size[0] - overlap[0]) + 1,
        (image_size[1] - window_size[1]) // (window_size[1] - overlap[1]) + 1,
    )
    x = xs.reshape(ny, nx)
    y = ys.reshape(ny, nx)

    # check first row/column match expected offsets
    np.testing.assert_allclose(x[0], expected_centers["xs"], rtol=1e-6)
    np.testing.assert_allclose(y[:, 0], expected_centers["ys"], rtol=1e-6)

    # check meshgrid property: every row of Y is constant, every column
    # of X is constant
    for i in range(ny):
        assert np.allclose(y[i, :], expected_centers["ys"][i], rtol=1e-6)
    for j in range(nx):
        assert np.allclose(x[:, j], expected_centers["xs"][j], rtol=1e-6)


def test_number_of_centers_matches_field_shape(
    image_size=(64, 64), window_size=(32, 32), overlap=(16, 16)
):
    """get_rect_coordinates centre count equals rows * cols."""
    # compute num rows/cols
    n_rows, n_cols = get_field_shape(image_size, window_size, overlap)
    xs, ys = get_rect_coordinates(image_size, window_size, overlap)

    # total number of centers must equal n_rows * n_cols
    assert xs.size == n_rows * n_cols
    assert ys.size == n_rows * n_cols


def test_get_rect_coordinates_with_random_images(random_images):
    """Rectangular window centers match pyprocess for random image sizes."""
    img1, _ = random_images
    H, W = img1.shape[1], img1.shape[2]
    image_size = (H, W)

    # use any window/overlap that fits
    window_size = (32, 32)
    overlap = (16, 16)

    xs, ys = get_rect_coordinates(image_size, window_size, overlap)
    x_ref, y_ref = pyprocess.get_rect_coordinates(
        image_size, window_size, overlap
    )

    np.testing.assert_allclose(xs, x_ref.flatten(), rtol=1e-6)
    np.testing.assert_allclose(ys, y_ref.flatten(), rtol=1e-6)


def test_normalize_intensity(random_images):
    """JAX intensity normalization matches pyprocess for each image."""
    img1, img2 = random_images[0], random_images[1]
    for img in np.vstack((img1, img2)):
        normalized = normalize_intensity(jnp.asarray(img)[None, ...])[0, ...]
        normalized_gt = pyprocess.normalize_intensity(img[:, :])
        np.testing.assert_allclose(normalized, normalized_gt, rtol=1e-6)


def test_sliding_window(random_images):
    """Sliding-window extraction matches pyprocess shapes and values."""
    img1, img2 = random_images[0], random_images[1]
    search_area_size = (16, 16)
    overlap = (15, 15)

    aa = sliding_window_array(img1, search_area_size, overlap)
    bb = sliding_window_array(img2, search_area_size, overlap)
    sol = np.concatenate((aa, bb), axis=0)

    n_rows, n_cols = get_field_shape(img1.shape[1:], search_area_size, overlap)

    assert sol.shape == (img1.shape[0] * 2, n_rows * n_cols, *search_area_size)

    for i, sj in zip(np.concatenate((img1, img2), axis=0), sol, strict=True):
        assert i.shape == img1.shape[1:]
        assert sj.shape == (n_rows * n_cols, *search_area_size)

        # compute ground truth using pyprocess
        gt = pyprocess.sliding_window_array(i, search_area_size, overlap)

        # check shape
        assert sj.shape == gt.shape, f"Shape mismatch: {sj.shape} != {gt.shape}"
        # check values
        np.testing.assert_allclose(np.asarray(sj), gt, rtol=1e-6)


def test_fft_correlate_images(random_images):
    """FFT cross-correlation matches pyprocess for each window."""
    img1, img2 = random_images[0], random_images[1]
    search_area_size = (16, 16)
    overlap = (15, 15)

    aa = sliding_window_array(img1, search_area_size, overlap)
    bb = sliding_window_array(img2, search_area_size, overlap)

    corr = fft_correlate_images(aa, bb)
    for c, a, b in zip(corr, aa, bb, strict=True):
        for cw, aw, bw in zip(c, a, b, strict=True):
            assert cw.shape == search_area_size
            assert aw.shape == search_area_size
            assert bw.shape == search_area_size
            corr_gt = pyprocess.fft_correlate_images(
                np.asarray(aw), np.asarray(bw)
            )
            np.testing.assert_allclose(cw, corr_gt, rtol=1e-6)


def test_find_all_first_peaks(random_images):
    """First-peak detection returns one peak per correlation window."""
    img1, img2 = random_images[0], random_images[1]
    search_area_size = (16, 16)
    overlap = (15, 15)

    aa = sliding_window_array(img1, search_area_size, overlap)
    bb = sliding_window_array(img2, search_area_size, overlap)

    corr = fft_correlate_images(aa, bb)
    peaks_i, peaks_j = find_all_first_peaks(corr)

    n_rows, n_cols = get_field_shape(img1.shape[1:], search_area_size, overlap)
    for c, pi, pj in zip(corr, peaks_i, peaks_j, strict=True):
        assert c.shape == (n_rows * n_cols, *search_area_size)
        assert pi.shape == (n_rows * n_cols,)
        assert pj.shape == (n_rows * n_cols,)
        idx = pyprocess.find_all_first_peaks(np.asarray(c))[0]
        peaks_i_gt = idx[:, 1]
        peaks_j_gt = idx[:, 2]

        np.testing.assert_allclose(pi, peaks_i_gt, rtol=1e-6)
        np.testing.assert_allclose(pj, peaks_j_gt, rtol=1e-6)


def test_sig2noise_batched_leading_dims(random_images):
    """sig2noise_ratio vectorizes over arbitrary leading dimensions."""
    img1, img2 = random_images
    aa = sliding_window_array(
        jnp.asarray(img1, dtype=jnp.float32), (32, 32), (16, 16)
    )
    bb = sliding_window_array(
        jnp.asarray(img2, dtype=jnp.float32), (32, 32), (16, 16)
    )
    corr = fft_correlate_images(aa, bb)  # (batch, n_windows, H, W)

    for method in ("peak2peak", "peak2mean"):
        out = np.asarray(sig2noise_ratio(corr, sig2noise_method=method))
        flat = np.asarray(
            sig2noise_ratio(
                corr.reshape(-1, *corr.shape[-2:]), sig2noise_method=method
            )
        )
        assert out.shape == corr.shape[:-2]
        np.testing.assert_allclose(out.reshape(-1), flat, rtol=1e-6)


def test_sig2noise_invalid_method_raises():
    """An unknown method raises ValueError, mirroring the reference."""
    corr = jnp.ones((1, 8, 8))
    with pytest.raises(ValueError, match="sig2noise_method"):
        sig2noise_ratio(corr, sig2noise_method="peak2median")


def test_extended_search_area_piv_flow_only_return(random_images):
    """Omitting sig2noise_method preserves the original single-array API."""
    img1, img2 = random_images
    flow = extended_search_area_piv(
        jnp.asarray(img1, dtype=jnp.float32),
        jnp.asarray(img2, dtype=jnp.float32),
        window_size=32,
        overlap=16,
        search_area_size=32,
    )
    assert isinstance(flow, jnp.ndarray)
    assert flow.shape[-1] == 2


def test_correlation_method_default_is_circular(random_images):
    """The default correlation method matches an explicit 'circular' request."""
    img1, img2 = random_images
    a = jnp.asarray(img1, dtype=jnp.float32)
    b = jnp.asarray(img2, dtype=jnp.float32)
    kwargs = {"window_size": 32, "overlap": 16, "search_area_size": 32}
    default = np.asarray(extended_search_area_piv(a, b, **kwargs))
    explicit = np.asarray(
        extended_search_area_piv(a, b, correlation_method="circular", **kwargs)
    )
    np.testing.assert_array_equal(np.isnan(default), np.isnan(explicit))
    mask = ~np.isnan(default)
    np.testing.assert_array_equal(default[mask], explicit[mask])


def test_subpixel_method_default_is_gaussian(random_images):
    """The default sub-pixel method matches an explicit 'gaussian' request."""
    img1, img2 = random_images
    a = jnp.asarray(img1, dtype=jnp.float32)
    b = jnp.asarray(img2, dtype=jnp.float32)
    kwargs = {"window_size": 32, "overlap": 16, "search_area_size": 32}
    default = np.asarray(extended_search_area_piv(a, b, **kwargs))
    explicit = np.asarray(
        extended_search_area_piv(a, b, subpixel_method="gaussian", **kwargs)
    )
    np.testing.assert_array_equal(np.isnan(default), np.isnan(explicit))
    mask = ~np.isnan(default)
    np.testing.assert_array_equal(default[mask], explicit[mask])
