"""Numerical equivalence tests for JAX window deformation.

``deform_windows`` is pinned against openpiv's ``windef.deform_windows`` at
linear interpolation (``interpolation_order = interpolation_order2 = 1``),
which is the configuration the JAX implementation reproduces. openpiv's
default cubic field interpolation (``RectBivariateSpline`` degree 3) has no
exact JAX equivalent and is intentionally out of scope.

``multipass_deform`` (fixed-resolution iterative window deformation) is
pinned against a reference loop built from openpiv's own primitives
(``windef.deform_windows`` + ``pyprocess.extended_search_area_piv``) using
the same ``deformation_method="second image"`` recipe: deform the second
image by the current estimate, re-correlate for the residual, accumulate.
Because each primitive matches openpiv exactly, the whole loop does too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.ndimage as scn
from openpiv import pyprocess, windef

from flowgym.flow.open_piv.process import (
    deform_windows,
    get_field_shape,
    get_rect_coordinates,
    multipass_deform,
)


def _grid_and_flow(height, width, n_rows, n_cols, seed):
    """Build a window-centre meshgrid and a random displacement field."""
    rng = np.random.RandomState(seed)
    y1 = np.linspace(8, height - 8, n_rows)
    x1 = np.linspace(8, width - 8, n_cols)
    x, y = np.meshgrid(x1, y1)
    u = (rng.randn(n_rows, n_cols) * 2.0).astype(np.float64)
    v = (rng.randn(n_rows, n_cols) * 2.0).astype(np.float64)
    return x, y, u, v


@pytest.mark.parametrize(
    "height, width, n_rows, n_cols",
    [
        (48, 48, 4, 4),
        (64, 80, 5, 6),
        (72, 56, 6, 4),
        (96, 96, 5, 5),
    ],
)
def test_deform_windows_matches_reference(height, width, n_rows, n_cols):
    """JAX deform_windows matches openpiv at linear interpolation."""
    rng = np.random.RandomState(height * width + n_rows)
    frame = rng.rand(height, width).astype(np.float32)
    x, y, u, v = _grid_and_flow(height, width, n_rows, n_cols, seed=n_cols)

    ref = windef.deform_windows(
        frame.copy(),
        x,
        y,
        u,
        v,
        interpolation_order=1,
        interpolation_order2=1,
    )
    got = np.asarray(
        deform_windows(
            jnp.asarray(frame),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(u),
            jnp.asarray(v),
        )
    )

    assert got.shape == np.asarray(ref).shape
    np.testing.assert_allclose(got, np.asarray(ref), atol=1e-4)


def test_deform_windows_zero_flow_is_identity():
    """A zero displacement field returns the input image unchanged."""
    rng = np.random.RandomState(0)
    frame = rng.rand(64, 64).astype(np.float32)
    x, y, u, v = _grid_and_flow(64, 64, 5, 5, seed=1)
    u[:] = 0.0
    v[:] = 0.0
    got = np.asarray(
        deform_windows(
            jnp.asarray(frame),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(u),
            jnp.asarray(v),
        )
    )
    np.testing.assert_allclose(got, frame, atol=1e-5)


def test_deform_windows_constant_flow_matches_reference():
    """A spatially uniform shift agrees with the reference everywhere."""
    rng = np.random.RandomState(2)
    frame = rng.rand(64, 64).astype(np.float32)
    x, y, u, v = _grid_and_flow(64, 64, 5, 5, seed=3)
    u[:] = 1.5
    v[:] = -2.0

    ref = windef.deform_windows(
        frame.copy(), x, y, u, v, interpolation_order=1, interpolation_order2=1
    )
    got = np.asarray(
        deform_windows(
            jnp.asarray(frame),
            jnp.asarray(x),
            jnp.asarray(y),
            jnp.asarray(u),
            jnp.asarray(v),
        )
    )
    np.testing.assert_allclose(got, np.asarray(ref), atol=1e-4)


def test_deform_windows_jit():
    """deform_windows traces under jit and matches its eager output."""
    rng = np.random.RandomState(4)
    frame = jnp.asarray(rng.rand(64, 64).astype(np.float32))
    x, y, u, v = _grid_and_flow(64, 64, 5, 5, seed=5)
    x, y = jnp.asarray(x), jnp.asarray(y)
    u, v = jnp.asarray(u), jnp.asarray(v)

    eager = deform_windows(frame, x, y, u, v)
    compiled = jax.jit(deform_windows)(frame, x, y, u, v)
    # Chained map_coordinates fuse differently under XLA; the eager/compiled
    # gap is pure float32 round-off (~1e-6).
    np.testing.assert_allclose(
        np.asarray(eager), np.asarray(compiled), rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# multipass_deform
# ---------------------------------------------------------------------------
def _warped_pair(height, width, seed):
    """Build (img1, img2, true_dx, true_dy) for a smooth non-uniform flow.

    img2 is img1 resampled at ``(y - V, x + U)``, so the displacement of
    img1 -> img2 is ``dx = -U``, ``dy = +V``.
    """
    rng = np.random.RandomState(seed)
    img1 = rng.rand(height, width).astype(np.float32)
    sx = np.arange(width)
    sy = np.arange(height)
    grid_x, grid_y = np.meshgrid(sx, sy)
    big_u = 2.0 * np.sin(2 * np.pi * grid_y / height)
    big_v = 1.5 * np.cos(2 * np.pi * grid_x / width)
    img2 = scn.map_coordinates(
        img1, [grid_y - big_v, grid_x + big_u], order=1, mode="nearest"
    ).astype(np.float32)
    return img1, img2, big_u, big_v


def _reference_multipass(img1, img2, ws, sas, ov, n_passes):
    """Reference loop using openpiv primitives, same recipe and fixed grid."""
    xs, ys = get_rect_coordinates(
        (img1.shape[0], img1.shape[1]), (sas, sas), (ov, ov)
    )
    n_rows, n_cols = get_field_shape(
        (img1.shape[0], img1.shape[1]), (sas, sas), (ov, ov)
    )
    x = np.asarray(xs).reshape(n_rows, n_cols)
    y = np.asarray(ys).reshape(n_rows, n_cols)

    def one_pass(frame_a, frame_b):
        u, v, _ = pyprocess.extended_search_area_piv(
            frame_a.copy(),
            frame_b.copy(),
            window_size=ws,
            overlap=ov,
            search_area_size=sas,
            correlation_method="circular",
            subpixel_method="gaussian",
            sig2noise_method="peak2peak",
            normalized_correlation=True,
            use_vectorized=True,
        )
        return np.nan_to_num(np.asarray(u)), np.nan_to_num(np.asarray(v))

    u, v = one_pass(img1, img2)
    for _ in range(n_passes - 1):
        deformed = windef.deform_windows(
            img2.copy(),
            x,
            y,
            u,
            -v,
            interpolation_order=1,
            interpolation_order2=1,
        )
        du, dv = one_pass(img1, deformed)
        u = u + du
        v = v + dv
    return u, v


@pytest.mark.parametrize("n_passes", [2, 3, 4])
def test_multipass_matches_openpiv_primitive_loop(n_passes):
    """JAX multipass matches the openpiv-primitive reference loop."""
    img1, img2, _, _ = _warped_pair(96, 96, seed=7)
    ws = sas = 32
    ov = 16

    ref_u, ref_v = _reference_multipass(img1, img2, ws, sas, ov, n_passes)
    flow = np.asarray(
        multipass_deform(
            jnp.asarray(img1)[None],
            jnp.asarray(img2)[None],
            window_size=ws,
            search_area_size=sas,
            overlap=ov,
            n_passes=n_passes,
        )
    )[0]

    np.testing.assert_allclose(flow[..., 0], ref_u, atol=1e-3)
    np.testing.assert_allclose(flow[..., 1], ref_v, atol=1e-3)


def test_multipass_single_pass_equals_extended_search():
    """n_passes=1 returns exactly the single-pass result."""
    from flowgym.flow.open_piv.process import extended_search_area_piv

    img1, img2, _, _ = _warped_pair(96, 96, seed=2)
    a, b = jnp.asarray(img1)[None], jnp.asarray(img2)[None]
    kwargs = {"window_size": 32, "search_area_size": 32, "overlap": 16}
    single = np.asarray(extended_search_area_piv(a, b, **kwargs))
    multi1 = np.asarray(multipass_deform(a, b, n_passes=1, **kwargs))
    np.testing.assert_array_equal(np.isnan(single), np.isnan(multi1))
    mask = ~np.isnan(single)
    np.testing.assert_array_equal(single[mask], multi1[mask])


def test_multipass_improves_accuracy():
    """Iterative deformation measurably improves the single-pass estimate.

    On a smooth non-uniform flow the first deformation (pass 2) is where the
    method earns its keep and must cut the single-pass median endpoint error
    by a solid margin. Further passes must not blow up past the single pass:
    without between-pass outlier replacement the error is non-monotonic and
    creeps back up (it peaks at pass 2 and worsens after, see #65), so the
    pass-3 bound is a no-worse guard rather than strict improvement. Error is
    measured against the true displacement (dx = -U, dy = +V).
    """
    img1, img2, big_u, big_v = _warped_pair(128, 128, seed=0)
    ws = sas = 32
    ov = 16
    xs, ys = get_rect_coordinates((128, 128), (sas, sas), (ov, ov))
    n_rows, n_cols = get_field_shape((128, 128), (sas, sas), (ov, ov))
    x = np.asarray(xs).reshape(n_rows, n_cols).astype(int)
    y = np.asarray(ys).reshape(n_rows, n_cols).astype(int)
    true_dx = -big_u[y, x]
    true_dy = big_v[y, x]

    def endpoint_err(flow):
        u, v = flow[..., 0], flow[..., 1]
        m = np.isfinite(u) & np.isfinite(v)
        return np.nanmedian(
            np.sqrt((u[m] - true_dx[m]) ** 2 + (v[m] - true_dy[m]) ** 2)
        )

    a, b = jnp.asarray(img1)[None], jnp.asarray(img2)[None]
    kwargs = {"window_size": ws, "search_area_size": sas, "overlap": ov}
    err1 = endpoint_err(
        np.asarray(multipass_deform(a, b, n_passes=1, **kwargs))[0]
    )
    err2 = endpoint_err(
        np.asarray(multipass_deform(a, b, n_passes=2, **kwargs))[0]
    )
    err3 = endpoint_err(
        np.asarray(multipass_deform(a, b, n_passes=3, **kwargs))[0]
    )
    # Pass 2 (the first deformation) must cut the single-pass error by a solid
    # margin (measured ~0.21 -> 0.13). The old `err3 <= err1 + 1e-3` bound
    # caught only catastrophic regressions.
    assert err2 <= 0.8 * err1, f"no improvement: {err1:.4f} -> {err2:.4f}"
    # Later passes must not drift past the single pass (see #65).
    assert err3 <= err1, f"drifted past single pass: {err1:.4f} -> {err3:.4f}"


def test_multipass_jit():
    """multipass_deform traces under jit (static geometry and pass count)."""
    img1, img2, _, _ = _warped_pair(64, 64, seed=3)
    a, b = jnp.asarray(img1)[None], jnp.asarray(img2)[None]
    jitted = jax.jit(
        multipass_deform,
        static_argnames=(
            "window_size",
            "search_area_size",
            "overlap",
            "n_passes",
        ),
    )
    eager = multipass_deform(
        a, b, window_size=32, search_area_size=32, overlap=16, n_passes=3
    )
    compiled = jitted(
        a, b, window_size=32, search_area_size=32, overlap=16, n_passes=3
    )
    np.testing.assert_allclose(
        np.asarray(eager), np.asarray(compiled), rtol=1e-5, atol=1e-5
    )
