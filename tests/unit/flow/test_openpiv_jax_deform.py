"""Numerical equivalence tests for JAX window deformation.

``deform_windows`` is pinned against openpiv's ``windef.deform_windows`` at
linear interpolation (``interpolation_order = interpolation_order2 = 1``),
which is the configuration the JAX implementation reproduces. openpiv's
default cubic field interpolation (``RectBivariateSpline`` degree 3) has no
exact JAX equivalent and is intentionally out of scope.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openpiv import windef

from flowgym.flow.open_piv.process import deform_windows


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
