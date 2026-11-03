"""Unit tests for the LIMA processing ops (warp, correlation, padding)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.lima.process import (
    bilinear_warp,
    coords_grid,
    local_correlation,
    pad_for_conv,
    symmetric_warp,
)


def test_coords_grid_xy_order():
    """coords_grid returns (x, y) pixel coordinates in NHWC layout."""
    grid = coords_grid(2, 3, 4)
    assert grid.shape == (2, 3, 4, 2)
    # x increases along the last spatial axis (columns)
    assert jnp.allclose(grid[0, 0, :, 0], jnp.array([0.0, 1.0, 2.0, 3.0]))
    # y increases along the first spatial axis (rows)
    assert jnp.allclose(grid[0, :, 0, 1], jnp.array([0.0, 1.0, 2.0]))


def test_bilinear_warp_zero_flow_is_identity():
    """Warping by a zero flow returns the input unchanged."""
    key = jax.random.PRNGKey(0)
    feat = jax.random.normal(key, (1, 8, 8, 3))
    flow = jnp.zeros((1, 8, 8, 2))
    warped = bilinear_warp(feat, flow)
    assert jnp.allclose(warped, feat, atol=1e-4)


def test_bilinear_warp_integer_shift():
    """An integer (x, y) flow shifts the sampled content accordingly."""
    # feat[b, y, x] = x so we can track horizontal shifts exactly.
    x = jnp.arange(8.0)
    feat = jnp.broadcast_to(x[None, None, :, None], (1, 8, 8, 1))
    # Flow of +1 in x means we sample feat at x+1 -> content shifts left by 1.
    flow = jnp.zeros((1, 8, 8, 2)).at[..., 0].set(1.0)
    warped = bilinear_warp(feat, flow)
    # Interior columns: warped[...,x] == feat at x+1 == x+1.
    assert jnp.allclose(warped[0, 4, :7, 0], x[1:8], atol=1e-4)


def test_symmetric_warp_aligns_consistent_pair_at_true_flow():
    """Warping by the true displacement aligns both frames at the midpoint.

    Uses the repo flow convention ``feat2[x] = feat1[x - d]``. At ``flow == d``
    the two warped maps must coincide (the residual correlation peak moves to
    the zero-shift center channel); at ``flow == 0`` the peak is off-center.
    This pins the warp DIRECTION, which a tautological "halves are opposite"
    assertion cannot do.
    """
    R, d = 2, 2
    feat1 = jax.random.normal(jax.random.PRNGKey(0), (1, 16, 16, 4))
    feat2 = jnp.roll(feat1, shift=d, axis=2)  # feat2[x] = feat1[x - d]
    flow_true = jnp.zeros((1, 16, 16, 2)).at[..., 0].set(float(d))

    w1, w2 = symmetric_warp(feat1, feat2, flow_true, stride=1)
    # interior slice avoids the roll wrap-around near the borders
    interior = (slice(None), slice(4, -4), slice(4, -4), slice(None))
    assert jnp.allclose(w1[interior], w2[interior], atol=1e-4)

    center = (0 + R) * (2 * R + 1) + (0 + R)  # channel for (dy=0, dx=0)
    # interior-mean correlation is the robust autocorrelation peak
    corr_true = jnp.mean(local_correlation(w1, w2, R)[interior], axis=(0, 1, 2))
    assert int(jnp.argmax(corr_true)) == center

    # A zero estimate leaves the residual shift present -> peak off-center.
    z1, z2 = symmetric_warp(feat1, feat2, jnp.zeros_like(flow_true), stride=1)
    corr_zero = jnp.mean(local_correlation(z1, z2, R)[interior], axis=(0, 1, 2))
    assert int(jnp.argmax(corr_zero)) != center


def test_symmetric_warp_halves_are_opposite_and_scaled_by_stride():
    """The two warps use opposite half-displacements scaled by the stride."""
    f1 = jax.random.normal(jax.random.PRNGKey(1), (1, 8, 8, 2))
    f2 = jax.random.normal(jax.random.PRNGKey(2), (1, 8, 8, 2))
    flow = jnp.ones((1, 8, 8, 2)) * 2.0  # input-pixel units
    stride = 2
    w1, w2 = symmetric_warp(f1, f2, flow, stride)
    half = 0.5 * flow / stride  # level-pixel half displacement
    assert jnp.allclose(w1, bilinear_warp(f1, -half), atol=1e-5)
    assert jnp.allclose(w2, bilinear_warp(f2, half), atol=1e-5)


@pytest.mark.parametrize("search_range", [1, 2, 4])
def test_local_correlation_shape(search_range):
    """Local correlation produces (2R+1)^2 channels regardless of C."""
    key = jax.random.PRNGKey(3)
    f1 = jax.random.normal(key, (2, 6, 7, 5))
    f2 = jax.random.normal(jax.random.PRNGKey(4), (2, 6, 7, 5))
    corr = local_correlation(f1, f2, search_range)
    assert corr.shape == (2, 6, 7, (2 * search_range + 1) ** 2)


def test_local_correlation_matches_bruteforce():
    """Local correlation equals the explicit shifted inner-product mean."""
    key = jax.random.PRNGKey(5)
    B, H, W, C, R = 1, 5, 5, 4, 2
    f1 = jax.random.normal(key, (B, H, W, C))
    f2 = jax.random.normal(jax.random.PRNGKey(6), (B, H, W, C))
    corr = np.asarray(local_correlation(f1, f2, R, padding_mode="zeros"))
    f1n, f2n = np.asarray(f1), np.asarray(f2)
    k = 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            for y in range(H):
                for x in range(W):
                    ys, xs = y + dy, x + dx
                    if 0 <= ys < H and 0 <= xs < W:
                        ref = np.mean(f1n[0, y, x] * f2n[0, ys, xs])
                    else:
                        ref = 0.0  # zero padding
                    assert np.allclose(corr[0, y, x, k], ref, atol=1e-4)
            k += 1


def test_local_correlation_detects_known_shift():
    """The correlation peak channel encodes the true (dx, dy) shift."""
    key = jax.random.PRNGKey(7)
    R = 2
    f1 = jax.random.normal(key, (1, 9, 9, 8))
    # f2 is f1 shifted right by one column: f2[..., x] = f1[..., x - 1].
    f2 = jnp.roll(f1, shift=1, axis=2)
    corr = local_correlation(f1, f2, R)
    # corr[y, x, k] = f1[y, x] . f2[y + dy, x + dx]; match at dx=+1, dy=0.
    # Channel ordering is row-major over (dy, dx) in [-R, R].
    expected = (0 + R) * (2 * R + 1) + (1 + R)  # dy=0, dx=+1
    interior = corr[0, 4, 4, :]  # interior pixel, unaffected by padding
    assert int(jnp.argmax(interior)) == expected


@pytest.mark.parametrize("mode", ["zeros", "replicate", "reflect"])
def test_pad_for_conv_preserves_size_after_valid_conv(mode):
    """pad_for_conv pads exactly enough for a dilated VALID conv."""
    x = jnp.ones((1, 10, 12, 3))
    for dilation in (1, 2, 4):
        padded = pad_for_conv(x, dilation, kernel=3, mode=mode)
        p = dilation  # floor(d*(k-1)/2) for k=3
        assert padded.shape == (1, 10 + 2 * p, 12 + 2 * p, 3)


def test_pad_for_conv_replicate_vs_zeros_values():
    """Replicate padding copies the edge; zero padding inserts zeros."""
    x = jnp.arange(1.0, 5.0).reshape(1, 1, 4, 1)
    zeros = pad_for_conv(x, 1, kernel=3, mode="zeros")
    repl = pad_for_conv(x, 1, kernel=3, mode="replicate")
    assert zeros[0, 1, 0, 0] == 0.0
    assert repl[0, 1, 0, 0] == 1.0  # left edge replicated
    assert repl[0, 1, -1, 0] == 4.0  # right edge replicated
