"""Tests for photometric augmentation."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.common.augment import (
    photometric_augment,
    photometric_augment_validate_params,
)

IDENTITY = dict(
    gain_range=(1.0, 1.0),
    offset_range=(0.0, 0.0),
    frame_gain_range=(1.0, 1.0),
)


def _pair(seed=0):
    return jax.random.uniform(jax.random.PRNGKey(seed), (2, 16, 16)) * 255.0


def test_identity_ranges_leave_images_unchanged():
    pair = _pair()
    out = photometric_augment(pair, jax.random.PRNGKey(1), **IDENTITY)
    assert jnp.allclose(out, pair, atol=1e-3)


def test_output_stays_in_range_and_depends_on_key():
    pair = _pair()
    a = photometric_augment(pair, jax.random.PRNGKey(1))
    b = photometric_augment(pair, jax.random.PRNGKey(2))
    assert a.shape == pair.shape
    assert 0.0 <= float(a.min()) and float(a.max()) <= 255.0
    assert not jnp.allclose(a, b)
    assert jnp.array_equal(a, photometric_augment(pair, jax.random.PRNGKey(1)))


def test_pixel_order_is_preserved():
    """Intensities change but particle positions and ranking do not."""
    pair = _pair()
    out = photometric_augment(pair, jax.random.PRNGKey(3))
    for f in range(2):
        x, y = pair[f].ravel(), out[f].ravel()
        order = jnp.argsort(x)
        assert bool(jnp.all(jnp.diff(y[order]) >= -1e-4))


def test_frames_get_different_gains():
    pair = jnp.full((2, 8, 8), 100.0)
    out = photometric_augment(
        pair,
        jax.random.PRNGKey(0),
        gain_range=(1.0, 1.0),
        offset_range=(0.0, 0.0),
    )
    assert not jnp.allclose(out[0], out[1])


def test_works_under_vmap_and_jit():
    pairs = jnp.stack([_pair(0), _pair(1)])
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    out = jax.jit(jax.vmap(photometric_augment))(pairs, keys)
    assert out.shape == pairs.shape


@pytest.mark.parametrize(
    "bad",
    [
        {"gain_range": (1.5, 0.5)},
        {"gain_range": (0.0, 2.0)},
        {"frame_gain_range": (0.0, 1.1)},
        {"offset_range": (5.0, -5.0)},
        {"max_value": 0.0},
    ],
)
def test_validate_params_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        photometric_augment_validate_params(**bad)
