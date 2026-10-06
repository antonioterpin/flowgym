"""Estimator-level tests for the consolidated JAX OpenPIV estimator.

These cover the configuration surface that the cache pipeline relies on:
``get_config`` (pure parameters, like every other estimator), the
``get_cache_id_suffix`` recipe that namespaces per-config caches, and the
wiring of the sweep knobs (subpixel/correlation method, multipass,
sig2noise rejection) into the actual flow computation.
"""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthpix import SynthpixBatch

from flowgym.flow.open_piv.openpiv_jax import OpenPIVJAXEstimator

# Canonical config used by the on-disk sweep cache library. Its pure-param
# ``get_config`` md5 is pinned below; the cache re-key migration targets it.
_CANONICAL_KW = dict(
    window_size=16,
    search_size=16,
    overlap=8,
    subpixel_method="gaussian",
    correlation_method="circular",
    n_passes=3,
    openpiv_outlier_replacement_kernel_size=3,
    openpiv_outlier_replacement_n_iter=10,
)
_CANONICAL_SUFFIX = "_cf8badbaa"


def _run(estimator, img1, img2):
    """Run one estimation step and return the flow field (B, H, W, 2)."""
    b, h, w = img1.shape
    init = jnp.zeros((b, h, w, 2), dtype=jnp.float32)
    state = estimator.create_state(
        img1, init, image_history_size=2, rng=jax.random.PRNGKey(0)
    )
    state, _ = estimator(img2, state, None)
    return state["estimates"][:, -1]


@pytest.fixture
def image_pair():
    rng = np.random.RandomState(0)
    img1 = jnp.asarray(rng.rand(1, 64, 64).astype(np.float32))
    img2 = jnp.asarray(rng.rand(1, 64, 64).astype(np.float32))
    return img1, img2


def test_get_config_is_pure_params():
    """get_config exposes only parameters (no estimator-name key)."""
    est = OpenPIVJAXEstimator(**_CANONICAL_KW)
    config = est.get_config()
    assert config == {
        "window_size": 16,
        "search_size": 16,
        "overlap": 8,
        "subpixel_method": "gaussian",
        "correlation_method": "circular",
        "n_passes": 3,
        "sig2noise_threshold": None,
        "outlier_kernel_size": 3,
        "outlier_n_iter": 10,
    }
    assert "estimator" not in config


def test_cache_id_suffix_recipe_and_golden():
    """Suffix is md5 of the sorted-json config, pinned to a golden value."""
    est = OpenPIVJAXEstimator(**_CANONICAL_KW)
    suffix = est.get_cache_id_suffix(None)
    expected = (
        "_c"
        + hashlib.md5(
            json.dumps(est.get_config(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
    )
    assert suffix == expected
    # Golden: changing the recipe means existing caches must be re-keyed.
    assert suffix == _CANONICAL_SUFFIX


@pytest.mark.parametrize(
    "override",
    [
        {"subpixel_method": "parabolic"},
        {"correlation_method": "linear"},
        {"n_passes": 1},
        {"sig2noise_threshold": 1.3},
        {"window_size": 32, "overlap": 16},
    ],
)
def test_cache_id_suffix_distinguishes_axes(override):
    """Each swept axis yields a distinct suffix (the NaN-bug regression)."""
    base = OpenPIVJAXEstimator(**_CANONICAL_KW).get_cache_id_suffix(None)
    other = OpenPIVJAXEstimator(
        **{**_CANONICAL_KW, **override}
    ).get_cache_id_suffix(None)
    assert base != other


def test_multipass_changes_output(image_pair):
    """n_passes is wired into _estimate (output differs from single pass)."""
    img1, img2 = image_pair
    single = _run(
        OpenPIVJAXEstimator(**{**_CANONICAL_KW, "n_passes": 1}), img1, img2
    )
    multi = _run(
        OpenPIVJAXEstimator(**{**_CANONICAL_KW, "n_passes": 3}), img1, img2
    )
    assert single.shape == multi.shape
    assert not np.allclose(np.asarray(single), np.asarray(multi))


def test_sig2noise_changes_output(image_pair):
    """sig2noise_threshold is wired into _estimate (output differs)."""
    img1, img2 = image_pair
    off = _run(
        OpenPIVJAXEstimator(**{**_CANONICAL_KW, "n_passes": 1}), img1, img2
    )
    on = _run(
        OpenPIVJAXEstimator(
            **{**_CANONICAL_KW, "n_passes": 1, "sig2noise_threshold": 1.3}
        ),
        img1,
        img2,
    )
    assert not np.allclose(np.asarray(off), np.asarray(on))


def test_backward_compatible_minimal_construction(image_pair):
    """Old-style construction (no sweep knobs) still builds and runs."""
    img1, img2 = image_pair
    est = OpenPIVJAXEstimator(
        window_size=16,
        search_size=21,
        overlap=8,
        openpiv_outlier_replacement_kernel_size=3,
        openpiv_outlier_replacement_n_iter=10,
    )
    flow = _run(est, img1, img2)
    assert flow.shape == (1, 64, 64, 2)
    # Unspecified knobs fall back to the documented defaults.
    cfg = est.get_config()
    assert cfg["subpixel_method"] == "gaussian"
    assert cfg["correlation_method"] == "circular"
    assert cfg["n_passes"] == 1
    assert cfg["sig2noise_threshold"] is None


@pytest.mark.parametrize(
    "override",
    [
        {"window_size": 0},
        {"search_size": -1},
        {"overlap": -1},
        {"overlap": 16},
        {"window_size": [16, 24], "overlap": [8, 24]},
        {"window_size": [16, 0]},
        {"window_size": 16.9},
        {"search_size": [21, 29.5]},
        {"window_size": [16, 24, 32]},
        {"overlap": [8]},
    ],
)
def test_invalid_geometry_raises(override):
    """Non-positive, non-integer or malformed sizes are rejected."""
    with pytest.raises(ValueError):
        OpenPIVJAXEstimator(**{**_CANONICAL_KW, **override})


@pytest.mark.parametrize("override", [{"n_passes": 1}, {"n_passes": 2}])
def test_subpixel_method_is_forwarded(image_pair, override):
    """subpixel_method reaches every pass, single and multipass."""
    img1, img2 = image_pair
    kw = {
        **_CANONICAL_KW,
        "window_size": 32,
        "search_size": 32,
        "overlap": 16,
        **override,
    }
    gauss = _run(OpenPIVJAXEstimator(**kw), img1, img2)
    parab = _run(
        OpenPIVJAXEstimator(**{**kw, "subpixel_method": "parabolic"}),
        img1,
        img2,
    )
    assert not np.allclose(np.asarray(gauss), np.asarray(parab))


def test_sig2noise_has_no_effect_with_multipass(image_pair):
    """Documented: the S/N threshold only applies when n_passes == 1."""
    img1, img2 = image_pair
    kw = {**_CANONICAL_KW, "n_passes": 2}
    off = _run(OpenPIVJAXEstimator(**kw), img1, img2)
    on = _run(
        OpenPIVJAXEstimator(**{**kw, "sig2noise_threshold": 1.3}), img1, img2
    )
    np.testing.assert_array_equal(np.asarray(off), np.asarray(on))


def test_enrich_returns_per_sample_epe(image_pair):
    """enrich returns EPE of the estimate for the missing samples only."""
    img1, img2 = image_pair
    imgs1 = jnp.concatenate([img1, img2])
    imgs2 = jnp.concatenate([img2, img1])
    gt = jnp.ones((2, 64, 64, 2), dtype=jnp.float32)
    est = OpenPIVJAXEstimator(**_CANONICAL_KW)
    batch = SynthpixBatch(
        images1=imgs1,
        images2=imgs2,
        flow_fields=gt,
        keys=jnp.arange(2, dtype=jnp.uint64),
    )
    payload = est.enrich(batch, np.array([1]))
    flow = _run(est, imgs1[1:], imgs2[1:])
    expected = jnp.mean(jnp.linalg.norm(flow - gt[1:], axis=-1), axis=(1, 2))
    np.testing.assert_allclose(payload["epe"], expected, rtol=1e-5)
    assert payload["relative_epe"].shape == (1,)

    no_gt = SynthpixBatch(
        images1=imgs1,
        images2=imgs2,
        flow_fields=None,
        keys=jnp.arange(2, dtype=jnp.uint64),
    )
    assert est.enrich(no_gt, np.array([0])) is None


def test_integer_like_sizes_are_accepted():
    """Integer-valued floats normalize to ints, so the cache id is unchanged."""
    est = OpenPIVJAXEstimator(
        **{**_CANONICAL_KW, "window_size": 16.0, "search_size": [16.0, 16]}
    )
    assert est.window_size == 16
    assert est.search_size == (16, 16)
