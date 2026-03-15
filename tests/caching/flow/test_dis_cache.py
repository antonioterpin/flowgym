"""Tests for DIS estimator caching functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthpix import SynthpixBatch

from flowgym.flow.dis.dis_jax import DISJAXFlowFieldEstimator, PresetType
from flowgym.training.caching import CacheManager, enrich_batch
from flowgym.types import CachePayload


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    d = tmp_path / "cache_test"
    d.mkdir()
    return d


def test_cache_manager_spec_storage(cache_dir):
    """Test that spec is correctly stored in the cache meta.json."""
    import json

    spec = {"epe": (np.dtype("float32"), ())}

    cm = CacheManager(
        root_dir=str(cache_dir),
        cache_id="test_spec",
        spec=spec,
    )

    # Check meta.json contents
    meta_path = cm.cache_dir / "meta.json"
    with open(meta_path) as f:
        saved_meta = json.load(f)

    assert "spec" in saved_meta
    assert saved_meta["spec"]["epe"] == ["float32", []]


def test_dis_get_config():
    """Test get_config method of DIS estimator."""
    model = DISJAXFlowFieldEstimator(preset=PresetType.FAST)
    config = model.get_config()

    assert config["preset"] == "FAST"
    assert config["patch_size"] == 9
    # Check a few others
    assert "start_level" in config


def test_dis_enrich(tmp_path):
    """Test enrich returns correct EPE payload."""
    # Setup model
    model = DISJAXFlowFieldEstimator(preset=PresetType.ULTRAFAST)

    # Create fake batch
    B, H, W = 2, 32, 32
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    images1 = jax.random.uniform(k1, (B, H, W))
    images2 = jax.random.uniform(k2, (B, H, W))

    # Create a flow field that we can predict (or just random to check
    # output shape/type)
    # For DIS to work reasonably, we can just check it runs and produces output.
    gt_flow = jax.random.normal(k3, (B, H, W, 2))

    batch = SynthpixBatch(
        images1=images1,
        images2=images2,
        flow_fields=gt_flow,
        keys=jnp.array([1, 2], dtype=jnp.uint32),
    )

    # We want to simulate a cache miss for index 0 and 1
    miss_idxs = jnp.array([0, 1])

    # Run enrich
    payload = model.enrich(batch, miss_idxs)

    assert payload is not None
    assert payload["epe"] is not None
    assert payload["epe"].shape == (B,)
    assert not jnp.isnan(payload["epe"]).any()

    # Check partial miss
    miss_idxs_partial = jnp.array([0])
    payload_partial = model.enrich(batch, miss_idxs_partial)
    assert payload_partial is not None
    assert payload_partial["epe"].shape == (1,)


def test_integration_enrich(cache_dir):
    """Test full enrich flow with CacheManager and DIS model."""
    spec = {"epe": (np.dtype("float32"), ())}
    model = DISJAXFlowFieldEstimator(preset=PresetType.ULTRAFAST)

    cm = CacheManager(
        root_dir=str(cache_dir),
        cache_id="test_enrich",
        spec=spec,
    )

    B, H, W = 2, 32, 32
    key = jax.random.PRNGKey(99)
    images1 = jax.random.uniform(key, (B, H, W))
    images2 = jax.random.uniform(key, (B, H, W))
    gt_flow = jnp.zeros((B, H, W, 2))  # Flat flow to keep things simple

    # Use JAX keys with uint32 type
    batch = SynthpixBatch(
        images1=images1,
        images2=images2,
        flow_fields=gt_flow,
        keys=jnp.array([100, 101], dtype=jnp.uint32),
    )

    # 1. Enrich (should be miss -> compute -> write)
    payload = enrich_batch(batch, model, cache_manager=cm)
    assert payload is not None
    assert payload.epe is not None
    assert payload.epe.shape == (B,)

    # 2. Flush to disk
    cm.flush()

    # 3. New CacheManager (warm start to load cached data)
    cm2 = CacheManager(
        root_dir=str(cache_dir),
        cache_id="test_enrich",
        spec=spec,
        warm_start="all",  # Load data into memory for lookup
    )

    # 4. Lookup (should be hit)
    payload2, hits = cm2.lookup(np.array([100, 101], dtype=np.uint64))
    assert hits.all()
    assert np.allclose(payload.epe, payload2["epe"])


def test_dis_uses_cached_metrics_without_running_flow(monkeypatch):
    """DIS should short-circuit when cache provides precomputed metrics."""
    model = DISJAXFlowFieldEstimator(preset=PresetType.ULTRAFAST)

    B, H, W = 2, 32, 32
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key, 2)
    images1 = jax.random.uniform(k1, (B, H, W))
    images2 = jax.random.uniform(k2, (B, H, W))

    init_flow = jnp.zeros((B, H, W, 2), dtype=jnp.float32)
    state = model.create_state(
        images1, init_flow, image_history_size=2, rng=key
    )
    trainable_state = model.create_trainable_state(images1, key)

    cache_payload = CachePayload(
        has_precomputed_errors=True,
        epe=np.array([0.2, 0.3], dtype=np.float32),
        relative_epe=np.array([1.2, 1.3], dtype=np.float32),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "estimate_dis_flow should not be called on cache hit"
        )

    monkeypatch.setattr(
        "flowgym.flow.dis.process.estimate_dis_flow",
        fail_if_called,
    )

    new_state, metrics = model(
        images2,
        state,
        trainable_state,
        cache_payload=cache_payload,
    )

    np.testing.assert_allclose(np.asarray(metrics["errors"]), [0.2, 0.3])
    np.testing.assert_allclose(
        np.asarray(metrics["relative_errors"]), [1.2, 1.3]
    )
    assert new_state["estimates"].shape == (B, 2, H, W, 2)
