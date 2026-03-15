"""Basic caching tests for flow field estimators."""

import numpy as np

from flowgym.training.caching import CacheManager


class TestCacheManagerBasics:
    """Test basic CacheManager functionality."""

    def test_write_and_lookup(self, mock_cache_dir):
        """Test basic write and lookup operations."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_write_lookup",
            spec=spec,
            warm_start="none",
        )

        keys = np.array([1, 2, 3], dtype=np.uint64)
        payload = {"epe": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        cm.write(keys, payload)

        # Lookup should find all keys
        payload_read, hit = cm.lookup(keys)
        assert np.all(hit)
        np.testing.assert_allclose(payload_read["epe"], payload["epe"])

    def test_partial_hit(self, mock_cache_dir):
        """Test lookup with partial cache hits."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_partial",
            spec=spec,
            warm_start="none",
        )

        # Write only keys 1, 2
        keys = np.array([1, 2], dtype=np.uint64)
        payload = {"epe": np.array([0.1, 0.2], dtype=np.float32)}
        cm.write(keys, payload)

        # Lookup keys 1, 2, 3 (3 is a miss)
        query_keys = np.array([1, 2, 3], dtype=np.uint64)
        payload_read, hit = cm.lookup(query_keys)

        assert hit[0] and hit[1]  # Keys 1, 2 are hits
        assert not hit[2]  # Key 3 is a miss
        np.testing.assert_allclose(payload_read["epe"][:2], payload["epe"])

    def test_persistence_across_instances(self, mock_cache_dir):
        """Test that cache persists after closing and reopening."""
        spec = {"epe": (np.dtype("float32"), ())}
        cache_id = "test_persistence"

        # Write data with first instance
        cm1 = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id=cache_id,
            spec=spec,
            warm_start="none",
        )
        keys = np.array([10, 20], dtype=np.uint64)
        payload = {"epe": np.array([1.0, 2.0], dtype=np.float32)}
        cm1.write(keys, payload)
        cm1.flush()
        cm1.close()

        # Read with second instance
        cm2 = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id=cache_id,
            spec=spec,
            warm_start="all",
        )
        payload_read, hit = cm2.lookup(keys)
        assert np.all(hit)
        np.testing.assert_allclose(payload_read["epe"], payload["epe"])

    def test_spec_storage(self, mock_cache_dir):
        """Test that spec is correctly stored and retrieved in meta.json."""
        import json

        spec = {"epe": (np.dtype("float32"), ())}

        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_spec_storage",
            spec=spec,
        )

        # Check meta.json contents (stored at cache_dir / meta.json)
        meta_path = cm.cache_dir / "meta.json"
        with open(meta_path) as f:
            saved_meta = json.load(f)

        assert "spec" in saved_meta
        assert saved_meta["spec"]["epe"] == ["float32", []]


class TestCacheManagerEnrich:
    """Test CacheManager.enrich integration with estimators."""

    def test_enrich_computes_misses(self, mock_cache_dir, mock_synthpix_batch):
        """Test that enrich_batch calls model.enrich for misses."""
        from unittest.mock import MagicMock

        from flowgym.training.caching import enrich_batch

        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_enrich",
            spec=spec,
        )

        # Mock model
        model = MagicMock()
        B = mock_synthpix_batch.images1.shape[0]

        def compute_miss(batch, miss_idxs, **kwargs):
            return {"epe": np.ones(len(miss_idxs), dtype=np.float32) * 0.5}

        model.enrich.side_effect = compute_miss

        # First call - all misses
        payload = enrich_batch(mock_synthpix_batch, model, cache_manager=cm)
        assert payload is not None
        assert model.enrich.call_count == 1
        np.testing.assert_allclose(payload.epe, np.ones(B) * 0.5)

        # Second call - all hits
        model.enrich.reset_mock()
        payload2 = enrich_batch(mock_synthpix_batch, model, cache_manager=cm)
        assert model.enrich.call_count == 0  # No misses
        np.testing.assert_allclose(payload2.epe, np.ones(B) * 0.5)

    def test_enrich_mixed_hits_misses(self, mock_cache_dir):
        """Test enrich_batch handles mixed hit/miss scenarios."""
        from unittest.mock import MagicMock

        import jax.numpy as jnp
        from synthpix import SynthpixBatch

        from flowgym.training.caching import enrich_batch

        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_mixed",
            spec=spec,
        )

        # Pre-populate with one key
        cm.write(
            np.array([100], dtype=np.uint64),
            {"epe": np.array([0.1], dtype=np.float32)},
        )

        # Batch with one cached key and one new key
        batch = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=jnp.array(
                [100, 200], dtype=jnp.uint64
            ),  # 100 is cached, 200 is not
        )

        model = MagicMock()

        def compute_miss(batch, miss_idxs, **kwargs):
            # Only the second sample (index 1) should be a miss
            return {"epe": np.array([0.2], dtype=np.float32)}

        model.enrich.side_effect = compute_miss

        payload = enrich_batch(batch, model, cache_manager=cm)
        assert model.enrich.call_count == 1

        # Check merged result
        np.testing.assert_allclose(payload.epe[0], 0.1)  # From cache
        np.testing.assert_allclose(payload.epe[1], 0.2)  # From compute


class TestEstimatorCacheInterface:
    """Test the estimator.enrich interface."""

    def test_base_estimator_returns_none(self):
        """Test that base FlowFieldEstimator.enrich returns None."""
        from flowgym.flow.dis.dis_jax import (
            DISJAXFlowFieldEstimator,
            PresetType,
        )

        # Use a concrete estimator to test default behavior
        estimator = DISJAXFlowFieldEstimator(preset=PresetType.ULTRAFAST)

        # With no flow_fields in batch, enrich should work
        import jax.numpy as jnp
        from synthpix import SynthpixBatch

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=None,  # No ground truth
            keys=jnp.array([1, 2], dtype=jnp.uint64),
        )

        # Without flow_fields, DIS returns None
        result = estimator.enrich(batch, np.array([0, 1]))
        assert result is None

    def test_cache_id_suffix_includes_config(self):
        """Test that cache ID suffix is based on configuration."""
        from flowgym.flow.dis.dis_jax import (
            DISJAXFlowFieldEstimator,
            PresetType,
        )

        estimator = DISJAXFlowFieldEstimator(preset=PresetType.FAST)
        suffix = estimator.get_cache_id_suffix(trainable_state=None)

        # DIS returns a config-based suffix starting with _c
        assert suffix.startswith("_c")
