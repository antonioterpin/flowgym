"""Tests for generic caching with mixed scalar/vector fields."""

import tempfile

import numpy as np

from flowgym.training.caching import CacheManager


def test_generic_payload_write_read():
    """Test writing/reading generic payload with scalar/vector fields."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_generic"
        # Spec: "epe": scalar, "scores": vector(3)
        spec = {
            "epe": (np.float32, ()),
            "scores": (np.float32, (3,)),
        }

        cm = CacheManager(
            root_dir=tmp_dir, cache_id=cache_id, spec=spec, warm_start="none"
        )

        # Write data
        keys = np.array([1, 2], dtype=np.uint64)
        payload = {
            "epe": np.array([0.5, 0.6], dtype=np.float32),
            "scores": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        }
        cm.write(keys, payload)

        # Read back (hit)
        payload_out, hit = cm.lookup(keys)
        assert np.all(hit)
        np.testing.assert_allclose(payload_out["epe"], payload["epe"])
        np.testing.assert_allclose(payload_out["scores"], payload["scores"])

        # Read back (partial miss)
        keys_mixed = np.array([1, 3], dtype=np.uint64)
        payload_mixed, hit_mixed = cm.lookup(keys_mixed)
        assert hit_mixed[0]
        assert not hit_mixed[1]

        # Check hit values
        np.testing.assert_allclose(payload_mixed["epe"][0], payload["epe"][0])
        np.testing.assert_allclose(
            payload_mixed["scores"][0], payload["scores"][0]
        )

        # Check miss values (should be zero/empty initialized)
        assert payload_mixed["epe"][1] == 0.0
        assert np.all(payload_mixed["scores"][1] == 0.0)


def test_warm_start_all_generic():
    """Test warm_start='all' with generic payload."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_warm_generic"
        spec = {"val": (np.float32, (1,)), "meta": (np.int32, ())}

        # Write data to disk first
        cm_write = CacheManager(tmp_dir, cache_id, spec=spec)
        keys = np.array([10], dtype=np.uint64)
        payload = {
            "val": np.array([[3.14]], dtype=np.float32),
            "meta": np.array([42], dtype=np.int32),
        }
        cm_write.write(keys, payload)
        cm_write.close()

        # Warm start
        cm_read = CacheManager(tmp_dir, cache_id, spec=spec, warm_start="all")

        assert cm_read.all_data is not None
        assert "val" in cm_read.all_data
        assert "meta" in cm_read.all_data

        np.testing.assert_allclose(
            cm_read.all_data["val"][0], payload["val"][0]
        )
        np.testing.assert_allclose(
            cm_read.all_data["meta"][0], payload["meta"][0]
        )


def test_enrich_generic():
    """Test enrich with generic payload."""
    from unittest.mock import MagicMock

    from flowgym.training.caching import enrich_batch

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_enrich_generic"
        spec = {"foo": (np.float32, ())}
        cm = CacheManager(tmp_dir, cache_id, spec=spec)

        class MockBatch:
            def __init__(self, keys):
                self.keys = keys

        batch = MockBatch(np.array([100], dtype=np.uint64))

        model = MagicMock()
        # enrich returns correct dict
        model.enrich.return_value = {"foo": np.array([99.9], dtype=np.float32)}

        payload = enrich_batch(batch, model, cache_manager=cm)
        assert payload is not None
        assert "foo" in payload.extras
        np.testing.assert_allclose(payload.extras["foo"], [99.9])

        # Check it was written to pending buffer
        assert 100 in cm.pending_buffer["key"]
        idx = cm.pending_buffer["key"].index(100)
        np.testing.assert_allclose(cm.pending_buffer["foo"][idx], 99.9)
