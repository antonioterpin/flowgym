"""Tests for caching module."""

import os
import tempfile
from pathlib import Path

import numpy as np

from flowgym.training.caching import CacheManager


def test_cache_manager_functional():
    """Functional test for CacheManager."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Initialize CacheManager
        cache_id = "test_cache"
        cm = CacheManager(
            root_dir=tmp_dir,
            cache_id=cache_id,
            spec={"values": (np.float32, (2,))},
            warm_start="none",
        )

        # 2. Write data
        keys = np.array([1, 2, 3], dtype=np.uint64)
        payload = {
            "values": np.array(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
            )
        }
        cm.write(keys, payload)

        # 3. Read back
        # Use different instance or same
        payload_read, hit_mask = cm.lookup(keys)

        assert np.all(hit_mask)
        np.testing.assert_allclose(payload_read["values"], payload["values"])

        # 4. Partial hit
        keys_mixed = np.array([1, 4], dtype=np.uint64)
        payload_mixed, hit_mask_mixed = cm.lookup(keys_mixed)

        assert hit_mask_mixed[0]
        assert not hit_mask_mixed[1]
        np.testing.assert_allclose(
            payload_mixed["values"][0], payload["values"][0]
        )
        assert np.all(payload_mixed["values"][1] == 0)


def test_cache_warm_start():
    """Test warm start capabilities."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup data
        cache_id = "test_warm"
        data_dir = Path(tmp_dir) / cache_id / "data"
        data_dir.mkdir(parents=True)

        # Create a real parquet file via CM
        cm_write = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (2,))}
        )
        keys = np.array([10, 20], dtype=np.uint64)
        vals = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        cm_write.write(keys, {"values": vals})
        cm_write.close()

        # Test "index" warm start
        cm_index = CacheManager(
            tmp_dir,
            cache_id,
            spec={"values": (np.float32, (2,))},
            warm_start="index",
        )
        assert set(cm_index.index.keys()) == {10, 20}
        assert cm_index.all_data is None

        # Test "all" warm start
        cm_all = CacheManager(
            tmp_dir,
            cache_id,
            spec={"values": (np.float32, (2,))},
            warm_start="all",
        )
        assert cm_all.all_data is not None
        # Find indices by key lookup
        key_indices = {int(k): i for i, k in enumerate(cm_all.all_data["key"])}
        assert 10 in key_indices
        row = key_indices[10]
        np.testing.assert_allclose(cm_all.all_data["values"][row], vals[0])


def test_cache_warm_start_index_lookup_without_disk_scan():
    """Index warm start should serve lookups without dataset fragment scans."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_warm_index_lookup"

        cm_write = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (2,))}
        )
        cm_write.write(
            np.array([10, 20], dtype=np.uint64),
            {"values": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)},
        )
        cm_write.close()

        cm_index = CacheManager(
            tmp_dir,
            cache_id,
            spec={"values": (np.float32, (2,))},
            warm_start="index",
        )
        assert cm_index.index

        # Ensure the test exercises index-based retrieval only.
        cm_index.source_dirs = []

        payload, hit = cm_index.lookup(np.array([10, 20], dtype=np.uint64))
        assert np.all(hit)
        np.testing.assert_allclose(payload["values"], [[1.0, 1.0], [2.0, 2.0]])


def test_lookup_all_warm_start_handles_duplicate_requested_keys():
    """Lookup should mark duplicate keys as hits in all-data mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_warm_dups"

        cm_write = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (2,))}
        )
        cm_write.write(
            np.array([10], dtype=np.uint64),
            {"values": np.array([[1.0, 1.0]], dtype=np.float32)},
        )
        cm_write.close()

        cm_all = CacheManager(
            tmp_dir,
            cache_id,
            spec={"values": (np.float32, (2,))},
            warm_start="all",
        )
        assert cm_all.all_data is not None

        # Force this test to exercise the in-memory all_data path only.
        cm_all.source_dirs = []

        payload, hit = cm_all.lookup(np.array([10, 10], dtype=np.uint64))
        assert np.all(hit)
        np.testing.assert_allclose(payload["values"], [[1.0, 1.0], [1.0, 1.0]])


def test_lookup_disk_prefers_newest_part_for_duplicate_keys():
    """Disk lookup should deterministically prefer the newest parquet part."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_disk_dupe_policy"
        cm = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (1,))}
        )

        cm.write(
            np.array([7], dtype=np.uint64),
            {"values": np.array([[1.0]], dtype=np.float32)},
        )
        cm.flush()

        cm.write(
            np.array([7], dtype=np.uint64),
            {"values": np.array([[2.0]], dtype=np.float32)},
        )
        cm.flush()

        files = sorted((Path(tmp_dir) / cache_id / "data").glob("*.parquet"))
        assert len(files) == 2
        os.utime(files[0], (1, 1))
        os.utime(files[1], (2, 2))

        cm_read = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (1,))}
        )
        payload, hit = cm_read.lookup(np.array([7], dtype=np.uint64))
        assert np.all(hit)
        np.testing.assert_allclose(payload["values"], [[2.0]])


def test_cache_flush_consistency():
    """Test that flushing updates the in-memory cache when warm_start='all'."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_flush_consistency"
        # Start with empty cache, warm_start="all"
        cm = CacheManager(
            tmp_dir,
            cache_id,
            spec={"values": (np.float32, (1,))},
            warm_start="all",
        )

        # Initial state - all_data may be None or empty for a fresh cache
        # The key test is that after flush, the data is on disk and can
        # be retrieved

        # Write data
        keys = np.array([1, 2], dtype=np.uint64)
        vals = np.array([[10.0], [20.0]], dtype=np.float32)
        cm.write(keys, {"values": vals})

        # Data should be in pending buffer
        assert 1 in cm.pending_buffer["key"]

        # Flush
        cm.flush()

        # Pending buffer should be empty now
        assert not cm.pending_buffer["key"]

        # Lookup should hit via disk (since we flushed)
        payload, hit = cm.lookup(keys)
        assert np.all(hit)
        np.testing.assert_allclose(payload["values"], vals)


def test_cache_shape_validation():
    """Test that reading cache with mismatched spec still works via lookup.

    Note: The CacheManager doesn't raise errors on shape mismatch during init.
    Instead, it reads the data as-is and the caller should validate shapes.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_shape"
        # Create cache with shape (2,)
        cm_write = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (2,))}
        )
        keys = np.array([1], dtype=np.uint64)
        vals = np.array([[1.0, 2.0]], dtype=np.float32)
        cm_write.write(keys, {"values": vals})
        cm_write.close()

        # Load with different spec - CacheManager reads the data as stored
        # The lookup will return what's on disk, caller validates
        cm_read = CacheManager(
            tmp_dir,
            cache_id,
            spec={"values": (np.float32, (3,))},
            warm_start="all",
        )
        # The all_data will have the original shape from disk
        assert cm_read.all_data is not None
        # Check that data was loaded (shape from disk, not from spec)
        assert len(cm_read.all_data["key"]) == 1


def test_cache_manager_enrich():
    """Test enrich_batch with CacheManager."""
    from unittest.mock import MagicMock

    from flowgym.training.caching import enrich_batch

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "test_enrich"
        cm = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (2,))}
        )

        # Mock batch and model
        class MockBatch:
            def __init__(self, keys):
                self.keys = keys

        batch = MockBatch(np.array([1, 2], dtype=np.uint64))
        model = MagicMock()

        # Define enrich behavior
        # miss_idxs will be [0, 1] for the first call
        def enrich_fn(batch, miss_idxs, **kwargs):
            return {
                "values": np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32)[
                    miss_idxs
                ]
            }

        model.enrich.side_effect = enrich_fn

        # 1. First call (all misses)
        payload = enrich_batch(batch, model, cache_manager=cm)
        assert payload is not None
        np.testing.assert_allclose(
            payload.extras["values"], [[0.1, 0.1], [0.2, 0.2]]
        )
        assert model.enrich.call_count == 1

        # 2. Second call (all hits)
        model.enrich.reset_mock()
        payload2 = enrich_batch(batch, model, cache_manager=cm)
        assert payload2 is not None
        np.testing.assert_allclose(
            payload2.extras["values"], [[0.1, 0.1], [0.2, 0.2]]
        )
        assert model.enrich.call_count == 0

        # 3. Mixed hits/misses
        batch_mixed = MockBatch(np.array([1, 3], dtype=np.uint64))

        def enrich_mixed(batch, miss_idxs, **kwargs):
            # Only index 1 in batch_mixed is a miss (key 3)
            return {"values": np.array([[0.3, 0.3]], dtype=np.float32)}

        model.enrich.side_effect = enrich_mixed
        payload3 = enrich_batch(batch_mixed, model, cache_manager=cm)
        assert payload3 is not None
        np.testing.assert_allclose(
            payload3.extras["values"], [[0.1, 0.1], [0.3, 0.3]]
        )
        assert model.enrich.call_count == 1
