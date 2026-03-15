"""Tests for buffered writes in CacheManager."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from flowgym.training.caching import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


def test_buffered_writes_no_flush_small(temp_cache_dir):
    """Test that small writes are buffered and not flushed immediately.

    The CacheManager buffers writes until 1000 items accumulate.
    Small writes should stay in the pending_buffer.
    """
    cache_id = "test_buffer_small"
    cm = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="none",
    )

    keys = np.array([1, 2, 3], dtype=np.uint64)
    payload = {"values": np.array([[0.1], [0.2], [0.3]], dtype=np.float32)}

    cm.write(keys, payload)

    # Assert no files yet (auto-flush threshold is 1000)
    data_dir = cm.cache_dir / "data"
    files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []
    assert len(files) == 0

    # Assert data is in pending buffer
    assert len(cm.pending_buffer["key"]) == 3
    assert len(cm.pending_buffer["values"]) == 3

    cm.close()


def test_buffered_writes_flush_trigger_count(temp_cache_dir):
    """Test that exceeding 1000 items triggers automatic flush."""
    cache_id = "test_buffer_flush_count"
    cm = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="none",
    )

    # Write 999 items (just under threshold)
    keys1 = np.arange(999, dtype=np.uint64)
    payload1 = {"values": np.arange(999, dtype=np.float32).reshape(-1, 1)}
    cm.write(keys1, payload1)

    data_dir = cm.cache_dir / "data"
    files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []
    assert len(files) == 0, "Should not flush at 999 items"

    # Write 2 more to exceed threshold
    keys2 = np.array([1000, 1001], dtype=np.uint64)
    payload2 = {"values": np.array([[1000.0], [1001.0]], dtype=np.float32)}
    cm.write(keys2, payload2)

    files = list(data_dir.glob("*.parquet"))
    assert len(files) == 1, "Should flush after exceeding 1000 items"

    # Buffer should be empty after flush
    assert len(cm.pending_buffer["key"]) == 0

    # Verify file content
    table = pq.read_table(files[0])
    assert table.num_rows == 1001

    cm.close()


def test_lookup_sees_pending_buffer(temp_cache_dir):
    """Test that lookup finds data in the pending buffer before disk."""
    cache_id = "test_lookup_pending"
    cm = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="none",
    )

    keys = np.array([1, 2], dtype=np.uint64)
    payload = {"values": np.array([[10.0], [20.0]], dtype=np.float32)}
    cm.write(keys, payload)

    # Nothing on disk
    data_dir = cm.cache_dir / "data"
    files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []
    assert len(files) == 0

    # Lookup should find key 1 in buffer, miss key 3
    keys_lookup = np.array([1, 3], dtype=np.uint64)
    result_payload, hit_mask = cm.lookup(keys_lookup)

    assert hit_mask[0] is True or hit_mask[0]
    assert hit_mask[1] is False or not hit_mask[1]
    np.testing.assert_allclose(result_payload["values"][0], [10.0])

    cm.close()


def test_context_manager_flushes(temp_cache_dir):
    """Test that context manager flushes on exit."""
    cache_id = "test_ctx_mgr"

    with CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="none",
    ) as cm:
        keys = np.array([1, 2], dtype=np.uint64)
        payload = {"values": np.array([[10.0], [20.0]], dtype=np.float32)}
        cm.write(keys, payload)

        # Verify not flushed yet
        data_dir = cm.cache_dir / "data"
        files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []
        assert len(files) == 0

    # After exit, should be flushed
    files = list((Path(temp_cache_dir) / cache_id / "data").glob("*.parquet"))
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert table.num_rows == 2


def test_close_idempotent(temp_cache_dir):
    """Test that close is safe to call multiple times or on empty buffer."""
    cache_id = "test_close_idem"
    cm = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="none",
    )
    cm.close()  # Empty buffer - should be safe

    # Write and close
    keys = np.array([1], dtype=np.uint64)
    cm.write(keys, {"values": np.array([[1.0]], dtype=np.float32)})
    cm.close()  # Flushes

    # Should be flushed
    data_dir = cm.cache_dir / "data"
    files = list(data_dir.glob("*.parquet"))
    assert len(files) == 1
    assert len(cm.pending_buffer["key"]) == 0

    cm.close()  # Should do nothing (idempotent)
    files = list(data_dir.glob("*.parquet"))
    assert len(files) == 1


def test_warm_start_updates_from_flush(temp_cache_dir):
    """Test that flushed data can be read via warm start in a new instance.

    Note: The CacheManager's in-memory index is only populated at init time
    during warm start, not dynamically after flush. To see flushed data in the
    index, you need to create a new CacheManager instance.
    """
    cache_id = "test_warm_update"

    # 1. Write and flush data
    cm = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="none",  # No warm start for writer
    )

    keys = np.array([1], dtype=np.uint64)
    cm.write(keys, {"values": np.array([[1.0]], dtype=np.float32)})
    cm.flush()
    cm.close()

    # 2. Re-open with warm_start="index" to load keys
    cm_index = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="index",
    )

    # Index should contain the flushed key
    assert 1 in cm_index.index
    cm_index.close()

    # 3. Re-open with warm_start="all" to load full data
    cm_all = CacheManager(
        temp_cache_dir,
        cache_id,
        spec={"values": (np.float32, (1,))},
        warm_start="all",
    )

    # Data should be loaded into memory
    assert cm_all.all_data is not None
    assert len(cm_all.all_data["key"]) == 1
    np.testing.assert_allclose(cm_all.all_data["values"][0], [1.0])

    cm_all.close()
