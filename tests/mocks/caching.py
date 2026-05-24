"""Cache-directory and synthpix-batch fixtures for the caching tests."""

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Create a temporary cache directory with optional pre-populated data.

    Yields the path to the cache directory. Use this fixture for testing
    cache read/write operations without hitting the real filesystem.
    """
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir
    # Cleanup is handled by tmp_path


@pytest.fixture
def populated_cache_dir(mock_cache_dir):
    """Create a pre-populated cache with sample EPE data.

    Returns (cache_dir, cache_id, keys, epe_values) tuple for verification.
    """
    from flowgym.training.caching import CacheManager

    cache_id = "test_estimator_cache"
    spec = {
        "epe": (np.dtype("float32"), ()),
        "relative_epe": (np.dtype("float32"), ()),
    }

    cm = CacheManager(
        root_dir=str(mock_cache_dir),
        cache_id=cache_id,
        spec=spec,
        warm_start="none",
    )

    # Write sample data
    keys = np.array([1001, 1002, 1003], dtype=np.uint64)
    payload = {
        "epe": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "relative_epe": np.array([0.01, 0.02, 0.03], dtype=np.float32),
    }
    cm.write(keys, payload)
    cm.flush()
    cm.close()

    return mock_cache_dir, cache_id, keys, payload


@pytest.fixture
def mock_synthpix_batch():
    """Create a mock SynthpixBatch for cache testing."""
    from synthpix import SynthpixBatch

    B, H, W = 4, 64, 64  # Small dims for fast tests
    return SynthpixBatch(
        images1=jnp.zeros((B, H, W)),
        images2=jnp.zeros((B, H, W)),
        flow_fields=jnp.zeros((B, H, W, 2)),
        keys=jnp.array([100, 101, 102, 103], dtype=jnp.uint64),
    )
