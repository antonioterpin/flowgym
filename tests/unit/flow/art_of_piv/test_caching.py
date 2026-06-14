"""Tests for Art of PIV caching module.

Unit tests for cache management, aggregation, and unified index building.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flowgym.flow.art_of_piv import caching

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_cache_manager():
    """Create a mock CacheManager with basic functionality."""
    cm = MagicMock()
    cm.all_data = {
        "key": np.array([1, 2, 3], dtype=np.int64),
        "epe": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    # Pre-calculate sorted keys so code doesn't try to sort a MagicMock
    cm._sorted_keys = cm.all_data["key"]
    cm._sorted_order = np.arange(len(cm.all_data["key"]))

    cm.spec = {
        "epe": (np.dtype("float32"), ()),
        "relative_epe": (np.dtype("float32"), ()),
    }

    def mock_lookup(keys):
        keys_arr = np.asarray(keys)
        payload = {"epe": np.zeros(len(keys_arr), dtype=np.float32)}
        hit = np.isin(keys_arr, cm.all_data["key"])
        if np.any(hit):
            for i, key in enumerate(keys_arr):
                if key in cm.all_data["key"]:
                    idx = np.where(cm.all_data["key"] == key)[0][0]
                    payload["epe"][i] = cm.all_data["epe"][idx]
        return payload, hit

    cm.lookup = mock_lookup
    cm.write = MagicMock()
    return cm


@pytest.fixture
def mock_sub_model():
    """Create a mock sub-model for testing."""
    model = MagicMock()
    model.get_cache_id_suffix.return_value = "_mock_suffix"
    model.compute_cache_miss.return_value = {
        "epe": np.array([0.5, 0.6], dtype=np.float32),
        "relative_epe": np.array([0.05, 0.06], dtype=np.float32),
    }
    model.enrich.return_value = {
        "epe": np.array([0.5, 0.6], dtype=np.float32),
        "relative_epe": np.array([0.05, 0.06], dtype=np.float32),
    }
    return model


@pytest.fixture
def mock_batch():
    """Create a mock SynthpixBatch."""
    batch = MagicMock()
    batch.flow_fields = np.random.rand(4, 64, 64, 2).astype(np.float32)
    batch.keys = np.array([1, 2, 3, 4], dtype=np.int64)
    return batch


# ──────────────────────────────────────────────────────────────────────────────
# Tests: preload_caches
# ──────────────────────────────────────────────────────────────────────────────


def test_preload_caches_loads_existing_caches(tmp_path, mock_sub_model):
    """Test that preload_caches loads existing sub-model caches."""
    # Create a fake cache directory
    # Standard MagicMock name is 'MagicMock',
    # so expected folder is 'MagicMock_mock_suffix'
    # unless we force the class name. Let's force it for clarity.
    mock_sub_model.__class__.__name__ = "MockSubModel"
    cache_dir = tmp_path / "MockSubModel_mock_suffix"
    cache_dir.mkdir()

    meta_path = cache_dir / "meta.json"
    meta_path.write_text('{"spec": {"epe": ["float32", []]}}')

    sub_models = [mock_sub_model]
    sub_model_states = [None]
    estimator_configs = [{"name": "TestModel"}]
    sub_cache_managers = {}

    with patch("flowgym.training.caching.CacheManager") as MockCM:
        mock_cm = MagicMock()
        mock_cm.all_data = {"key": np.array([1, 2, 3])}
        MockCM.return_value = mock_cm

        caching.preload_caches(
            root_dir=str(tmp_path),
            warm_start="all",
            sub_models=sub_models,
            sub_model_states=sub_model_states,
            estimator_configs=estimator_configs,
            sub_cache_managers=sub_cache_managers,
        )

        # Should have created a cache manager
        assert "MockSubModel_mock_suffix" in sub_cache_managers
        MockCM.assert_called_once()


def test_preload_caches_skips_missing_directories(tmp_path, mock_sub_model):
    """Test that preload_caches gracefully skips non-existent cache dirs."""
    sub_models = [mock_sub_model]
    sub_model_states = [None]
    estimator_configs = [{"name": "TestModel"}]
    sub_cache_managers = {}

    # Don't create any cache directories
    caching.preload_caches(
        root_dir=str(tmp_path),
        warm_start="all",
        sub_models=sub_models,
        sub_model_states=sub_model_states,
        estimator_configs=estimator_configs,
        sub_cache_managers=sub_cache_managers,
    )

    # Should complete without error
    assert len(sub_cache_managers) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: build_unified_index
# ──────────────────────────────────────────────────────────────────────────────


def test_build_unified_index_returns_empty_dict_when_no_caches():
    """Test that build_unified_index returns {} when no caches are loaded."""
    result = caching.build_unified_index(
        sub_models=[],
        sub_model_states=[],
        sub_cache_managers={},
    )

    assert result == {}


def test_build_unified_index_creates_lookup_dict(mock_sub_model):
    """Test that build_unified_index creates a unified lookup dictionary."""
    mock_sub_model.get_cache_id_suffix.return_value = "_suffix1"
    mock_sub_model.__class__.__name__ = "MockSubModel"

    cm1 = MagicMock()
    cm1.all_data = {
        "key": np.array([1, 2, 3], dtype=np.int64),
        "epe": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }

    cm2 = MagicMock()
    cm2.all_data = {
        "key": np.array([1, 2, 4], dtype=np.int64),
        "epe": np.array([0.15, 0.25, 0.45], dtype=np.float32),
    }

    # Create a second mock with a custom class to match cache_id pattern
    mock_sub_model2 = MagicMock()
    mock_sub_model2.get_cache_id_suffix.return_value = "_suffix2"
    # Set __class__ properly for name lookup
    type(mock_sub_model2).__name__ = "MockModel2"

    sub_models = [mock_sub_model, mock_sub_model2]
    sub_model_states = [None, None]
    sub_cache_managers = {
        "MockSubModel_suffix1": cm1,
        "MockModel2_suffix2": cm2,
    }

    unified_index = caching.build_unified_index(
        sub_models=sub_models,
        sub_model_states=sub_model_states,
        sub_cache_managers=sub_cache_managers,
    )

    assert unified_index is not None
    assert isinstance(unified_index, dict)
    # Common keys are 1 and 2
    assert len(unified_index) == 2
    assert 1 in unified_index
    assert 2 in unified_index


def test_build_unified_index_only_includes_common_keys(mock_sub_model):
    """Test that unified index only includes keys present in ALL caches."""
    mock_sub_model.get_cache_id_suffix.return_value = "_suffix1"
    mock_sub_model.__class__.__name__ = "MockSubModel"

    cm1 = MagicMock()
    cm1.all_data = {
        "key": np.array([1, 2, 3], dtype=np.int64),
        "epe": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }

    cm2 = MagicMock()
    cm2.all_data = {
        "key": np.array([2, 3, 4], dtype=np.int64),  # Only 2 and 3 overlap
        "epe": np.array([0.25, 0.35, 0.45], dtype=np.float32),
    }

    mock_sub_model2 = MagicMock()
    mock_sub_model2.get_cache_id_suffix.return_value = "_suffix2"

    sub_models = [mock_sub_model, mock_sub_model2]
    sub_model_states = [None, None]
    sub_cache_managers = {
        "MockSubModel_suffix1": cm1,
        "MagicMock_suffix2": cm2,
    }

    unified_index = caching.build_unified_index(
        sub_models=sub_models,
        sub_model_states=sub_model_states,
        sub_cache_managers=sub_cache_managers,
    )

    assert unified_index is not None
    assert len(unified_index) == 2  # Only keys 2 and 3
    assert 2 in unified_index
    assert 3 in unified_index
    assert 1 not in unified_index  # Only in cm1
    assert 4 not in unified_index  # Only in cm2


# ──────────────────────────────────────────────────────────────────────────────
# Tests: enrich_from_sub_caches
# ──────────────────────────────────────────────────────────────────────────────


def test_enrich_from_sub_caches_uses_unified_index(mock_batch, mock_sub_model):
    """Test that enrich_from_sub_caches uses unified index for O(1) lookups."""
    mock_batch.keys = np.array([1, 2, 3], dtype=np.int64)

    # Create unified index
    unified_index = {
        1: (np.array([0.1], dtype=np.float32), np.array([True])),
        2: (np.array([0.2], dtype=np.float32), np.array([True])),
        3: (np.array([0.3], dtype=np.float32), np.array([True])),
    }

    with patch("flowgym.training.caching.compute_batch_keys") as mock_keys:
        mock_keys.return_value = np.array([1, 2, 3], dtype=np.int64)

        payload = caching.enrich_from_sub_caches(
            batch=mock_batch,
            compute_misses=True,
            sub_models=[mock_sub_model],
            sub_model_states=[None],
            sub_cache_managers={},
            unified_epe_index=unified_index,
        )

    assert "epe_all" in payload
    assert "epe_mask" in payload
    assert payload["epe_all"].shape == (3, 1)
    assert payload["epe_mask"].shape == (3, 1)
    # All keys found in unified index
    assert np.all(payload["epe_mask"])


def test_enrich_from_sub_caches_fallback_to_per_cache_lookup(
    mock_batch, mock_sub_model, mock_cache_manager
):
    """Test that enrich_from_sub_caches handles empty unified index."""
    mock_batch.keys = np.array([1, 2, 3], dtype=np.int64)
    mock_sub_model.get_cache_id_suffix.return_value = "_suffix"

    sub_cache_managers = {
        "MagicMock_suffix": mock_cache_manager
    }  # Actual MagicMock class name

    with patch("flowgym.training.caching.compute_batch_keys") as mock_keys:
        mock_keys.return_value = np.array([1, 2, 3], dtype=np.int64)

        payload = caching.enrich_from_sub_caches(
            batch=mock_batch,
            compute_misses=False,
            sub_models=[mock_sub_model],
            sub_model_states=[None],
            sub_cache_managers=sub_cache_managers,
            unified_epe_index={},
        )

    assert payload["epe_all"].shape == (3, 1)
    assert payload["epe_mask"].shape == (3, 1)
    assert not np.any(payload["epe_mask"])


def test_enrich_from_sub_caches_computes_misses_when_enabled(
    mock_batch, mock_sub_model, mock_cache_manager
):
    """Test that missing entries are computed when compute_misses=True."""
    # Key 4 is missing from unified index
    mock_batch.keys = np.array([1, 2, 3, 4], dtype=np.int64)
    mock_batch.flow_fields = np.random.rand(4, 64, 64, 2).astype(np.float32)
    mock_sub_model.get_cache_id_suffix.return_value = "_suffix"

    # Unified index missing key 4
    unified_index = {
        1: (np.array([0.1], dtype=np.float32), np.array([True])),
        2: (np.array([0.2], dtype=np.float32), np.array([True])),
        3: (np.array([0.3], dtype=np.float32), np.array([True])),
        # 4 is missing
    }

    sub_cache_managers = {"MagicMock_suffix": mock_cache_manager}

    with patch("flowgym.training.caching.compute_batch_keys") as mock_keys:
        mock_keys.return_value = np.array([1, 2, 3, 4], dtype=np.int64)

        # Fix return shape for single miss
        mock_sub_model.enrich.return_value = {
            "epe": np.array([0.5], dtype=np.float32),
            "relative_epe": np.array([0.05], dtype=np.float32),
        }

        payload = caching.enrich_from_sub_caches(
            batch=mock_batch,
            compute_misses=True,
            sub_models=[mock_sub_model],
            sub_model_states=[None],
            sub_cache_managers=sub_cache_managers,
            unified_epe_index=unified_index,
        )

    # Should have computed the miss via sub-model
    mock_sub_model.enrich.assert_called()
    assert payload["epe_mask"][3, 0]  # Key 4 should now have data


def test_enrich_from_sub_caches_skips_misses_when_disabled(
    mock_batch, mock_sub_model, mock_cache_manager
):
    """Test that missing entries are skipped when compute_misses=False."""
    mock_batch.keys = np.array([1, 2, 3, 4], dtype=np.int64)
    mock_batch.flow_fields = np.random.rand(4, 64, 64, 2).astype(np.float32)
    mock_sub_model.get_cache_id_suffix.return_value = "_suffix"

    # Unified index missing key 4
    unified_index = {
        1: (np.array([0.1], dtype=np.float32), np.array([True])),
        2: (np.array([0.2], dtype=np.float32), np.array([True])),
        3: (np.array([0.3], dtype=np.float32), np.array([True])),
        # 4 is missing
    }

    sub_cache_managers = {"MagicMock_suffix": mock_cache_manager}

    with patch("flowgym.training.caching.compute_batch_keys") as mock_keys:
        mock_keys.return_value = np.array([1, 2, 3, 4], dtype=np.int64)

        payload = caching.enrich_from_sub_caches(
            batch=mock_batch,
            compute_misses=False,  # Don't compute misses
            sub_models=[mock_sub_model],
            sub_model_states=[None],
            sub_cache_managers=sub_cache_managers,
            unified_epe_index=unified_index,
        )

    # Should NOT have computed misses
    mock_sub_model.enrich.assert_not_called()
    assert not payload["epe_mask"][3, 0]  # Key 4 should still be missing
