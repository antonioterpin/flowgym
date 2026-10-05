"""Tests for the cache_id_base parameter in Art of PIV caching.

Verifies that preload_caches and build_unified_index use a configurable
base string instead of the sub-model class name when cache_id_base is set,
and that default behaviour (class name) is preserved when it is unset.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from flowgym.flow.art_of_piv import caching

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_model(class_name: str, suffix: str) -> MagicMock:
    model = MagicMock()
    model.__class__.__name__ = class_name
    model.get_cache_id_suffix.return_value = suffix
    return model


def _write_meta(cache_dir, spec: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text(json.dumps({"spec": spec}))


# ──────────────────────────────────────────────────────────────────────────────
# preload_caches: cache_id_base overrides class name
# ──────────────────────────────────────────────────────────────────────────────


def test_preload_caches_uses_cache_id_base(tmp_path):
    """preload_caches finds dirs named <cache_id_base><suffix>."""
    model = _make_model("RealClassName", "_suf")
    base = "mybase"
    expected_id = "mybase_suf"

    _write_meta(tmp_path / expected_id, {"epe": ["float32", []]})

    sub_cache_managers: dict = {}
    with patch("flowgym.training.caching.CacheManager") as MockCM:
        mock_cm = MagicMock()
        mock_cm.all_data = {"key": np.array([1])}
        MockCM.return_value = mock_cm

        caching.preload_caches(
            root_dir=str(tmp_path),
            warm_start="all",
            sub_models=[model],
            sub_model_states=[None],
            estimator_configs=[{"name": "m"}],
            sub_cache_managers=sub_cache_managers,
            cache_id_base=base,
        )

    assert expected_id in sub_cache_managers
    # Class name dir must NOT have been opened.
    assert "RealClassName_suf" not in sub_cache_managers
    MockCM.assert_called_once_with(
        root_dir=str(tmp_path),
        cache_id=expected_id,
        spec={"epe": (np.dtype("float32"), ())},
        warm_start="all",
    )


def test_preload_caches_default_uses_class_name(tmp_path):
    """preload_caches uses class name when cache_id_base is unset."""
    model = _make_model("RealClassName", "_suf")
    expected_id = "RealClassName_suf"

    _write_meta(tmp_path / expected_id, {"epe": ["float32", []]})

    sub_cache_managers: dict = {}
    with patch("flowgym.training.caching.CacheManager") as MockCM:
        mock_cm = MagicMock()
        mock_cm.all_data = {"key": np.array([1])}
        MockCM.return_value = mock_cm

        caching.preload_caches(
            root_dir=str(tmp_path),
            warm_start="all",
            sub_models=[model],
            sub_model_states=[None],
            estimator_configs=[{"name": "m"}],
            sub_cache_managers=sub_cache_managers,
        )

    assert expected_id in sub_cache_managers


# ──────────────────────────────────────────────────────────────────────────────
# build_unified_index: cache_id_base overrides class name
# ──────────────────────────────────────────────────────────────────────────────


def _make_cm(keys, epe) -> MagicMock:
    cm = MagicMock()
    cm.all_data = {
        "key": np.array(keys, dtype=np.int64),
        "epe": np.array(epe, dtype=np.float32),
    }
    return cm


def test_build_unified_index_uses_cache_id_base():
    """build_unified_index resolves managers by <cache_id_base><suffix>."""
    model = _make_model("RealClass", "_s1")
    base = "mybase"
    cm = _make_cm([10, 20], [0.1, 0.2])

    unified = caching.build_unified_index(
        sub_models=[model],
        sub_model_states=[None],
        sub_cache_managers={"mybase_s1": cm},
        cache_id_base=base,
    )

    assert 10 in unified
    assert 20 in unified


def test_build_unified_index_default_uses_class_name():
    """build_unified_index resolves managers by class name when base unset."""
    model = _make_model("RealClass", "_s1")
    cm = _make_cm([10, 20], [0.1, 0.2])

    unified = caching.build_unified_index(
        sub_models=[model],
        sub_model_states=[None],
        sub_cache_managers={"RealClass_s1": cm},
    )

    assert 10 in unified
    assert 20 in unified


def test_build_unified_index_base_does_not_match_class_name():
    """Empty index when base overrides but managers keyed by class."""
    model = _make_model("RealClass", "_s1")
    cm = _make_cm([10, 20], [0.1, 0.2])

    unified = caching.build_unified_index(
        sub_models=[model],
        sub_model_states=[None],
        # Manager is stored under class-name key, but base overrides lookup.
        sub_cache_managers={"RealClass_s1": cm},
        cache_id_base="differentbase",
    )

    assert unified == {}
