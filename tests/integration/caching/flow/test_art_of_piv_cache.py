"""Tests for Art of PIV meta-model caching functionality."""

import jax.numpy as jnp
import numpy as np
from synthpix import SynthpixBatch

from flowgym.training.caching import CacheManager, enrich_batch
from tests.unit.flow.art_of_piv.mocks import (
    MockSubModel,
    build_testable_art_of_piv_estimator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Tests for compute_cache_miss
# ──────────────────────────────────────────────────────────────────────────────
class TestArtOfPIVComputeCacheMiss:
    """Tests for Art of PIV compute_cache_miss functionality."""

    def test_returns_epe_all_and_mask(self):
        """Test that compute_cache_miss returns epe_all and epe_mask."""
        model1 = MockSubModel("m1", epe_value=0.1)
        model2 = MockSubModel("m2", epe_value=0.2)

        estimator = build_testable_art_of_piv_estimator([model1, model2])

        batch = SynthpixBatch(
            images1=jnp.zeros((4, 64, 64)),
            images2=jnp.zeros((4, 64, 64)),
            flow_fields=jnp.zeros((4, 64, 64, 2)),
            keys=jnp.array([1, 2, 3, 4], dtype=jnp.uint64),
        )

        miss_idxs = np.array([0, 1, 2])
        payload = estimator.enrich(batch, miss_idxs)

        assert payload is not None
        assert "epe_all" in payload
        assert "epe_mask" in payload

        epe_all = payload["epe_all"]
        epe_mask = payload["epe_mask"]

        assert epe_all.shape == (3, 2)  # 3 samples, 2 models
        assert epe_mask.shape == (3, 2)

    def test_mask_reflects_sub_model_support(self):
        """Test that epe_mask reflects which sub-models support caching."""
        model1 = MockSubModel("m1", supports_epe=True)
        model2 = MockSubModel("m2", supports_epe=False)  # Does not support

        estimator = build_testable_art_of_piv_estimator([model1, model2])

        batch = SynthpixBatch(
            images1=jnp.zeros((3, 64, 64)),
            images2=jnp.zeros((3, 64, 64)),
            flow_fields=jnp.zeros((3, 64, 64, 2)),
            keys=jnp.array([1, 2, 3], dtype=jnp.uint64),
        )

        payload = estimator.enrich(batch, np.array([0, 1, 2]))

        assert payload is not None
        epe_mask = payload["epe_mask"]

        # model1 (column 0) should be all True
        assert np.all(epe_mask[:, 0])
        # model2 (column 1) should be all False
        assert np.all(~epe_mask[:, 1])

    def test_returns_none_without_flow_fields(self):
        """Test that missing flow_fields yields empty-mask payload."""
        estimator = build_testable_art_of_piv_estimator([MockSubModel("m1")])

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 64, 64)),
            images2=jnp.zeros((2, 64, 64)),
            flow_fields=None,  # No flow fields # type: ignore
            keys=jnp.array([1, 2], dtype=jnp.uint64),
        )

        payload = estimator.enrich(batch, np.array([0, 1]))
        assert payload is not None
        assert np.allclose(payload["epe_all"], 0.0)
        assert np.all(~payload["epe_mask"])

    def test_cold_start_bootstrap_computes_without_existing_sub_caches(
        self, mock_cache_dir
    ):
        """Cold-start should compute misses even when no sub-caches exist."""
        estimator = build_testable_art_of_piv_estimator(
            [MockSubModel("cold_start", epe_value=0.42)]
        )
        estimator._unified_epe_index = None

        with CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="art_of_piv_meta_bootstrap",
            spec={
                "epe_all": (np.dtype("float32"), (1,)),
                "epe_mask": (np.dtype("bool"), (1,)),
            },
            warm_start="index",
        ) as meta_cm:
            batch = SynthpixBatch(
                images1=jnp.zeros((2, 64, 64)),
                images2=jnp.zeros((2, 64, 64)),
                flow_fields=jnp.zeros((2, 64, 64, 2)),
                keys=jnp.array([10, 11], dtype=jnp.uint64),
            )
            payload = estimator.enrich(
                batch,
                np.array([0, 1]),
                trainable_state=None,
                cache_manager=meta_cm,
            )

        assert payload is not None
        np.testing.assert_allclose(payload["epe_all"][:, 0], [0.42, 0.42])
        assert np.all(payload["epe_mask"][:, 0])
        assert estimator._cache_bootstrapped
        assert isinstance(estimator._unified_epe_index, dict)

    def test_cold_start_creates_sub_cache_and_buffers_write(
        self, mock_cache_dir
    ):
        """First miss should create sub-cache directory and pending writes."""
        model = MockSubModel("create_sub_cache", epe_value=0.88)
        estimator = build_testable_art_of_piv_estimator([model])
        estimator._unified_epe_index = None

        with CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="art_of_piv_meta_create",
            spec={
                "epe_all": (np.dtype("float32"), (1,)),
                "epe_mask": (np.dtype("bool"), (1,)),
            },
            warm_start="index",
        ) as meta_cm:
            batch = SynthpixBatch(
                images1=jnp.zeros((2, 64, 64)),
                images2=jnp.zeros((2, 64, 64)),
                flow_fields=jnp.zeros((2, 64, 64, 2)),
                keys=jnp.array([100, 101], dtype=jnp.uint64),
            )
            estimator.enrich(
                batch,
                np.array([0, 1]),
                trainable_state=None,
                cache_manager=meta_cm,
            )

        expected_cache_id = "MockSubModel_mock_create_sub_cache"
        expected_cache_dir = mock_cache_dir / expected_cache_id
        assert expected_cache_dir.exists()
        assert (expected_cache_dir / "meta.json").exists()

        cm = estimator._sub_cache_managers.get(expected_cache_id)
        assert cm is not None
        assert len(cm.pending_buffer["key"]) == 2

    def test_second_call_hits_unified_index_without_reload(
        self, mock_cache_dir
    ):
        """Second call in same run should reuse in-memory unified index."""

        class CountingSubModel(MockSubModel):
            def __init__(self, name, epe_value):
                super().__init__(name=name, epe_value=epe_value)
                self.calls = 0

            def enrich(self, batch, miss_idxs, **kwargs):
                self.calls += 1
                return super().enrich(batch, miss_idxs, **kwargs)

        model = CountingSubModel("in_memory_hit", epe_value=0.31)
        estimator = build_testable_art_of_piv_estimator([model])
        estimator._unified_epe_index = None

        with CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="art_of_piv_meta_hits",
            spec={
                "epe_all": (np.dtype("float32"), (1,)),
                "epe_mask": (np.dtype("bool"), (1,)),
            },
            warm_start="index",
        ) as meta_cm:
            batch = SynthpixBatch(
                images1=jnp.zeros((2, 64, 64)),
                images2=jnp.zeros((2, 64, 64)),
                flow_fields=jnp.zeros((2, 64, 64, 2)),
                keys=jnp.array([200, 201], dtype=jnp.uint64),
            )
            estimator.enrich(
                batch,
                np.array([0, 1]),
                trainable_state=None,
                cache_manager=meta_cm,
            )
            estimator.enrich(
                batch,
                np.array([0, 1]),
                trainable_state=None,
                cache_manager=meta_cm,
            )

        assert model.calls == 1

    def test_finalize_metrics_flushes_lazy_sub_cache_writes(
        self, mock_cache_dir
    ):
        """Finalization should persist buffered writes for lazy sub-caches."""
        model = MockSubModel("finalize_flush", epe_value=0.77)
        estimator = build_testable_art_of_piv_estimator([model])
        estimator._unified_epe_index = None

        with CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="art_of_piv_meta_finalize",
            spec={
                "epe_all": (np.dtype("float32"), (1,)),
                "epe_mask": (np.dtype("bool"), (1,)),
            },
            warm_start="index",
        ) as meta_cm:
            batch = SynthpixBatch(
                images1=jnp.zeros((2, 64, 64)),
                images2=jnp.zeros((2, 64, 64)),
                flow_fields=jnp.zeros((2, 64, 64, 2)),
                keys=jnp.array([900, 901], dtype=jnp.uint64),
            )
            estimator.enrich(
                batch,
                np.array([0, 1]),
                trainable_state=None,
                cache_manager=meta_cm,
            )

            expected_cache_id = "MockSubModel_mock_finalize_flush"
            cm = estimator._sub_cache_managers.get(expected_cache_id)
            assert cm is not None
            assert len(cm.pending_buffer["key"]) == 2

            data_dir = mock_cache_dir / expected_cache_id / "data"
            assert not data_dir.exists() or not list(data_dir.glob("*.parquet"))

            estimator.finalize_metrics()
            estimator.finalize_metrics()

            parquet_files = list(data_dir.glob("*.parquet"))
            assert parquet_files
            assert len(cm.pending_buffer["key"]) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests for sub-model cache discovery
# ──────────────────────────────────────────────────────────────────────────────
class TestArtOfPIVCacheDiscovery:
    """Tests for Art of PIV sub-model cache discovery from root_dir."""

    def test_discovers_sub_model_cache(self, mock_cache_dir):
        """Test that Art of PIV discovers existing sub-model caches."""
        # Pre-populate a sub-model cache
        cache_id = "MockSubModel_mock_m1"
        spec = {"epe": (np.dtype("float32"), ())}

        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id=cache_id,
            spec=spec,
        )
        cm.write(
            np.array([1, 2], dtype=np.uint64),
            {"epe": np.array([0.5, 0.6], dtype=np.float32)},
        )
        cm.flush()
        cm.close()

        # Create estimator
        model1 = MockSubModel("m1")
        estimator = build_testable_art_of_piv_estimator([model1])

        # Preload caches to discover existing cache
        estimator.preload_caches(str(mock_cache_dir), warm_start="all")

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 64, 64)),
            images2=jnp.zeros((2, 64, 64)),
            flow_fields=jnp.zeros((2, 64, 64, 2)),
            keys=jnp.array([1, 2], dtype=jnp.uint64),
        )

        # enrich() now uses preloaded unified index
        payload = estimator.enrich(batch, np.array([0, 1]))

        assert payload is not None
        # Should have found cached values from preloaded cache
        np.testing.assert_allclose(payload["epe_all"][:, 0], [0.5, 0.6])

    def test_writes_back_to_sub_model_cache(self, mock_cache_dir):
        """Test that computed misses are written back to sub-model caches."""
        # Use unique name to avoid cache collision with other tests
        model1 = MockSubModel("writeback_test", epe_value=0.99)
        estimator = build_testable_art_of_piv_estimator([model1])

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 64, 64)),
            images2=jnp.zeros((2, 64, 64)),
            flow_fields=jnp.zeros((2, 64, 64, 2)),
            keys=jnp.array([100, 101], dtype=jnp.uint64),
        )

        # Compute - will compute EPEs
        payload = estimator.enrich(batch, np.array([0, 1]))

        assert payload is not None
        np.testing.assert_allclose(payload["epe_all"][:, 0], [0.99, 0.99])

        # Note: Cold-start now supports lazy sub-cache creation/write-back.
        # Dedicated tests above validate creation and same-run in-memory hits.


# ──────────────────────────────────────────────────────────────────────────────
# Tests for get_cache_id_suffix
# ──────────────────────────────────────────────────────────────────────────────
class TestArtOfPIVCacheIdSuffix:
    """Tests for Art of PIV cache ID suffix generation."""

    def test_suffix_includes_art_of_piv_prefix(self):
        """Suffix should use ``_art_of_piv`` prefix and an 8-char hash."""
        estimator = build_testable_art_of_piv_estimator([MockSubModel("m1")])

        suffix = estimator.get_cache_id_suffix(trainable_state=None)
        assert suffix.startswith("_art_of_piv"), (
            "Cache ID suffix should start with '_art_of_piv'."
        )
        assert len(suffix) == len("_art_of_piv") + 8, (
            "Cache ID suffix should include an 8-character hash."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────────────────────
class TestArtOfPIVCacheIntegration:
    """Integration tests for Art of PIV caching with CacheManager."""

    def test_meta_model_aggregates_from_multiple_sub_models(
        self, mock_cache_dir
    ):
        """Meta-model should aggregate per-model EPEs across sub-estimators."""
        # Use unique names to avoid cache collision
        model1 = MockSubModel("agg1", epe_value=0.1)
        model2 = MockSubModel("agg2", epe_value=0.2)
        model3 = MockSubModel("agg3", epe_value=0.3)

        estimator = build_testable_art_of_piv_estimator(
            [model1, model2, model3]
        )

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 64, 64)),
            images2=jnp.zeros((2, 64, 64)),
            flow_fields=jnp.zeros((2, 64, 64, 2)),
            keys=jnp.array([1001, 1002], dtype=jnp.uint64),  # Unique keys
        )

        payload = estimator.enrich(batch, np.array([0, 1]))

        assert payload is not None
        epe_all = payload["epe_all"]

        # Check shape: 2 samples x 3 models
        assert epe_all.shape == (2, 3)

        # Check values match expected EPEs from each model
        np.testing.assert_allclose(epe_all[:, 0], [0.1, 0.1])
        np.testing.assert_allclose(epe_all[:, 1], [0.2, 0.2])
        np.testing.assert_allclose(epe_all[:, 2], [0.3, 0.3])

    def test_enrich_batch_bootstraps_art_of_piv_from_cache_manager(
        self, mock_cache_dir
    ):
        """enrich_batch should pass cache context required for bootstrap."""
        estimator = build_testable_art_of_piv_estimator(
            [MockSubModel("bootstrap", epe_value=0.55)]
        )
        estimator._unified_epe_index = None

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 64, 64)),
            images2=jnp.zeros((2, 64, 64)),
            flow_fields=jnp.zeros((2, 64, 64, 2)),
            keys=jnp.array([301, 302], dtype=jnp.uint64),
        )

        with CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="art_of_piv_meta_enrich_batch",
            spec={
                "epe_all": (np.dtype("float32"), (1,)),
                "epe_mask": (np.dtype("bool"), (1,)),
            },
            warm_start="index",
        ) as meta_cm:
            payload = enrich_batch(
                batch,
                estimator,
                cache_manager=meta_cm,
                trainable_state=None,
            )

        np.testing.assert_allclose(payload.epe_all[:, 0], [0.55, 0.55])
        assert estimator._cache_bootstrapped

    def test_partial_cache_hit_scenario(self, mock_cache_dir):
        """Test partial cache hit where some sub-models are cached."""
        # Use unique name to avoid cache collision
        cache_id = "MockSubModel_mock_partial_m1"
        spec = {"epe": (np.dtype("float32"), ())}

        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id=cache_id,
            spec=spec,
        )
        # Only cache key 2001, not key 2002
        cm.write(
            np.array([2001], dtype=np.uint64),
            {"epe": np.array([0.11], dtype=np.float32)},
        )
        cm.flush()
        cm.close()

        model1 = MockSubModel("partial_m1", epe_value=0.99)  # If computed, 0.99
        model2 = MockSubModel("partial_m2", epe_value=0.22)

        estimator = build_testable_art_of_piv_estimator([model1, model2])

        # Preload caches to load existing cached values
        estimator.preload_caches(str(mock_cache_dir), warm_start="all")

        batch = SynthpixBatch(
            images1=jnp.zeros((2, 64, 64)),
            images2=jnp.zeros((2, 64, 64)),
            flow_fields=jnp.zeros((2, 64, 64, 2)),
            keys=jnp.array([2001, 2002], dtype=jnp.uint64),
        )

        payload = estimator.enrich(batch, np.array([0, 1]))

        assert payload is not None

        epe_all = payload["epe_all"]

        # Model1, sample 0: from cache (0.11)
        np.testing.assert_allclose(epe_all[0, 0], 0.11, atol=1e-5)
        # Model1, sample 1: computed (0.99) since key 2002 wasn't cached
        np.testing.assert_allclose(epe_all[1, 0], 0.99, atol=1e-5)
        # Model2: all computed (0.22)
        np.testing.assert_allclose(epe_all[:, 1], [0.22, 0.22], atol=1e-5)
