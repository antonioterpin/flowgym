"""Tests for RAFT estimator caching functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthpix import SynthpixBatch

from flowgym.training.caching import CacheManager
from flowgym.types import CachePayload


class TestRaftCacheIdSuffix:
    """Test RAFT estimator cache ID suffix generation."""

    def test_cache_suffix_includes_config_hash(self):
        """Test that cache suffix includes config hash."""
        from flowgym.flow.raft.raft_jax import RaftJaxEstimator

        estimator = RaftJaxEstimator(
            patch_size=32,
            patch_stride=8,
            hidden_dim=128,
            context_dim=128,
            corr_levels=4,
            corr_radius=4,
            iters=12,
        )

        suffix = estimator.get_cache_id_suffix(trainable_state=None)

        # Suffix should start with _c (config)
        assert suffix.startswith("_c")
        # Without trainable_state, no weights hash
        assert "_w" not in suffix

    def test_cache_suffix_changes_with_config(self):
        """Test that different configs produce different suffixes."""
        from flowgym.flow.raft.raft_jax import RaftJaxEstimator

        estimator1 = RaftJaxEstimator(
            patch_size=32,
            patch_stride=8,
            iters=12,
        )
        estimator2 = RaftJaxEstimator(
            patch_size=64,  # Different patch size
            patch_stride=8,
            iters=12,
        )

        suffix1 = estimator1.get_cache_id_suffix(trainable_state=None)
        suffix2 = estimator2.get_cache_id_suffix(trainable_state=None)

        assert suffix1 != suffix2

    @pytest.mark.run_explicitly
    def test_cache_suffix_includes_weights_hash(self, tmp_path):
        """Test cache suffix includes weights hash with trainable_state."""
        from flowgym.flow.raft.raft_jax import RaftJaxEstimator

        estimator = RaftJaxEstimator(
            patch_size=32,
            patch_stride=8,
            hidden_dim=64,  # Smaller for faster init
            context_dim=64,
            corr_levels=2,
            corr_radius=2,
            iters=6,
        )

        # Create trainable state
        dummy_input = jnp.zeros((1, 64, 64))
        key = jax.random.PRNGKey(42)
        trainable_state = estimator.create_trainable_state(dummy_input, key)

        suffix = estimator.get_cache_id_suffix(trainable_state)

        # Suffix should include both _c (config) and _w (weights)
        assert "_c" in suffix
        assert "_w" in suffix


class TestRaftComputeCacheMiss:
    """Test RAFT enrich functionality."""

    @pytest.mark.run_explicitly
    def test_enrich_returns_epe(self, tmp_path):
        """Test that enrich returns EPE payload."""
        from flowgym.flow.raft.raft_jax import RaftJaxEstimator

        estimator = RaftJaxEstimator(
            patch_size=32,
            patch_stride=8,
            hidden_dim=64,
            context_dim=64,
            corr_levels=2,
            corr_radius=2,
            iters=6,
        )

        # Create trainable state
        B, H, W = 2, 64, 64
        dummy_input = jnp.zeros((1, H, W))
        key = jax.random.PRNGKey(42)
        trainable_state = estimator.create_trainable_state(dummy_input, key)

        # Create batch
        batch = SynthpixBatch(
            images1=jax.random.uniform(key, (B, H, W)),
            images2=jax.random.uniform(key, (B, H, W)),
            flow_fields=jnp.zeros((B, H, W, 2)),
            keys=jnp.array([1, 2], dtype=jnp.uint64),
        )

        miss_idxs = np.array([0, 1])
        payload = estimator.enrich(
            batch, miss_idxs, trainable_state=trainable_state
        )

        assert payload is not None
        assert "epe" in payload
        assert payload["epe"].shape == (B,)
        assert "relative_epe" in payload


class TestRaftCacheIntegration:
    """Integration tests for RAFT caching with CacheManager."""

    @pytest.mark.run_explicitly
    def test_raft_enrich_cycle(self, mock_cache_dir):
        """Test full enrich cycle with RAFT model."""
        from flowgym.flow.raft.raft_jax import RaftJaxEstimator

        estimator = RaftJaxEstimator(
            patch_size=32,
            patch_stride=8,
            hidden_dim=64,
            context_dim=64,
            corr_levels=2,
            corr_radius=2,
            iters=6,
        )

        # Create trainable state
        B, H, W = 2, 64, 64
        key = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, H, W))
        trainable_state = estimator.create_trainable_state(dummy_input, key)

        # Build cache ID
        suffix = estimator.get_cache_id_suffix(trainable_state)
        cache_id = f"RaftJaxEstimator{suffix}"

        spec = {
            "epe": (np.dtype("float32"), ()),
            "relative_epe": (np.dtype("float32"), ()),
        }
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id=cache_id,
            spec=spec,
        )

        # Create batch
        batch = SynthpixBatch(
            images1=jax.random.uniform(key, (B, H, W)),
            images2=jax.random.uniform(key, (B, H, W)),
            flow_fields=jnp.zeros((B, H, W, 2)),
            keys=jnp.array([100, 101], dtype=jnp.uint64),
        )

        # First enrich - compute and cache
        payload = cm.enrich(batch, estimator, trainable_state=trainable_state)
        assert payload is not None
        assert "epe" in payload
        epe_first = payload["epe"].copy()

        # Flush to disk
        cm.flush()

        # Second enrich - should hit cache
        payload2 = cm.enrich(batch, estimator, trainable_state=trainable_state)
        np.testing.assert_allclose(payload2["epe"], epe_first)


def test_raft_uses_cached_metrics_without_running_flow(monkeypatch):
    """RAFT should short-circuit when cache provides precomputed metrics."""
    from flowgym.flow.raft.raft_jax import RaftJaxEstimator

    estimator = RaftJaxEstimator(
        patch_size=32,
        patch_stride=32,
        hidden_dim=64,
        context_dim=64,
        corr_levels=2,
        corr_radius=2,
        iters=4,
        optimizer_config={"name": "adam", "learning_rate": 0.0001},
    )

    B, H, W = 2, 64, 64
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key, 2)
    images1 = jax.random.uniform(k1, (B, H, W))
    images2 = jax.random.uniform(k2, (B, H, W))

    init_flow = jnp.zeros((B, H, W, 2), dtype=jnp.float32)
    state = estimator.create_state(
        images1, init_flow, image_history_size=2, rng=key
    )
    trainable_state = estimator.create_trainable_state(images1, key)

    cache_payload = CachePayload(
        has_precomputed_errors=True,
        epe=np.array([0.2, 0.3], dtype=np.float32),
        relative_epe=np.array([1.2, 1.3], dtype=np.float32),
    )

    # Monkeypatch the model's apply method to fail if called
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "model.apply should not be called when cache hit provides metrics"
        )

    monkeypatch.setattr(estimator.model, "apply", fail_if_called)

    # Call the estimator with cache_payload
    new_state, metrics = estimator(
        images2,
        state,
        trainable_state,
        cache_payload=cache_payload,
    )

    # Verify metrics match cached values
    np.testing.assert_allclose(np.asarray(metrics["errors"]), [0.2, 0.3])
    np.testing.assert_allclose(
        np.asarray(metrics["relative_errors"]), [1.2, 1.3]
    )
    # Verify state shape is correct
    assert new_state["estimates"].shape == (B, 2, H, W, 2)
