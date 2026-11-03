"""Unit tests for flowgym.training.caching: cache key generation.

Covers filename-based keys for real images and batch.keys for synthetic images.
"""

import hashlib
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
from synthpix import SynthpixBatch

from flowgym.training.caching import CacheManager, enrich_batch


class TestCacheKeyGenerationForRealImages:
    """Tests for cache key generation with real images."""

    def test_real_image_key_based_on_filename_only(self, mock_cache_dir):
        """Real-image batches hit the cache across epochs by name."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_real_keys",
            spec=spec,
        )

        # Create a batch mimicking RealImageSampler output
        # - has `files` attribute (filenames)
        # - has NO `params` attribute (no generation params)
        batch = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=jnp.array(
                [999, 888], dtype=jnp.uint32
            ),  # These should be ignored
        )
        # Add files attribute (simulating RealImageSampler)
        batch = batch.update(
            files=["path/to/image_001.mat", "path/to/image_002.mat"]
        )

        estimator = MagicMock()
        estimator.enrich.return_value = {
            "epe": np.array([0.1, 0.2], dtype=np.float32)
        }

        # First enrich call
        payload1 = enrich_batch(batch, estimator, cache_manager=cm)
        assert estimator.enrich.call_count == 1

        # Same filenames, different random keys (simulating different epoch)
        batch2 = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=jnp.array([111, 222], dtype=jnp.uint32),  # Different keys
        )
        batch2 = batch2.update(
            files=["path/to/image_001.mat", "path/to/image_002.mat"]
        )

        # Reset mock and call again
        estimator.enrich.reset_mock()
        payload2 = enrich_batch(batch2, estimator, cache_manager=cm)

        # Should be cache hit - same filenames means same cache keys
        assert estimator.enrich.call_count == 0
        np.testing.assert_allclose(payload1.epe, payload2.epe)

    def test_real_image_key_ignores_path_prefix(self, mock_cache_dir):
        """Cache key is derived from the basename only, not the full path."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_basename_keys",
            spec=spec,
        )

        batch1 = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=jnp.array([1], dtype=jnp.uint32),
        )
        batch1 = batch1.update(files=["/absolute/path/to/test_file.mat"])

        estimator = MagicMock()
        estimator.enrich.return_value = {
            "epe": np.array([0.5], dtype=np.float32)
        }

        # First enrich
        enrich_batch(batch1, estimator, cache_manager=cm)
        assert estimator.enrich.call_count == 1

        # Same filename but different path prefix
        batch2 = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=jnp.array([2], dtype=jnp.uint32),
        )
        batch2 = batch2.update(
            files=["different/dir/test_file.mat"]
        )  # Same basename

        estimator.enrich.reset_mock()
        enrich_batch(batch2, estimator, cache_manager=cm)

        # Should be cache hit - same basename = same cache key
        assert estimator.enrich.call_count == 0

    def test_different_filenames_produce_different_keys(self, mock_cache_dir):
        """Different filenames produce distinct keys and misses."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_diff_files",
            spec=spec,
        )

        batch1 = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=jnp.array([1], dtype=jnp.uint32),
        )
        batch1 = batch1.update(files=["file_A.mat"])

        estimator = MagicMock()
        estimator.enrich.return_value = {
            "epe": np.array([0.5], dtype=np.float32)
        }

        enrich_batch(batch1, estimator, cache_manager=cm)
        assert estimator.enrich.call_count == 1

        # Different filename
        batch2 = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=jnp.array([1], dtype=jnp.uint32),
        )
        batch2 = batch2.update(files=["file_B.mat"])  # Different filename

        estimator.enrich.reset_mock()
        enrich_batch(batch2, estimator, cache_manager=cm)

        # Should be cache miss - different filename = different cache key
        assert estimator.enrich.call_count == 1


class TestCacheKeyGenerationForSyntheticImages:
    """Tests for cache key generation with synthetic images."""

    def test_synthetic_image_uses_batch_keys(self, mock_cache_dir):
        """Synthetic batches use batch.keys as cache keys, ignoring params."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_synth_keys",
            spec=spec,
        )

        # Synthetic batch: has `params` attribute, NO `files`
        batch1 = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=np.array(
                [1001, 1002], dtype=np.uint64
            ),  # Use numpy for stable uint64
        )
        # Add params to signal synthetic batch
        batch1 = batch1.update(params={"some": "generation_params"})

        estimator = MagicMock()
        estimator.enrich.return_value = {
            "epe": np.array([0.1, 0.2], dtype=np.float32)
        }

        enrich_batch(batch1, estimator, cache_manager=cm)
        assert estimator.enrich.call_count == 1

        # Same keys should hit cache
        batch2 = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=np.array([1001, 1002], dtype=np.uint64),
        )
        batch2 = batch2.update(params={"different": "params"})

        estimator.enrich.reset_mock()
        enrich_batch(batch2, estimator, cache_manager=cm)

        # Should hit cache - same keys
        assert estimator.enrich.call_count == 0

    def test_synthetic_different_keys_produce_miss(self, mock_cache_dir):
        """Distinct batch.keys each produce a cache miss."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_synth_diff_keys",
            spec=spec,
        )

        batch1 = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=np.array([1001, 1002], dtype=np.uint64),
        )
        batch1 = batch1.update(params={"gen": "params"})

        estimator = MagicMock()
        estimator.enrich.return_value = {
            "epe": np.array([0.1, 0.2], dtype=np.float32)
        }

        enrich_batch(batch1, estimator, cache_manager=cm)
        assert estimator.enrich.call_count == 1

        # Different keys
        batch2 = SynthpixBatch(
            images1=jnp.zeros((2, 32, 32)),
            images2=jnp.zeros((2, 32, 32)),
            flow_fields=jnp.zeros((2, 32, 32, 2)),
            keys=np.array([2001, 2002], dtype=np.uint64),  # Different keys
        )
        batch2 = batch2.update(params={"gen": "params"})

        estimator.enrich.reset_mock()
        enrich_batch(batch2, estimator, cache_manager=cm)

        # Should be cache miss - different keys
        assert estimator.enrich.call_count == 1


class TestBatchTypeDetection:
    """Test how batch type (real vs synthetic) is detected."""

    def test_files_without_params_triggers_filename_keys(self, mock_cache_dir):
        """Batch with files and no params uses filename-derived cache keys."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_detection_files",
            spec=spec,
        )

        # Compute expected hash from filename
        filename = "unique_file_xyz.mat"
        _ = int(hashlib.md5(filename.encode()).hexdigest()[:16], 16)

        batch = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=jnp.array([12345], dtype=jnp.uint32),  # Should be overwritten
        )
        batch = batch.update(files=[f"some/path/{filename}"])

        estimator = MagicMock()
        captured_miss_idxs = []

        def capture_call(b, miss_idxs, **kwargs):
            captured_miss_idxs.append(miss_idxs)
            return {"epe": np.array([0.1], dtype=np.float32)}

        estimator.enrich.side_effect = capture_call
        enrich_batch(batch, estimator, cache_manager=cm)

        # Verify the internal key assignment happened
        # The key used should be based on the hash, not the original batch.keys
        # We verify by checking if the same filename produces same result
        estimator.enrich.reset_mock()
        batch2 = batch.update(
            keys=jnp.array([99999], dtype=jnp.uint32)
        )  # Different key
        batch2 = batch2.update(
            files=[f"different/path/{filename}"]
        )  # Same basename
        enrich_batch(batch2, estimator, cache_manager=cm)

        # Should be cache hit because same filename
        assert estimator.enrich.call_count == 0

    def test_files_with_params_uses_batch_keys(self, mock_cache_dir):
        """Batch with both files and params uses batch.keys, not filenames."""
        spec = {"epe": (np.dtype("float32"), ())}
        cm = CacheManager(
            root_dir=str(mock_cache_dir),
            cache_id="test_files_with_params",
            spec=spec,
        )

        # Batch with BOTH files and params - should use batch.keys
        batch = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=np.array([5001], dtype=np.uint64),
        )
        batch = batch.update(
            files=["some_file.mat"],
            params={
                "generation": "parameters"
            },  # Having params means synthetic
        )

        estimator = MagicMock()
        estimator.enrich.return_value = {
            "epe": np.array([0.1], dtype=np.float32)
        }

        enrich_batch(batch, estimator, cache_manager=cm)
        assert estimator.enrich.call_count == 1

        # Same filename but different key - should miss
        batch2 = SynthpixBatch(
            images1=jnp.zeros((1, 32, 32)),
            images2=jnp.zeros((1, 32, 32)),
            flow_fields=jnp.zeros((1, 32, 32, 2)),
            keys=np.array([5002], dtype=np.uint64),  # Different key
        )
        batch2 = batch2.update(
            files=["some_file.mat"],  # Same filename
            params={"generation": "parameters"},
        )

        estimator.enrich.reset_mock()
        enrich_batch(batch2, estimator, cache_manager=cm)

        # Should be cache miss - params present means we use batch.keys
        assert estimator.enrich.call_count == 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
