"""Robust integration tests for eval.py caching flows."""

import tempfile
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from eval import eval_full_dataset, evaluate_batches
from flowgym.common.base import Estimator
from flowgym.training.caching import CacheManager


class MockEvalBatch:
    def __init__(self, keys, height=32, width=32):
        self.images1 = jnp.zeros((len(keys), height, width))
        self.images2 = jnp.zeros((len(keys), height, width))
        self.flow_fields = jnp.zeros((len(keys), height, width, 2))
        self.keys = jnp.array([[0, k] for k in keys], dtype=jnp.uint32)
        self.mask = np.ones(len(keys), dtype=bool)
        self.params = MagicMock()
        self.params.seeding_densities = jnp.zeros((len(keys), height, width))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class MockEvalEstimator(Estimator):
    def __init__(self):
        super().__init__()
        self._compute_calls = []
        self._processed_metrics = []

    def is_oracle(self):
        return False

    def process_metrics(self, metrics):
        self._processed_metrics.append(metrics)
        return metrics

    def finalize_metrics(self):
        return {}

    def get_cache_id_suffix(self, trainable_state):
        return "_mock_eval"

    def _estimate(self, images, state, trainable_state, extras):
        return state, {"errors": np.zeros(len(images))}

    def enrich(self, batch, miss_idxs, **kwargs):
        self._compute_calls.append(miss_idxs)
        # Return dummy EPE
        return {"epe": np.ones(len(miss_idxs), dtype=np.float32) * 0.5}


def test_evaluate_batches_caching_robust():
    """Test evaluate_batches correctly enriches and passes payload."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = CacheManager(
            root_dir=tmp_dir,
            cache_id="eval_test",
            spec={"epe": (np.float32, ())},
        )

        model = MockEvalEstimator()

        # Batch 1: keys 200, 201
        batch1 = MockEvalBatch(keys=[200, 201])
        sampler = [batch1]

        def create_state_fn(img, key):
            return {"estimates": jnp.zeros((len(img), 1, 32, 32, 2))}

        def compute_estimate_fn(img2, state, trained_state, cache_payload=None):
            assert cache_payload is not None
            assert cache_payload.has_precomputed_errors
            assert cache_payload.epe is not None
            return state, {"errors": cache_payload.epe}

        # Run evaluate_batches
        results = evaluate_batches(
            model=model,
            sampler=sampler,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            trainable_state=None,
            num_batches=1,
            cache_manager=cache_manager,
            reset_sampler=False,
        )

        assert results["mean_error"] == 0.5
        assert len(model._compute_calls) == 1
        assert len(model._compute_calls[0]) == 2  # 2 misses

        # Verify it's now in cache
        _payload, hit = cache_manager.lookup(
            jnp.array([[0, 200], [0, 201]], dtype=jnp.uint32)
        )
        assert np.all(hit)


def test_eval_full_dataset_caching_robust():
    """Test eval_full_dataset with actual CacheManager on-disk state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "full_eval_test"
        cache_manager = CacheManager(
            root_dir=tmp_dir, cache_id=cache_id, spec={"epe": (np.float32, ())}
        )

        # Pre-seed cache for key 300
        cache_manager.write(
            jnp.array([[0, 300]], dtype=jnp.uint32),
            {"epe": np.array([0.1], dtype=np.float32)},
        )
        cache_manager.flush()

        model = MockEvalEstimator()

        # Batch: keys 300 (hit), 301 (miss)
        batch = MockEvalBatch(keys=[300, 301])

        class SimpleSampler:
            def __init__(self, b):
                self.b = b

            def __iter__(self):
                yield self.b

            def reset(self):
                pass

        sampler = SimpleSampler(batch)

        def create_state_fn(img, key):
            return {"estimates": jnp.zeros((len(img), 1, 32, 32, 2))}

        def compute_estimate_fn(img2, state, trained_state, cache_payload=None):
            # Payload should have 300 from cache (0.1) and 301 from enrich (0.5)
            # wait, enrich in MockEvalEstimator returns 0.5
            assert cache_payload is not None
            assert cache_payload.epe is not None
            return state, {"errors": cache_payload.epe}

        eval_full_dataset(
            model=model,
            sampler=sampler,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            trainable_state=None,
            cache_manager=cache_manager,
        )

        # Check that enrich was called only for 301 (1 sample)
        assert len(model._compute_calls) == 1
        assert len(model._compute_calls[0]) == 1  # Only 1 miss (301)
        assert model._compute_calls[0][0] == 1  # Index 1 in batch

        # Verify processed metrics
        # eval() calls process_metrics. The errors passed to eval were
        # [0.1, 0.5].
        # Mean should be 0.3
        assert len(model._processed_metrics) == 1
        np.testing.assert_allclose(
            model._processed_metrics[0]["errors"], [0.1, 0.5]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
