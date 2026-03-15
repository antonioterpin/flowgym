"""Tests for caching integration with the training loop."""

import tempfile
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np

from flowgym.training.caching import CacheManager
from flowgym.types import NNEstimatorTrainableState, SupervisedTrainStep
from src.train_supervised import train_supervised


class MockTrainStep(SupervisedTrainStep):
    def __call__(self, trainable_state, experience):
        return (jnp.array(0.0), trainable_state, {"loss": jnp.array(0.0)})


class MockEstimator:
    def create_train_step(self):
        return MockTrainStep()

    def prepare_experience_for_replay(self, experience, trainable_state):
        return experience

    def enrich(self, batch, miss_idxs, **kwargs):
        """Return None by default (opt out of caching)."""
        return None


class MockBatch:
    def __init__(self, batch_size=2, keys=None):
        self.images1 = jnp.zeros((batch_size, 32, 32, 3))
        self.images2 = jnp.zeros((batch_size, 32, 32, 3))
        self.flow_fields = jnp.zeros((batch_size, 32, 32, 2))
        self.keys = (
            keys if keys is not None else np.arange(batch_size, dtype=np.uint64)
        )


def mock_sampler(num_batches, keys_list):
    """Yields MockBatches with specified keys."""
    for keys in keys_list:
        yield MockBatch(batch_size=len(keys), keys=keys)


@patch("src.train_supervised.save_model")
def test_train_supervised_caching_integration(mock_save_model):
    """Integration test for caching in train_supervised."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_id = "integration_test"
        cache_manager = CacheManager(
            root_dir=tmp_dir,
            cache_id=cache_id,
            spec={"values": (np.float32, (2,))},
            warm_start="none",
        )

        # 1. Setup Mock Components
        model = MockEstimator()
        model_config = {"config": {"jit": False}}
        trainable_state = MagicMock(spec=NNEstimatorTrainableState)
        trainable_state.params = {}
        trainable_state.opt_state = {}
        trainable_state.extras = {}

        create_state_fn = MagicMock(return_value=MagicMock())
        compute_estimate_fn = MagicMock()

        # Define model.enrich to compute payload for missing keys
        def model_enrich(batch, miss_idxs, **kwargs):
            keys = np.asarray(batch.keys)
            miss_keys = keys[miss_idxs]
            # Payload values: [[key, key], ...] just to verify
            values = np.stack([miss_keys, miss_keys], axis=1).astype(np.float32)
            return {"values": values}

        model.enrich = MagicMock(side_effect=model_enrich)

        # 2. Run Training - Pass 1 (All Misses)
        # 2 batches: [1, 2], [3, 4]
        keys_pass1 = [
            np.array([1, 2], dtype=np.uint64),
            np.array([3, 4], dtype=np.uint64),
        ]

        sampler_pass1 = MagicMock()
        sampler_pass1.__iter__.return_value = (
            MockBatch(keys=k) for k in keys_pass1
        )
        sampler_pass1.shutdown = MagicMock()

        train_supervised(
            model=model,
            model_config=model_config,
            trainable_state=trainable_state,
            out_dir=tmp_dir,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=sampler_pass1,
            num_batches=2,
            cache_manager=cache_manager,
            val_sampler=None,
        )

        # Verification Pass 1:
        # Check that enrich was called for all 4 items
        # It's called per batch.
        assert model.enrich.call_count == 2

        # Flush pending writes before verification
        cache_manager.flush()

        # Check data on disk via new CacheManager
        cm_verify = CacheManager(
            tmp_dir, cache_id, spec={"values": (np.float32, (2,))}
        )
        payload, hit = cm_verify.lookup(np.array([1, 2, 3, 4], dtype=np.uint64))

        assert np.all(hit)
        np.testing.assert_array_equal(payload["values"][:, 0], [1, 2, 3, 4])

        # 3. Run Training - Pass 2 (Mixed Hits/Misses)
        # Batch: [2, 5] (2 is hit, 5 is miss)
        keys_pass2 = [np.array([2, 5], dtype=np.uint64)]

        sampler_pass2 = MagicMock()
        sampler_pass2.__iter__.return_value = (
            MockBatch(keys=k) for k in keys_pass2
        )
        sampler_pass2.shutdown = MagicMock()

        model.enrich.reset_mock()

        train_supervised(
            model=model,
            model_config=model_config,
            trainable_state=trainable_state,
            out_dir=tmp_dir,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=sampler_pass2,
            num_batches=1,
            cache_manager=cache_manager,
            val_sampler=None,
        )

        # Verification Pass 2:
        # model.enrich should be called ONCE for the batch
        # And inside, it receives miss_idxs.
        # But our mock side_effect handles the partial return.

        assert model.enrich.call_count == 1

        # Verify 5 is now in cache
        payload_5, hit_5 = cm_verify.lookup(np.array([5], dtype=np.uint64))
        assert hit_5[0]
        assert payload_5["values"][0, 0] == 5.0


@patch("src.train_supervised.save_model")
def test_estimator_enrich(mock_save_model):
    """Test that Estimator.enrich is called for cache misses."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = CacheManager(
            root_dir=tmp_dir,
            cache_id="est_compute",
            spec={"values": (np.float32, (1,))},
            warm_start="none",
        )

        # Mock Estimator that returns a payload
        model = MagicMock(spec=MockEstimator)
        model.create_train_step.return_value = MockTrainStep()

        def enrich_fn(batch, miss_idxs, **kwargs):
            # Return fixed value 99.0 for misses
            N = len(miss_idxs)
            return {"values": np.full((N, 1), 99.0, dtype=np.float32)}

        model.enrich.side_effect = enrich_fn

        # Batch with missing key
        batch = MockBatch(keys=np.array([10], dtype=np.uint64))

        sampler = MagicMock()
        sampler.__iter__.return_value = iter([batch])
        sampler.shutdown = MagicMock()

        trainable_state = MagicMock(spec=NNEstimatorTrainableState)
        trainable_state.params = {}
        trainable_state.opt_state = {}
        trainable_state.extras = {}

        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=trainable_state,
            out_dir=tmp_dir,
            create_state_fn=MagicMock(),
            compute_estimate_fn=MagicMock(),
            sampler=sampler,
            num_batches=1,
            cache_manager=cache_manager,
            val_sampler=None,
        )

        # Verify
        assert model.enrich.call_count == 1

        # Flush before verification
        cache_manager.flush()

        # Check disk
        payload, hit = cache_manager.lookup(np.array([10], dtype=np.uint64))
        assert hit[0]
        assert payload["values"][0, 0] == 99.0
