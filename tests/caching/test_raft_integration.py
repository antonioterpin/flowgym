"""Integration tests for RaftJaxEstimator caching in train_supervised loop."""

import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np

from flowgym.flow.raft.raft_jax import RaftJaxEstimator
from flowgym.training.caching import CacheManager
from flowgym.types import PRNGKey
from src.train_supervised import train_supervised


class MiniRaftBatch:
    """Minimal batch compatible with RaftJaxEstimator."""

    def __init__(self, batch_size=1, height=64, width=64):
        self.images1 = jnp.zeros((batch_size, height, width))
        self.images2 = jnp.zeros((batch_size, height, width))
        self.flow_fields = jnp.zeros((batch_size, height, width, 2))
        self.keys = jnp.array(
            [[0, i] for i in range(batch_size)], dtype=jnp.uint32
        )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


@patch("src.train_supervised.save_model")
def test_raft_integration_caching(mock_save_model):
    """Test RaftJaxEstimator caching in the real training loop."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Initialize a mini RAFT estimator
        model = RaftJaxEstimator(
            patch_size=32,
            patch_stride=32,  # Single patch for 64x64
            hidden_dim=8,  # Tiny
            context_dim=8,  # Tiny
            corr_levels=2,  # Minimal
            corr_radius=1,  # Minimal
            iters=1,  # Minimal
            optimizer_config={"name": "adam", "learning_rate": 1e-4},
        )

        # Create a dummy batch to initialize state
        init_batch = MiniRaftBatch(batch_size=1)
        key = jax.random.PRNGKey(42)
        trainable_state = model.create_trainable_state(init_batch.images1, key)

        cache_id = f"raft_test_{model.get_cache_id_suffix(trainable_state)}"
        cache_manager = CacheManager(
            root_dir=tmp_dir, cache_id=cache_id, spec={"epe": (np.float32, ())}
        )

        # 2. Setup sampler yielding 2 consistent batches
        keys1 = jnp.array([[0, 100], [0, 101]], dtype=jnp.uint32)
        keys2 = jnp.array(
            [[0, 101], [0, 102]], dtype=jnp.uint32
        )  # 101 is overlap

        batch1 = MiniRaftBatch(batch_size=2)
        batch1.keys = keys1

        batch2 = MiniRaftBatch(batch_size=2)
        batch2.keys = keys2

        sampler = MagicMock()
        sampler.__iter__.return_value = iter([batch1, batch2])
        sampler.shutdown = MagicMock()
        from goggles.history.types import History

        # 3. First run - should compute all (3 unique samples total:
        # 100, 101, 102). We need to provide a create_state_fn and
        # compute_estimate_fn for the loop
        def create_state_fn(images: jnp.ndarray, rng: PRNGKey) -> History:
            # RAFT expects "keys" in state for patch sampling
            h = History(history_images=images[:, None])
            h["keys"] = (
                jnp.zeros((len(images), 1, 2), dtype=jnp.uint32)
                .at[:, 0]
                .set(rng)
            )
            return h

        def compute_estimate_fn(model, state, params, rng):
            return jnp.zeros((len(state.history_images), 64, 64, 2))

        train_supervised(
            model=model,
            model_config={"config": {"jit": True}},
            trainable_state=trainable_state,
            out_dir=tmp_dir,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=sampler,
            num_batches=1,
            cache_manager=cache_manager,
            val_sampler=None,
        )

        # Verify cache has data for 100, 101, 102
        # Use compute_batch_keys logic to check
        k100 = (keys1[0, 0].astype(np.uint64) << 32) | keys1[0, 1].astype(
            np.uint64
        )
        k101 = (keys1[1, 0].astype(np.uint64) << 32) | keys1[1, 1].astype(
            np.uint64
        )

        payload, hit = cache_manager.lookup(
            np.array([k100, k101], dtype=np.uint64)
        )
        assert np.all(hit), f"Expected 2 hits, got {hit}"
        assert payload["epe"].shape == (2,)

        # 4. Verify that second batch (101, 102) utilized cache for 101
        # We can check enrich call count on model
        # But RaftJaxEstimator.enrich is called per batch with miss_idxs
        # If batch 2 had 101 cached, miss_idxs should be [1] (only for 102)

        with patch.object(
            RaftJaxEstimator,
            "enrich",
            wraps=model.enrich,
        ) as mock_miss:
            # Reset sampler for a second "epoch" or another run
            sampler.__iter__.return_value = iter([batch2])

            # Create a fresh cache manager with warm start to see the data
            cm2 = CacheManager(
                tmp_dir,
                cache_id,
                spec={"epe": (np.float32, ())},
                warm_start="index",
            )

            train_supervised(
                model=model,
                model_config={"config": {"jit": True}},
                trainable_state=trainable_state,
                out_dir=tmp_dir,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                sampler=sampler,
                num_batches=1,
                cache_manager=cm2,
                val_sampler=None,
            )

            # Check calls to enrich
            assert mock_miss.call_count == 1
            args, _kwargs = mock_miss.call_args
            # args[1] is miss_idxs
            miss_idxs = args[1]
            assert len(miss_idxs) == 1
            assert miss_idxs[0] == 1  # Only 102 was missing (index 1 in batch2)
