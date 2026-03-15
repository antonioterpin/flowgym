from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from synthpix.types import SynthpixBatch

from eval import eval_full_dataset
from flowgym.common.base import Estimator


def test_eval_full_dataset_with_caching():
    """Test that eval_full_dataset correctly uses the cache manager."""

    # Mock dependencies
    mock_model = MagicMock(spec=Estimator)
    # Mock default return for process_metrics to avoid iteration errors
    mock_model.process_metrics.return_value = {"errors": np.array([0.1])}
    mock_model.finalize_metrics.return_value = {}
    mock_model.is_oracle.return_value = False

    # Mock enrich to return payload for missing keys
    def enrich_fn(batch, miss_idxs, **kwargs):
        return {"cached_value": np.array([123])}

    mock_model.enrich.side_effect = enrich_fn

    # Create a dummy batch
    batch = SynthpixBatch(
        images1=np.zeros((1, 32, 32, 3)),
        images2=np.zeros((1, 32, 32, 3)),
        flow_fields=np.zeros((1, 32, 32, 2)),
        keys=np.array([123]),
        mask=np.array([True]),
    )

    # Mock sampler as an iterator
    mock_sampler = MagicMock()
    mock_sampler.__iter__ = Mock(return_value=iter([batch]))
    mock_sampler.reset = Mock()

    # Mock CacheManager
    mock_cache_manager = MagicMock()
    # Support context manager
    mock_cache_manager.__enter__.return_value = mock_cache_manager
    mock_cache_manager.__exit__.return_value = None
    # Mock lookup to return a miss
    mock_cache_manager.lookup.return_value = (
        {"cached_value": np.zeros((1,))},
        np.array([False]),
    )
    mock_cache_manager.spec = {"cached_value": (np.float32, ())}
    mock_cache_manager.write = MagicMock()

    # Mock other functions
    mock_create_state_fn = Mock(
        return_value={"estimates": np.zeros((1, 1, 32, 32, 2))}
    )
    mock_compute_estimate_fn = Mock(
        return_value=({"estimates": np.zeros((1, 1, 32, 32, 2))}, {})
    )

    # Run eval_full_dataset
    try:
        eval_full_dataset(
            model=mock_model,
            sampler=mock_sampler,
            create_state_fn=mock_create_state_fn,
            compute_estimate_fn=mock_compute_estimate_fn,
            trainable_state=None,
            estimate_type="flow",
            cache_manager=mock_cache_manager,
        )
    except TypeError as e:
        pytest.fail(
            f"eval_full_dataset raised TypeError, likely due to missing "
            f"arguments: {e}"
        )

    # Verify interactions
    # 1. Verify cache_manager context was entered
    mock_cache_manager.__enter__.assert_called_once()
    mock_cache_manager.__exit__.assert_called_once()

    # 2. Verify cache lookup was called
    mock_cache_manager.lookup.assert_called()

    # 3. Verify model.enrich was called for cache misses
    mock_model.enrich.assert_called()

    # 4. Verify sampler was iterated
    mock_sampler.__iter__.assert_called()

    # 5. Verify cache_payload was passed to compute_estimate_fn
    _, kwargs = mock_compute_estimate_fn.call_args
    assert "cache_payload" in kwargs
    # Payload should have the enriched data
    assert kwargs["cache_payload"] is not None
    assert "cached_value" in kwargs["cache_payload"].extras
