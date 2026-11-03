"""Shared test doubles for ART of PIV caching tests."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator


class MockSubModel:
    """Minimal sub-estimator double for cache aggregation tests."""

    def __init__(
        self,
        name: str,
        *,
        supports_epe: bool = True,
        epe_value: float = 0.1,
        class_name: str = "MockSubModel",
        cache_suffix_prefix: str = "_mock_",
    ) -> None:
        self.name = name
        self.supports_epe = supports_epe
        self.epe_value = epe_value
        self._cache_suffix_prefix = cache_suffix_prefix
        # Match cache-id behavior expected by tests that infer IDs from class.
        self.__class__.__name__ = class_name

    def get_cache_id_suffix(self, trainable_state) -> str:
        """Return a deterministic suffix used to derive sub-cache IDs."""
        return f"{self._cache_suffix_prefix}{self.name}"

    def enrich(self, batch, miss_idxs, **kwargs):
        """Return constant EPE values for requested missing indices."""
        if not self.supports_epe:
            return None
        return {
            "epe": np.full(len(miss_idxs), self.epe_value, dtype=np.float32)
        }

    def compute_cache_miss(self, batch, miss_idxs, **kwargs):
        """Mirror ``enrich`` for compatibility with estimator code paths."""
        return self.enrich(batch, miss_idxs, **kwargs)


def build_testable_art_of_piv_estimator(
    sub_models: Sequence[MockSubModel],
    *,
    include_name_in_config: bool = False,
) -> ArtOfPIVEstimator:
    """Construct a minimal ArtOfPIVEstimator without full constructor setup."""
    estimator = ArtOfPIVEstimator.__new__(ArtOfPIVEstimator)
    estimator.sub_models = tuple(sub_models)
    estimator.num_estimators = len(sub_models)
    estimator.use_temporal_propagation = False
    estimator.use_last_estimate = False
    estimator.schedule = lambda step: 0.1
    estimator.estimator_configs = [
        {
            "estimator": model.name,
            "config": {},
            **({"name": model.name} if include_name_in_config else {}),
        }
        for model in sub_models
    ]
    estimator.q_estimate = lambda *args, **kwargs: jnp.zeros(
        (1, estimator.num_estimators)
    )
    estimator._initialize_estimators = False
    estimator._estimators_initialized = True
    estimator.sub_model_states = (None,) * estimator.num_estimators
    estimator._sub_cache_managers = {}
    estimator._cache_id_base = None
    estimator._unified_epe_index = {}
    estimator._cache_bootstrapped = False
    estimator._cache_root_dir = None
    estimator._allow_create_missing_sub_caches = False
    return estimator
