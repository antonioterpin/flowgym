"""Cache enrichment reuses compilation without retaining discarded models."""

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthpix.types import SynthpixBatch

from flowgym.flow.dummy import DummyEstimator


@pytest.mark.parametrize("with_ground_truth", [False, True])
def test_enrichment_releases_discarded_estimators(
    with_ground_truth: bool, monkeypatch: pytest.MonkeyPatch
):
    """Completed enrichment must not root models in a shared JIT cache."""
    monkeypatch.setattr(
        "flowgym.common.base.estimator.logger.info", lambda _: None
    )
    images = jnp.ones((1, 16, 16), dtype=jnp.float32)
    batch = SynthpixBatch(
        images1=images,
        images2=images,
        flow_fields=jnp.zeros((1, 16, 16, 2)) if with_ground_truth else None,
        keys=jnp.zeros((1,), dtype=jnp.uint32),
    )
    refs = []
    for _ in range(3):
        estimator = DummyEstimator()
        refs.append(weakref.ref(estimator))
        payload = estimator.enrich(batch, np.array([0]))
        jax.block_until_ready(payload)
        assert payload["corr16_peak"].shape == (1,), "Score each missing pair"
        del estimator, payload
    gc.collect()
    assert all(ref() is None for ref in refs), (
        "Discarded models must be collectible without jax.clear_caches()"
    )


def test_enrichment_reuses_compilation(monkeypatch: pytest.MonkeyPatch):
    """Repeated enrichment with the same shape traces the estimator once."""
    calls = []
    original = DummyEstimator._estimate

    def estimate(self, *args):
        calls.append(None)
        return original(self, *args)

    monkeypatch.setattr(DummyEstimator, "_estimate", estimate)
    estimator = DummyEstimator()
    images = jnp.ones((1, 16, 16), dtype=jnp.float32)
    batch = SynthpixBatch(
        images1=images,
        images2=images,
        flow_fields=None,
        keys=jnp.zeros((1,), dtype=jnp.uint32),
    )
    for _ in range(3):
        payload = estimator.enrich(batch, np.array([0]))
        jax.block_until_ready(payload)
    assert len(calls) == 1, "Reuse the compiled enrichment across batches"
