"""Tests for Estimator caching behavior in __call__."""

import jax.numpy as jnp
import pytest
from goggles.history import create_history
from goggles.history.spec import HistorySpec
from goggles.history.types import History

from flowgym.common.base import Estimator


class MockEstimator(Estimator):
    """Estimator with spy for _estimate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimate_called = False
        self.last_extras = None

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state,
        extras: dict,
    ):
        self.estimate_called = True
        self.last_extras = extras
        B = images.shape[0]
        estimates = jnp.zeros((B, 2))
        return estimates, extras, {}


def _make_state(B: int, H: int, W: int, img_T: int, est_T: int) -> History:
    cfg = {
        "images": {
            "length": img_T,
            "shape": (H, W),
            "dtype": jnp.float32,
            "init": "zeros",
        },
        "estimates": {
            "length": est_T,
            "shape": (2,),
            "dtype": jnp.float32,
            "init": "zeros",
        },
    }
    spec = HistorySpec.from_config(cfg)
    return create_history(spec, B, rng=None)


@pytest.fixture
def estimator():
    return MockEstimator()


def test_call_passes_epe_payload_to_estimator(estimator):
    """__call__ should pass cache payload through to downstream estimator."""
    B, H, W = 2, 4, 4
    state = _make_state(B, H, W, 1, 1)
    images = jnp.zeros((B, H, W))

    cache_payload = {"epe": jnp.array([0.5, 0.6])}

    new_state, metrics = estimator(
        images, state, None, cache_payload=cache_payload
    )

    assert estimator.estimate_called
    assert estimator.last_extras is not None
    assert estimator.last_extras["cache_payload"].has_precomputed_errors
    assert jnp.allclose(
        estimator.last_extras["cache_payload"].epe, cache_payload["epe"]
    )
    assert metrics == {}
    assert new_state["estimates"].shape == (B, 1, 2)


def test_call_normalizes_legacy_errors_field(estimator):
    """Legacy 'errors' payload should be normalized into cache_payload.epe."""
    B, H, W = 2, 4, 4
    state = _make_state(B, H, W, 1, 1)
    images = jnp.zeros((B, H, W))

    cache_payload = {"errors": jnp.array([0.1, 0.2])}

    _new_state, metrics = estimator(
        images, state, None, cache_payload=cache_payload
    )

    assert estimator.estimate_called
    assert estimator.last_extras is not None
    assert estimator.last_extras["cache_payload"].has_precomputed_errors
    assert jnp.allclose(
        estimator.last_extras["cache_payload"].epe, cache_payload["errors"]
    )
    assert metrics == {}


def test_call_exposes_cached_estimates_without_apply(estimator):
    """Cached estimates are passed through.

    Only downstream estimator can apply them.
    """
    B, H, W = 2, 4, 4
    state = _make_state(B, H, W, 1, 1)
    images = jnp.zeros((B, H, W))

    cached_est = jnp.ones((B, 2)) * 7.0
    cache_payload = {"errors": jnp.array([0.1, 0.2]), "estimates": cached_est}

    new_state, _ = estimator(images, state, None, cache_payload=cache_payload)

    assert estimator.estimate_called
    assert estimator.last_extras is not None
    assert jnp.allclose(
        estimator.last_extras["cache_payload"].estimates, cached_est
    )
    # MockEstimator always returns zeros; base class should not override this.
    assert jnp.allclose(new_state["estimates"][:, -1], jnp.zeros((B, 2)))


def test_call_does_not_persist_cache_payload(estimator):
    """__call__ should NOT persist cache_payload keys into the state."""
    B, H, W = 2, 4, 4
    state = _make_state(B, H, W, 1, 1)
    images = jnp.zeros((B, H, W))

    # Add a large dummy array that should NOT be persisted
    cache_payload = {
        "epe": jnp.array([0.5, 0.6]),
        "transient_large_array": jnp.ones((10, 10)),
    }

    new_state, _ = estimator(images, state, None, cache_payload=cache_payload)

    assert "transient_large_array" not in new_state
    assert "epe" not in new_state
