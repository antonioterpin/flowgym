"""Unit tests for flowgym.common.base Estimator cache_payload handling.

Covers cache_payload injection, legacy 'errors' normalization, cached
estimate pass-through, and transient-key exclusion from persisted state.
"""

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
    """Cached estimates are passed through; base class does not apply them."""
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


class ConfiguredEstimator(MockEstimator):
    """Estimator whose output depends on one parameter."""

    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha


def test_get_config_records_from_config_params():
    """The default config is every constructor param from_config passes."""
    est = ConfiguredEstimator.from_config(
        {"alpha": 0.5, "optimizer_config": {"name": "adam"}, "unused": 1}
    )
    assert est.get_config() == {
        "alpha": 0.5,
        "optimizer_config": {"name": "adam"},
    }


def test_get_config_requires_from_config(estimator):
    """Directly constructed estimators without an override cannot be hashed."""
    with pytest.raises(NotImplementedError, match="from_config"):
        estimator.get_config()
    with pytest.raises(NotImplementedError):
        estimator.get_cache_id_suffix(None)


def test_cache_id_suffix_hashes_config():
    """The default suffix is the md5 of the sorted-json get_config."""
    suffix = ConfiguredEstimator.from_config(
        {"alpha": 0.5}
    ).get_cache_id_suffix(None)
    # md5(json.dumps({"alpha": 0.5}, sort_keys=True))[:8]
    assert suffix == "_c6c2e41f9"
    other = ConfiguredEstimator.from_config({"alpha": 0.6})
    assert suffix != other.get_cache_id_suffix(None)


def test_cache_id_suffix_includes_preprocessing():
    """Preprocessing changes the output, so it changes the suffix."""
    plain = ConfiguredEstimator.from_config({"alpha": 0.5})
    capped = ConfiguredEstimator.from_config(
        {
            "alpha": 0.5,
            "preprocess": [{"name": "intensity_capping", "n": 2.0}],
        }
    )
    assert plain.get_cache_id_suffix(None) != capped.get_cache_id_suffix(None)


def test_cache_id_suffix_empty_without_params():
    """An estimator built with no parameters keeps the bare suffix."""
    assert MockEstimator.from_config({}).get_cache_id_suffix(None) == ""


class HookEstimator(MockEstimator):
    """Estimator that names its cache through the hook, like RAFT does."""

    def _cache_id_suffix(self, trainable_state):
        del trainable_state
        return "_hook"


def test_overriding_get_cache_id_suffix_is_rejected():
    """Subclasses must use the hook so processing is always hashed."""
    with pytest.raises(TypeError, match="_cache_id_suffix"):

        class _Bad(MockEstimator):
            def get_cache_id_suffix(self, trainable_state):
                return "_bad"


def test_hook_suffix_gets_processing_tag():
    """A custom hook keeps its id; non-empty steps are appended to it."""
    crop = {"name": "clahe", "clip_limit": 2.0}
    plain = HookEstimator().get_cache_id_suffix(None)
    capped = HookEstimator(preprocessing_steps=[crop]).get_cache_id_suffix(None)
    other = HookEstimator(
        preprocessing_steps=[{**crop, "clip_limit": 3.0}]
    ).get_cache_id_suffix(None)
    assert plain == "_hook"
    assert capped.startswith("_hook_p")
    assert len({plain, capped, other}) == 3
