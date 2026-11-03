"""Unit tests for flowgym.common.base.estimator.

Covers create_state shape/tiling, extras merging, RNG handling,
and validation of invalid input shapes and RNG types.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from flowgym.common.base import Estimator


class DummyEstimator(Estimator):
    """Minimal concrete subclass to allow instantiation."""

    def _estimate(self, *args, **kwargs):
        pass


class TypedEstimator(DummyEstimator):
    """Estimator with typed parameters, as configured from YAML."""

    def __init__(
        self,
        mode: Literal["fast", "slow"] = "fast",
        size: int = 1,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.size = size
        self.scale = scale


@pytest.fixture
def dummy_estimator():
    return DummyEstimator()


def test_create_state_shapes_basic(dummy_estimator):
    """create_state produces images/estimates histories with correct shapes."""
    B, H, W = 3, 16, 16
    images = jnp.ones((B, H, W))
    estimates = jnp.zeros((B, 2))
    history = dummy_estimator.create_state(
        images,
        estimates,
        image_history_size=4,
        estimate_history_size=2,
        rng=None,
    )

    assert "images" in history and "estimates" in history
    assert history["images"].shape == (B, 4, H, W)
    assert history["estimates"].shape == (B, 2, 2)
    assert jnp.allclose(history["images"][:, -1], images)
    assert jnp.allclose(history["estimates"][:, -1], estimates)


def test_create_state_with_extras(dummy_estimator):
    """Extras config fields are merged into the state with correct shapes."""
    B, H, W = 2, 8, 8
    images = jnp.zeros((B, H, W))
    estimates = jnp.ones((B, 4))
    extras = {
        "reward": {
            "length": 3,
            "shape": (),
            "dtype": jnp.float32,
            "init": "zeros",
        },
        "mask": {
            "length": 3,
            "shape": (1,),
            "dtype": jnp.int32,
            "init": "zeros",
        },
    }
    history = dummy_estimator.create_state(
        images,
        estimates,
        image_history_size=2,
        estimate_history_size=2,
        extras=extras,
        rng=None,
    )

    assert set(extras.keys()).issubset(history.keys())
    assert history["reward"].shape == (B, 3)
    assert history["mask"].dtype == jnp.int32


@pytest.mark.parametrize("rng_input", [None, 42, jax.random.PRNGKey(0)])
def test_rng_behavior(dummy_estimator, rng_input):
    """None rng omits keys; int or PRNGKey seeds add per-batch key arrays."""
    B, H, W = 4, 8, 8
    images = jnp.ones((B, H, W))
    estimates = jnp.zeros((B, 3))

    history = dummy_estimator.create_state(
        images,
        estimates,
        image_history_size=2,
        rng=rng_input,
    )

    if rng_input is None:
        # No RNG field should be added
        assert "keys" not in history
    else:
        # RNG field added with correct shape
        assert "keys" in history
        assert history["keys"].shape == (B, 1, 2)
        # Check that all keys are distinct
        keys_flat = history["keys"].reshape(-1)
        assert len(jnp.unique(keys_flat)) > 2


def test_invalid_shapes_raise(dummy_estimator):
    """Wrong image rank or mismatched batch size raises ValueError."""
    # Wrong image ndim
    bad_images = jnp.ones((4, 4))
    good_estimates = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match="must have shape"):
        dummy_estimator.create_state(
            bad_images, good_estimates, image_history_size=2
        )

    # Mismatched batch size
    good_images = jnp.ones((3, 4, 4))
    bad_estimates = jnp.zeros((2, 3))
    with pytest.raises(ValueError, match="Batch size mismatch"):
        dummy_estimator.create_state(
            good_images, bad_estimates, image_history_size=2
        )


def test_invalid_rng_type_raises(dummy_estimator):
    """An unsupported rng type raises TypeError in create_state."""
    images = jnp.ones((2, 4, 4))
    estimates = jnp.zeros((2, 1))
    bad_rng = "not_a_key"
    with pytest.raises(TypeError):
        dummy_estimator.create_state(
            images, estimates, image_history_size=2, rng=bad_rng
        )


def test_from_config_warns_on_unknown_keys(monkeypatch):
    """from_config warns about keys no __init__ takes, not caller keys."""
    from flowgym.common.base import estimator as estimator_mod

    warnings = []
    monkeypatch.setattr(estimator_mod.logger, "warning", warnings.append)
    DummyEstimator.from_config(
        {"jit": True, "rng": 0, "image_shape": None, "no_such_param": 1}
    )
    assert len(warnings) == 1
    assert "no_such_param" in warnings[0]


def test_from_config_accepts_values_matching_init_hints():
    """Valid values pass, including a YAML int for a float parameter."""
    est = TypedEstimator.from_config({"mode": "slow", "size": 3, "scale": 2})
    assert (est.mode, est.size, est.scale) == ("slow", 3, 2)


@pytest.mark.parametrize(
    "key, value",
    [("mode", "medium"), ("size", "3"), ("size", 2.5), ("scale", "1.0")],
)
def test_from_config_rejects_values_not_matching_init_hints(key, value):
    """A config value of the wrong type names the estimator and the key."""
    with pytest.raises(ValueError, match=rf"TypedEstimator.*{key}"):
        TypedEstimator.from_config({key: value})
