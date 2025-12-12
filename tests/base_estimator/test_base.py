import jax
import jax.numpy as jnp
import pytest

from flowgym.common import base as base_mod
from flowgym.common.base import Estimator, EstimatorTrainableState


class NonTrainableEstimator(Estimator):
    """Concrete estimator used for testing base behavior without training."""

    def __init__(self, preprocessing_steps=None):
        super().__init__(preprocessing_steps=preprocessing_steps)
        self.calls: list[dict] = []

    def _estimate(
        self,
        images: jnp.ndarray,
        state,
        trainable_state,
        extras: dict,
    ):
        """Simple deterministic estimate, tracking calls for assertions."""
        self.calls.append(
            {
                "images": images,
                "state": state,
                "trainable_state": trainable_state,
                "extras": extras,
            }
        )
        B, _, _ = images.shape
        estimates = jnp.full((B, 2), 1.0, dtype=images.dtype)
        metrics = {"mean_image": images.mean()}

        # Produce new "extras" slices with shape (B, 1, *shape)
        new_extras = {}
        for name, val in extras.items():
            # val has shape (B, T, *shape); append last step value broadcasted
            new_extras[name] = jnp.zeros((B, 1) + val.shape[2:], dtype=val.dtype)

        return estimates, new_extras, metrics


class TrainableEstimator(NonTrainableEstimator):
    """Concrete estimator that supports training."""

    def create_train_step(self):
        """Return a dummy training step function."""

        def train_step(state, batch):
            return "trained", state, batch

        return train_step


# ---------------------------------------------------------------------------
# __init__ / preprocessing_steps behavior
# ---------------------------------------------------------------------------


def test_init_with_no_preprocessing(monkeypatch):
    # Arrange
    validate_calls = []
    apply_calls = []

    def fake_validate_params(name, **params):
        validate_calls.append((name, params))

    def fake_apply_preprocessing(images, **kwargs):
        apply_calls.append(kwargs)
        return images

    monkeypatch.setattr(base_mod.estimator, "validate_params", fake_validate_params)
    monkeypatch.setattr(
        base_mod.estimator, "apply_preprocessing", fake_apply_preprocessing
    )

    # Act
    estimator = NonTrainableEstimator(preprocessing_steps=None)

    # Assert
    assert estimator.preprocessing_steps == []
    assert validate_calls == []
    assert apply_calls == []


def test_init_raises_on_non_dict_preprocessing_step(monkeypatch):
    # Arrange
    monkeypatch.setattr(base_mod.estimator, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(
        base_mod.estimator, "apply_preprocessing", lambda images, **kwargs: images
    )

    # Act / Assert
    with pytest.raises(ValueError, match="must be a dictionary"):
        NonTrainableEstimator(preprocessing_steps=["not-a-dict"])  # type: ignore


def test_init_raises_on_missing_name_key(monkeypatch):
    # Arrange
    monkeypatch.setattr(base_mod.estimator, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(
        base_mod.estimator, "apply_preprocessing", lambda images, **kwargs: images
    )

    bad_step = {"sigma": 1.0}

    # Act / Assert
    with pytest.raises(ValueError, match="must have a 'name' key"):
        NonTrainableEstimator(preprocessing_steps=[bad_step])


def test_init_valid_preprocessing_builds_partials_and_validates(monkeypatch):
    # Arrange
    validate_calls = []
    apply_calls = []

    def fake_validate_params(name, **params):
        validate_calls.append((name, params))

    def fake_apply_preprocessing(images, **kwargs):
        apply_calls.append(kwargs)
        # Return a clearly modified tensor to ensure it is used.
        return images + 1.0

    monkeypatch.setattr(base_mod.estimator, "validate_params", fake_validate_params)
    monkeypatch.setattr(
        base_mod.estimator, "apply_preprocessing", fake_apply_preprocessing
    )

    step_cfg = {"name": "normalize", "alpha": 0.1, "beta": 0.2}
    estimator = NonTrainableEstimator(preprocessing_steps=[step_cfg])

    # Act
    images = jnp.zeros((2, 4, 4))
    # stored step is a functools.partial around fake_apply_preprocessing
    (step,) = estimator.preprocessing_steps
    out = step(images)

    # Assert
    assert validate_calls == [("normalize", {"alpha": 0.1, "beta": 0.2})]
    assert apply_calls == [{"name": "normalize", "alpha": 0.1, "beta": 0.2}]
    assert jnp.array_equal(out, images + 1.0)


# ---------------------------------------------------------------------------
# create_state behavior
# ---------------------------------------------------------------------------


def test_create_state_valid_shapes_and_tiling_no_extras_rng_none():
    # Arrange
    B, H, W = 3, 8, 9
    images = jnp.ones((B, H, W))
    estimates = jnp.full((B, 2), 5.0)
    estimator = NonTrainableEstimator()

    # Act
    history = estimator.create_state(
        images,
        estimates,
        image_history_size=4,
        estimate_history_size=2,
        extras={
            "reward": {
                "length": 3,
                "shape": (),
                "dtype": jnp.float32,
                "init": "zeros",
            }
        },
        rng=None,
    )

    # Assert
    assert history["images"].shape == (B, 4, H, W)
    assert history["estimates"].shape == (B, 2, 2)
    # All history slots are initialized with the provided images/estimates.
    assert jnp.all(history["images"] == images[:, None, ...])
    assert jnp.all(history["estimates"] == estimates[:, None, :])
    # Extras
    assert "reward" in history
    assert history["reward"].shape == (B, 3)
    assert history["reward"].dtype == jnp.float32
    # No RNG keys when rng=None.
    assert "keys" not in history


def test_create_state_estimate_history_size_defaults_to_image_history_size():
    # Arrange
    B, H, W = 2, 5, 6
    images = jnp.zeros((B, H, W))
    estimates = jnp.zeros((B, 3))
    estimator = NonTrainableEstimator()

    # Act
    history = estimator.create_state(
        images,
        estimates,
        image_history_size=5,
        estimate_history_size=None,
        rng=None,
    )

    # Assert
    assert history["images"].shape == (B, 5, H, W)
    assert history["estimates"].shape == (B, 5, 3)


def test_create_state_with_rng_int_adds_per_batch_keys():
    # Arrange
    B, H, W = 2, 4, 4
    images = jnp.zeros((B, H, W))
    estimates = jnp.zeros((B, 2))
    estimator = NonTrainableEstimator()

    # Act
    history = estimator.create_state(
        images,
        estimates,
        image_history_size=3,
        estimate_history_size=3,
        rng=123,
    )

    # Assert
    assert "keys" in history
    keys = history["keys"]
    # The implementation currently stores one key per batch element.
    assert keys.shape == (B, 1, 2)
    assert keys.dtype == jnp.uint32


def test_create_state_rng_int_and_prngkey_are_equivalent():
    # Arrange
    B, H, W = 2, 4, 4
    images = jnp.zeros((B, H, W))
    estimates = jnp.zeros((B, 2))
    estimator = NonTrainableEstimator()

    # Act
    history_from_int = estimator.create_state(
        images,
        estimates,
        image_history_size=2,
        estimate_history_size=2,
        rng=0,
    )
    history_from_key = estimator.create_state(
        images,
        estimates,
        image_history_size=2,
        estimate_history_size=2,
        rng=jax.random.PRNGKey(0),
    )

    # Assert
    assert jnp.array_equal(history_from_int["keys"], history_from_key["keys"])


@pytest.mark.parametrize(
    "images_shape, estimates_shape",
    [
        ((8, 8), (2, 3)),  # images ndim != 3
        ((2, 8, 8), (2,)),  # estimates ndim < 2
        ((2, 8, 8), (3, 4)),  # batch mismatch
    ],
)
def test_create_state_raises_on_invalid_shapes(images_shape, estimates_shape):
    # Arrange
    images = jnp.zeros(images_shape)
    estimates = jnp.zeros(estimates_shape)
    estimator = NonTrainableEstimator()

    # Act / Assert
    with pytest.raises(ValueError):
        estimator.create_state(
            images,
            estimates,
            image_history_size=2,
            estimate_history_size=2,
        )


@pytest.mark.parametrize(
    "rng",
    [
        "not-an-int-or-key",
        jnp.ones((4,)),  # wrong shaped key
    ],
)
def test_create_state_raises_on_invalid_rng_type(rng):
    # Arrange
    images = jnp.zeros((2, 4, 4))
    estimates = jnp.zeros((2, 3))
    estimator = NonTrainableEstimator()

    # Act / Assert
    with pytest.raises(TypeError):
        estimator.create_state(
            images,
            estimates,
            image_history_size=2,
            estimate_history_size=2,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# create_trainable_state / create_train_step behavior
# ---------------------------------------------------------------------------


def test_create_trainable_state_returns_expected_type():
    # Arrange
    estimator = NonTrainableEstimator()
    dummy_input = jnp.zeros((2, 4, 4))
    key = jax.random.PRNGKey(0)

    # Act
    trainable_state = estimator.create_trainable_state(dummy_input, key)

    # Assert
    assert isinstance(trainable_state, EstimatorTrainableState)


def test_base_create_train_step_raises_not_implemented():
    # Arrange
    estimator = NonTrainableEstimator()

    # Act / Assert
    with pytest.raises(NotImplementedError):
        estimator.create_train_step()


def test_trainable_estimator_create_train_step_returns_callable():
    # Arrange
    estimator = TrainableEstimator()

    # Act
    train_step = estimator.create_train_step()

    # Assert
    assert callable(train_step)
    out, state_out, batch_out = train_step({"state": 1}, {"batch": 2})
    assert out == "trained"
    assert state_out == {"state": 1}
    assert batch_out == {"batch": 2}


# ---------------------------------------------------------------------------
# __call__ behavior (preprocessing, extras, history update)
# ---------------------------------------------------------------------------


def test_call_applies_preprocessing_and_updates_history(monkeypatch):
    # Arrange
    apply_calls = []

    def fake_apply_preprocessing(images, **kwargs):
        apply_calls.append(kwargs)
        return images * 2.0

    # validate_params is required during __init__, but behavior is simple here.
    monkeypatch.setattr(base_mod.estimator, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(
        base_mod.estimator, "apply_preprocessing", fake_apply_preprocessing
    )

    step_cfg = {"name": "scale", "factor": 2.0}
    estimator = NonTrainableEstimator(preprocessing_steps=[step_cfg])

    B, H, W = 2, 4, 4
    images = jnp.ones((B, H, W))
    init_estimates = jnp.zeros((B, 2))

    extras_cfg = {
        "reward": {
            "length": 3,
            "shape": (),
            "dtype": jnp.float32,
            "init": "zeros",
        }
    }
    # Use rng=None so that no "keys" field is present; extras field ensures extras dict
    # is non-empty inside __call__.
    state = estimator.create_state(
        images,
        init_estimates,
        image_history_size=3,
        estimate_history_size=2,
        extras=extras_cfg,
        rng=None,
    )
    trainable_state = estimator.create_trainable_state(images, jax.random.PRNGKey(0))

    # Act
    new_state, metrics = estimator(images, state, trainable_state)

    # Assert: preprocessing
    assert len(apply_calls) == 2  # One call during __call__, one during init
    assert apply_calls[0]["name"] == "scale"
    assert apply_calls[0]["factor"] == 2.0

    preprocessed = images * 2.0
    last_call = estimator.calls[-1]
    assert jnp.array_equal(last_call["images"], preprocessed)

    # History images should have last frame equal to preprocessed images.
    assert new_state["images"].shape[0] == B
    assert jnp.array_equal(new_state["images"][:, -1, ...], preprocessed)

    # Estimates history last entry should be equal to the estimates returned by _estimate.
    expected_estimates = jnp.full((B, 2), 1.0)
    assert jnp.array_equal(new_state["estimates"][:, -1, :], expected_estimates)

    # Extras should be passed in and out as dict.
    assert isinstance(last_call["extras"], dict)
    assert "reward" in last_call["extras"]

    # Metrics returned by _estimate should be propagated.
    assert "mean_image" in metrics
    assert pytest.approx(float(metrics["mean_image"])) == float(preprocessed.mean())


def test_call_uses_state_without_preprocessing(monkeypatch):
    """Sanity check when no preprocessing_steps are defined."""
    # Arrange
    monkeypatch.setattr(base_mod.estimator, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(
        base_mod.estimator, "apply_preprocessing", lambda images, **kwargs: images
    )

    estimator = NonTrainableEstimator(preprocessing_steps=[])
    B, H, W = 2, 3, 3
    images = jnp.arange(B * H * W, dtype=jnp.float32).reshape(B, H, W)
    init_estimates = jnp.zeros((B, 2))

    extras_cfg = {
        "aux": {
            "length": 2,
            "shape": (1,),
            "dtype": jnp.float32,
            "init": "zeros",
        }
    }
    state = estimator.create_state(
        images,
        init_estimates,
        image_history_size=2,
        estimate_history_size=2,
        extras=extras_cfg,
        rng=None,
    )
    trainable_state = estimator.create_trainable_state(images, jax.random.PRNGKey(0))

    # Act
    new_state, metrics = estimator(images, state, trainable_state)

    # Assert
    last_call = estimator.calls[-1]
    # Images should be passed through unchanged (no preprocessing).
    assert jnp.array_equal(last_call["images"], images)
    assert jnp.array_equal(new_state["images"][:, -1, ...], images)

    # Estimates last frame is the constant estimates from _estimate.
    expected_estimates = jnp.full((B, 2), 1.0)
    assert jnp.array_equal(new_state["estimates"][:, -1, :], expected_estimates)

    # Metrics still present.
    assert "mean_image" in metrics
