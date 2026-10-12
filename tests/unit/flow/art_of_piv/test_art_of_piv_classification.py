"""Tests for ART of PIV classification training mode."""

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.types import NNEstimatorTrainableState, SupervisedExperience


# ──────────────────────────────────────────────────────────────────────────────
# Mock classes for testing (reused and adapted from regression tests)
# ──────────────────────────────────────────────────────────────────────────────
class MockSubModel:
    """Mock sub-model that returns a fixed EPE."""

    def __init__(self, name, epe_value=0.1):
        self.name = name
        self.epe_value = epe_value

    def get_cache_id_suffix(self, trainable_state):
        return f"_mock_{self.name}"

    def compute_cache_miss(self, batch, miss_idxs, **kwargs):
        return {
            "epe": np.ones(len(miss_idxs), dtype=np.float32) * self.epe_value
        }

    def __call__(self, input_tuple):
        curr, _state = input_tuple
        B = curr.shape[0]
        return {
            "estimates": jnp.full((B, 1, 64, 64, 2), float(self.epe_value))
        }, {}


class MockArtOfPIVForClassification(ArtOfPIVEstimator):
    """Mock ART of PIV specifically for testing supervised_train_step."""

    def __init__(self, sub_models, training_mode="classification"):
        # Intentionally bypass ArtOfPIVEstimator.__init__ in this test double;
        # we wire only the attributes needed by supervised train-step tests.
        self.sub_models = tuple(sub_models)
        self.num_estimators = len(sub_models)
        self.training_mode = training_mode
        self.use_temporal_propagation = False
        self.use_last_estimate = False
        self.schedule = lambda step: 0.0  # Greedy
        self.explore = lambda q, r, k: jnp.argmax(q, axis=-1).reshape(
            (q.shape[0],)
        )
        self.target_transform = lambda x: x
        self.estimator_configs = [
            {"name": m.name, "config": {}} for m in sub_models
        ]
        self.estimator_fns = tuple([m.__call__ for m in sub_models])
        self.sub_model_states = (None,) * self.num_estimators

        # In classification mode, we expect a loss that takes logits and labels
        # For testing, we'll use a simple mock cross-entropy like loss
        def mock_ce_loss(logits, labels):
            # logits: (B, N), labels: (B,)
            one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
            return -jnp.mean(
                jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
            )

        self.loss_fn = mock_ce_loss
        self.optimizer_config = {"name": "sgd", "learning_rate": 0.1}
        self.preprocess = lambda x: x
        self.preprocessing_steps = []

        # Add mock model for compatibility with refactored _run_router
        class MockModel:
            def __init__(self, num_estimators):
                self.num_estimators = num_estimators

            def apply(self, variables, x):
                return jnp.ones((x.shape[0], self.num_estimators))

        self.model = MockModel(self.num_estimators)

    def q_estimate(self, prev, curr, trainable_state, flow=None):
        return trainable_state.params["router"]


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_training_mode_classification():
    """Test that classification mode trains to predict the best arm."""
    model1 = MockSubModel("m1", epe_value=0.1)  # Best
    model2 = MockSubModel("m2", epe_value=0.2)
    estimator = MockArtOfPIVForClassification(
        [model1, model2], training_mode="classification"
    )

    # Initial logits: slightly favoring m2 (index 1)
    params = freeze({"router": jnp.array([[0.0, 1.0]])})

    import optax

    tx = optax.sgd(0.1)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((1, 1))},
    )

    experience = SupervisedExperience(
        obs=(jnp.zeros((1, 64, 64)), jnp.zeros((1, 64, 64))),
        state={
            "images": jnp.zeros((1, 1, 64, 64)),
            "estimates": jnp.zeros((1, 1, 64, 64, 2)),
            "keys": jax.random.PRNGKey(0)[None, None, :],
            "choices": jnp.zeros((1, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((1, 64, 64, 2)),
        cache_payload={"epe_all": np.array([[0.1, 0.2]], dtype=np.float32)},
    )

    train_step = estimator.create_supervised_train_step()
    _loss, new_state, _metrics = train_step(trainable_state, experience)

    # Logits for m1 (index 0) should have increased because it's the best arm
    router_params = new_state.params["router"]
    assert isinstance(router_params, jnp.ndarray), (
        f"router_params is not a jnp.ndarray, got {type(router_params)}"
    )
    assert router_params.shape == (1, 2), (
        "router_params has wrong shape, expected (1, 2), "
        f"got {router_params.shape}"
    )
    assert router_params[0, 0] > 0.0, (
        "router_params[0, 0] should be greater than 0.0, "
        f"got {router_params[0, 0]}"
    )
    assert router_params[0, 1] < 1.0, (
        "router_params[0, 1] should be less than 1.0, "
        f"got {router_params[0, 1]}"
    )


def test_training_mode_classification_no_cache():
    """Test classification mode without relying on cached oracle labels."""
    model1 = MockSubModel("m1", epe_value=0.1)  # Best
    model2 = MockSubModel("m2", epe_value=0.2)
    estimator = MockArtOfPIVForClassification(
        [model1, model2], training_mode="classification"
    )

    # Initial logits: slightly favoring m2 (index 1)
    params = freeze({"router": jnp.array([[0.0, 1.0]])})

    import optax

    tx = optax.sgd(0.1)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((1, 1))},
    )

    experience = SupervisedExperience(
        obs=(jnp.zeros((1, 64, 64)), jnp.zeros((1, 64, 64))),
        state={
            "images": jnp.zeros((1, 1, 64, 64)),
            "estimates": jnp.zeros((1, 1, 64, 64, 2)),
            "keys": jax.random.PRNGKey(0)[None, None, :],
            "choices": jnp.zeros((1, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((1, 64, 64, 2)),
        cache_payload=None,  # NO CACHE
    )

    train_step = estimator.create_supervised_train_step()
    _loss, new_state, _metrics = train_step(trainable_state, experience)

    # Logits for m1 (index 0) should have increased because it's the best arm
    router_params = new_state.params["router"]

    assert isinstance(router_params, jnp.ndarray), (
        f"router_params is not a jnp.ndarray, got {type(router_params)}"
    )
    assert router_params.shape == (1, 2), (
        "router_params has wrong shape, expected (1, 2), "
        f"got {router_params.shape}"
    )
    assert router_params[0, 0] > 0.0, (
        "router_params[0, 0] should be greater than 0.0, "
        f"got {router_params[0, 0]}"
    )
    assert router_params[0, 1] < 1.0, (
        "router_params[0, 1] should be less than 1.0, "
        f"got {router_params[0, 1]}"
    )
