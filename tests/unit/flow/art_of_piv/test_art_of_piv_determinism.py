"""Tests for ART of PIV validation determinism (zero exploration)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import freeze

from flowgym.flow.art_of_piv import inference as art_of_piv_inference
from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.types import CachePayload, NNEstimatorTrainableState


def test_validation_determinism():
    """Verify deterministic validation behavior across different RNG keys."""
    # 1. Setup a minimal ArtOfPIVEstimator with high exploration
    dummy_yaml = os.path.join(
        os.path.dirname(__file__), "dummy_estimators.yaml"
    )

    config = {
        "router_config": {
            "features": [16],
            "kernel_sizes": [3],
            "strides": [1],
        },
        "estimators_list_path": dummy_yaml,
        "exploration_config": {
            "name": "epsilon_greedy",
            "schedule": {"name": "constant", "value": 1.0},  # 100% exploration
        },
        "optimizer_config": {"name": "sgd", "learning_rate": 0.01},
        "initialize_estimators": False,
    }

    # Mocking q_estimate and run_subestimators to observe choices
    class DeterminismMockEstimator(ArtOfPIVEstimator):
        def q_estimate(self, prev, curr, trainable_state, flow=None):
            # All actions look equally good (all ones)
            return jnp.ones((prev.shape[0], 10))

        def _estimate(self, images, state, trainable_state, extras):
            """Override to use mock run_subestimators for testing."""

            # Create a mock run_subestimators for testing
            def mock_run_subestimators(
                state,
                trainable_state,
                curr,
                choices,
                extras,
                qs,
                exploration_rate,
                **kwargs,
            ):
                B = choices.shape[0]
                flows = jnp.zeros((B, 32, 32, 2))
                out_extras = {"choices": choices}
                metrics = {
                    "choices": choices,
                    "exploration_rate": exploration_rate,
                }
                return flows, out_extras, metrics

            return art_of_piv_inference.estimate(
                images,
                state,
                trainable_state,
                extras,
                use_temporal_propagation=self.use_temporal_propagation,
                use_last_estimate=self.use_last_estimate,
                run_router_fn=self._run_router,
                run_subestimators_fn=mock_run_subestimators,
            )

    model = DeterminismMockEstimator(**config)

    # Initialize state
    B, H, W = 8, 32, 32
    params = freeze({})
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=optax.identity(),
        extras={"steps": jnp.zeros((B, 1), dtype=jnp.int32)},
    )

    # Create two states with DIFFERENT keys
    obs = jnp.zeros((B, H, W))
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(123)

    state1 = model.create_state(
        obs, jnp.zeros((B, H, W, 2)), image_history_size=1, rng=key1
    )
    state2 = model.create_state(
        obs, jnp.zeros((B, H, W, 2)), image_history_size=1, rng=key2
    )

    # 2. RUN WITH is_training=True (Exploration should happen)
    _, metrics1_train = model(
        obs,
        state1,
        trainable_state,
        cache_payload=CachePayload(is_training=True),
    )
    _, metrics2_train = model(
        obs,
        state2,
        trainable_state,
        cache_payload=CachePayload(is_training=True),
    )

    choices1_train = np.array(metrics1_train["choices"])
    choices2_train = np.array(metrics2_train["choices"])

    # Different keys SHOULD lead to different choices in training mode
    assert not np.array_equal(choices1_train, choices2_train), (
        f"Choices were identical in training ({choices1_train})"
    )

    # 3. RUN WITH is_training=False (Validation mode - Zero Exploration)
    _, metrics1_val = model(
        obs,
        state1,
        trainable_state,
        cache_payload=CachePayload(is_training=False),
    )
    _, metrics2_val = model(
        obs,
        state2,
        trainable_state,
        cache_payload=CachePayload(is_training=False),
    )

    choices1_val = np.array(metrics1_val["choices"])
    choices2_val = np.array(metrics2_val["choices"])

    # Different keys SHOULD lead to IDENTICAL greedy choices in validation.
    np.testing.assert_array_equal(choices1_val, choices2_val)
    # They should all be 0 because all Q-values are 1.0 and argmax picks
    # the first index.
    np.testing.assert_array_equal(choices1_val, np.zeros(B))
