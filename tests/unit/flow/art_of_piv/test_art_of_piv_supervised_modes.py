"""Tests for ART of PIV supervised training modes."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.core.frozen_dict import freeze

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.types import NNEstimatorTrainableState, SupervisedExperience


# ──────────────────────────────────────────────────────────────────────────────
# Mock classes for testing
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

    def __call__(self, *args):
        # Support both legacy tuple-call and current estimator callable
        # signature used by inference.run_subestimators.
        if len(args) == 1:
            curr, _state = args[0]
        else:
            curr = args[0]
        B = curr.shape[0]
        # Return state with a dummy estimate
        return {
            "estimates": jnp.full((B, 1, 64, 64, 2), float(self.epe_value))
        }, {}


class MockArtOfPIVForTraining(ArtOfPIVEstimator):
    """Mock ART of PIV specifically for testing supervised_train_step."""

    def __init__(self, sub_models, training_mode="bandits"):
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
        self.loss_fn = lambda q, t: jnp.mean((q - t) ** 2)
        self.optimizer_config = {"name": "sgd", "learning_rate": 0.1}
        self.preprocess = lambda x: x
        self.preprocessing_steps: list[Any] = []

        # Add mock model for compatibility with refactored _run_router
        class MockModel:
            def __init__(self, num_estimators):
                self.num_estimators = num_estimators

            def apply(self, variables, x):
                return jnp.ones((x.shape[0], self.num_estimators))

        self.model = MockModel(self.num_estimators)

    def q_estimate(self, prev, curr, trainable_state, flow=None):
        # Router output is params["router"]
        return trainable_state.params["router"]

    def _run_router(
        self, prev, curr, state, trainable_state, flow=None, train=True
    ):
        """Override _run_router to use the mocked q_estimate."""
        # Get Q-values from the overridden q_estimate
        qs = self.q_estimate(prev, curr, trainable_state, flow=flow)

        # Use exploration from schedule
        B = prev.shape[0]
        exploration_val = jnp.full((B,), self.schedule(0))

        # Sample actions
        key = jax.random.PRNGKey(0)
        actions = self.explore(qs, exploration_val, key)

        # Generate dummy estimate keys
        estimate_keys = jax.random.split(key, B)

        return actions, qs, exploration_val, estimate_keys


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_training_mode_bandits_one_action():
    """Test that bandits mode only trains on the selected action."""
    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.2)
    estimator = MockArtOfPIVForTraining(
        [model1, model2], training_mode="bandits"
    )

    # Initial Q-values (params)
    params = freeze(
        {"router": jnp.array([[1.0, 1.0]])}
    )  # B=1, num_estimators=2

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
        cache_payload=None,
    )

    train_step = estimator.create_supervised_train_step()
    _loss, new_state, metrics = train_step(trainable_state, experience)

    # selected action was 0 (greedy break ties)
    assert metrics["choices/m1"] == 1, "choices/m1 is not 1"

    # Q[0] should have decreased, Q[1] should remain 1.0
    router_params = new_state.params["router"]
    # grad = 2*(1.0 - (-0.1)) = 2.2. Wait, loss is mean((q-t)**2).
    # Since B=1, mean is identity? No, it's (q-t)**2.
    # grad = 2.2. Update = 1.0 - 0.1 * 2.2 = 0.78.

    assert isinstance(router_params, jnp.ndarray), (
        f"router_params is not a jnp.ndarray, got {type(router_params)}"
    )
    assert router_params.shape == (1, 2), (
        "router_params has wrong shape, expected (1, 2), "
        f"got {router_params.shape}"
    )
    assert router_params[0, 0] < 1.0, (
        "router_params[0, 0] should be less than 1.0, "
        f"got {router_params[0, 0]}"
    )
    assert router_params[0, 1] == 1.0, (
        f"router_params[0, 1] should be 1.0, got {router_params[0, 1]}"
    )


def test_training_mode_regression_all_actions():
    """Test that regression mode trains on all actions."""
    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.2)
    estimator = MockArtOfPIVForTraining(
        [model1, model2], training_mode="regression"
    )

    params = freeze({"router": jnp.array([[1.0, 1.0]])})

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

    # Both Q-values should have changed and they should be DIFFERENT
    # because targets are -0.1 and -0.2
    router_params = new_state.params["router"]

    assert isinstance(router_params, jnp.ndarray), (
        f"router_params is not a jnp.ndarray, got {type(router_params)}"
    )
    assert router_params.shape == (1, 2), (
        "router_params has wrong shape, expected (1, 2), "
        f"got {router_params.shape}"
    )
    assert router_params[0, 0] < 1.0, (
        "router_params[0, 0] should be less than 1.0, "
        f"got {router_params[0, 0]}"
    )
    assert router_params[0, 1] < 1.0, (
        "router_params[0, 1] should be less than 1.0, "
        f"got {router_params[0, 1]}"
    )
    assert not jnp.allclose(router_params[0, 0], router_params[0, 1]), (
        "router_params[0, 0] and router_params[0, 1] are equal"
    )


def test_regression_convergence_over_steps():
    """Test regression convergence across repeated optimization steps.

    This is a critical test that verifies the Q-values actually learn to predict
    the transformed EPE values correctly. If this test fails, there is likely a
    bug in the gradient computation or loss function.
    """
    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.5)
    model3 = MockSubModel("m3", epe_value=0.3)
    estimator = MockArtOfPIVForTraining(
        [model1, model2, model3], training_mode="regression"
    )

    # Initial Q-values - all equal, so the router doesn't know which is best
    initial_q = jnp.array([[0.0, 0.0, 0.0]])
    params = freeze({"router": initial_q})

    # Target values: -epe (reward = -epe)
    # So targets are: [-0.1, -0.5, -0.3]
    epe_all = np.array([[0.1, 0.5, 0.3]], dtype=np.float32)
    expected_targets = -epe_all  # [-0.1, -0.5, -0.3]

    tx = optax.sgd(0.5)  # Larger learning rate for faster convergence in test
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
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Run multiple training steps
    losses = []
    for _ in range(50):
        loss, trainable_state, _metrics = train_step(
            trainable_state, experience
        )
        losses.append(float(loss))

    final_q = jnp.asarray(trainable_state.params["router"])

    # Check 1: Loss should decrease
    assert losses[-1] < losses[0], (
        "Loss did not decrease: "
        f"initial={losses[0]:.6f}, final={losses[-1]:.6f}"
    )

    # Check 2: Q-values should be close to targets
    # targets are [-0.1, -0.5, -0.3]
    assert jnp.allclose(final_q, expected_targets, atol=0.05), (
        f"Q-values did not converge to targets.\n"
        f"Expected: {expected_targets}\n"
        f"Got: {final_q}"
    )

    # Check 3: The ranking should be correct (higher Q = better = lower EPE)
    # model1 (epe=0.1) should have highest Q (-0.1)
    # model3 (epe=0.3) should have middle Q (-0.3)
    # model2 (epe=0.5) should have lowest Q (-0.5)
    assert jnp.argmax(final_q) == 0, (
        f"Best model should be index 0 (lowest EPE), got {jnp.argmax(final_q)}"
    )


def test_regression_learns_correct_ranking():
    """Test that regression learns to rank estimators correctly by EPE.

    This test uses a more realistic scenario with multiple batch samples
    and verifies that the learned Q-values preserve the correct ordering.
    """
    # 5 estimators with different EPE values
    epe_values = [0.1, 0.4, 0.2, 0.3, 0.5]
    sub_models = [
        MockSubModel(f"m{i}", epe_value=epe) for i, epe in enumerate(epe_values)
    ]
    estimator = MockArtOfPIVForTraining(sub_models, training_mode="regression")

    # Initial Q-values - random initialization
    key = jax.random.PRNGKey(42)
    initial_q = jax.random.uniform(key, (1, 5), minval=-1.0, maxval=1.0)
    params = freeze({"router": initial_q})

    # Create batch with EPE values
    B = 1
    epe_all = np.array([epe_values], dtype=np.float32)  # Shape (1, 5)

    tx = optax.adam(0.1)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((B, 1))},
    )

    experience = SupervisedExperience(
        obs=(jnp.zeros((B, 64, 64)), jnp.zeros((B, 64, 64))),
        state={
            "images": jnp.zeros((B, 1, 64, 64)),
            "estimates": jnp.zeros((B, 1, 64, 64, 2)),
            "keys": jax.random.PRNGKey(0)[None, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, 64, 64, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Train for many steps
    for _ in range(100):
        _, trainable_state, _ = train_step(trainable_state, experience)

    final_q = jnp.asarray(trainable_state.params["router"])[0]  # Shape (5,)

    # The ranking of Q-values should be inverse of EPE ranking
    # Lower EPE = Higher Q (since reward = -EPE)
    # EPE ranking (low to high): [0, 2, 3, 1, 4]
    # corresponding to [0.1, 0.2, 0.3, 0.4, 0.5].
    # Q ranking (high to low) should be: [0, 2, 3, 1, 4]
    expected_q_order = np.argsort(epe_values)  # [0, 2, 3, 1, 4]
    actual_q_order = np.argsort(-np.array(final_q))  # Descending order of Q

    assert np.array_equal(expected_q_order, actual_q_order), (
        f"Q-value ranking does not match EPE ranking.\n"
        f"EPE values: {epe_values}\n"
        f"Expected Q order (best to worst): {expected_q_order}\n"
        f"Actual Q order (best to worst): {actual_q_order}\n"
        f"Final Q-values: {final_q}"
    )


def test_regression_gradient_direction():
    """Test that gradients point in the correct direction for regression.

    Verifies that when Q > target, gradient pushes Q down, and vice versa.
    """
    model1 = MockSubModel("m1", epe_value=0.5)  # target = -0.5
    model2 = MockSubModel("m2", epe_value=0.5)  # target = -0.5
    estimator = MockArtOfPIVForTraining(
        [model1, model2], training_mode="regression"
    )

    # Q[0] = 1.0 (too high, target = -0.5), should decrease
    # Q[1] = -1.0 (too low, target = -0.5), should increase
    params = freeze({"router": jnp.array([[1.0, -1.0]])})

    tx = optax.sgd(0.1)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((1, 1))},
    )

    epe_all = np.array([[0.5, 0.5]], dtype=np.float32)
    experience = SupervisedExperience(
        obs=(jnp.zeros((1, 64, 64)), jnp.zeros((1, 64, 64))),
        state={
            "images": jnp.zeros((1, 1, 64, 64)),
            "estimates": jnp.zeros((1, 1, 64, 64, 2)),
            "keys": jax.random.PRNGKey(0)[None, None, :],
            "choices": jnp.zeros((1, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((1, 64, 64, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()
    _, new_state, _ = train_step(trainable_state, experience)

    new_q = jnp.asarray(new_state.params["router"])
    old_q = jnp.asarray(params["router"])

    # Q[0] was 1.0, target is -0.5 -> should decrease
    assert new_q[0, 0] < old_q[0, 0], (
        "Q[0] should decrease (was too high). "
        f"Old: {old_q[0, 0]}, New: {new_q[0, 0]}"
    )

    # Q[1] was -1.0, target is -0.5 -> should increase
    assert new_q[0, 1] > old_q[0, 1], (
        "Q[1] should increase (was too low). "
        f"Old: {old_q[0, 1]}, New: {new_q[0, 1]}"
    )


def test_regression_with_target_transform():
    """Test that target transforms are applied correctly in regression mode.

    When a target transform is used (e.g., log1p), the Q-values should learn
    to predict the transformed values, not the raw EPE.
    """
    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=1.0)
    estimator = MockArtOfPIVForTraining(
        [model1, model2], training_mode="regression"
    )

    # Apply log1p transform
    estimator.target_transform = jnp.log1p

    params = freeze({"router": jnp.array([[0.0, 0.0]])})

    tx = optax.sgd(0.5)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((1, 1))},
    )

    epe_all = np.array([[0.1, 1.0]], dtype=np.float32)
    # Targets after transform:
    # -log1p([0.1, 1.0]) = -[0.0953, 0.6931] ~= [-0.095, -0.693].
    expected_targets = -jnp.log1p(jnp.array([0.1, 1.0]))

    experience = SupervisedExperience(
        obs=(jnp.zeros((1, 64, 64)), jnp.zeros((1, 64, 64))),
        state={
            "images": jnp.zeros((1, 1, 64, 64)),
            "estimates": jnp.zeros((1, 1, 64, 64, 2)),
            "keys": jax.random.PRNGKey(0)[None, None, :],
            "choices": jnp.zeros((1, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((1, 64, 64, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Train to convergence
    for _ in range(100):
        _, trainable_state, _ = train_step(trainable_state, experience)

    final_q = jnp.asarray(trainable_state.params["router"])[0]

    # Q-values should match transformed targets
    assert jnp.allclose(final_q, expected_targets, atol=0.05), (
        f"Q-values did not converge to transformed targets.\n"
        f"Expected: {expected_targets}\n"
        f"Got: {final_q}"
    )


def test_regression_batch_independence():
    """Test regression with batches that contain different EPE profiles.

    Each sample in the batch should contribute independently to the loss.
    """
    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.2)

    # Use a mock that returns batch-wise Q-values
    # For this test, we override to return fixed per-batch Q-values
    class BatchAwareEstimator(MockArtOfPIVForTraining):
        def q_estimate(self, prev, curr, trainable_state, flow=None):
            # Return the params repeated for batch size
            B = prev.shape[0]
            return jnp.tile(trainable_state.params["router"], (B, 1))

    estimator = BatchAwareEstimator(
        [model1, model2], training_mode="regression"
    )

    params = freeze({"router": jnp.array([[0.0, 0.0]])})

    tx = optax.sgd(0.1)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=lambda *args: None,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((2, 1))},
    )

    # Batch of 2 with different EPE profiles
    B = 2
    epe_all = np.array(
        [
            [0.1, 0.5],  # Sample 0: m1 is best
            [0.5, 0.1],  # Sample 1: m2 is best
        ],
        dtype=np.float32,
    )

    experience = SupervisedExperience(
        obs=(jnp.zeros((B, 64, 64)), jnp.zeros((B, 64, 64))),
        state={
            "images": jnp.zeros((B, 1, 64, 64)),
            "estimates": jnp.zeros((B, 1, 64, 64, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, 64, 64, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Since both models have equal average EPE across batch (0.3 each),
    # after training, Q-values should converge to the mean target = -0.3
    for _ in range(100):
        _, trainable_state, _ = train_step(trainable_state, experience)

    final_q = jnp.asarray(trainable_state.params["router"])[0]

    # Both Q-values should be approximately equal to -0.3
    expected = -0.3
    assert jnp.allclose(final_q[0], expected, atol=0.05), (
        f"Q[0] should be ~{expected}, got {final_q[0]}"
    )
    assert jnp.allclose(final_q[1], expected, atol=0.05), (
        f"Q[1] should be ~{expected}, got {final_q[1]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests using actual CNN model
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_regression_with_real_cnn_model():
    """Test regression training with the actual CNN model architecture.

    This test verifies that the real CNN-based Q-estimator can learn to predict
    EPE values through gradient descent. If this test fails but the mock tests
    pass, the issue is likely in how the CNN processes inputs or how gradients
    flow through the model.
    """
    from flowgym.nn.cnn import CNNQEstimatorModel

    num_estimators = 3
    H, W = 32, 32  # Smaller images for faster test
    B = 4

    # Create a simple CNN model
    model = CNNQEstimatorModel(
        features_list=(8, 16),
        norm_fn="none",
        kernel_sizes=(3, 3),
        strides=(2, 2),
        num_q_values=num_estimators,
        mlp_hidden_dim=32,
        postprocess=(),
    )

    # Initialize model parameters
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, H, W, 2))  # (B, H, W, channels)
    params = model.init(key, dummy_input)["params"]

    tx = optax.adam(1e-3)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((B, 1))},
    )

    # Create fixed EPE targets - different for each estimator
    epe_values = np.array([0.1, 0.5, 0.3])  # m0 is best, m1 is worst
    epe_all = np.tile(epe_values, (B, 1)).astype(np.float32)  # (B, 3)
    targets = -epe_all  # reward = -EPE

    # Create random but fixed input images
    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    def loss_fn(params):
        input_stack = jnp.stack([images, images], axis=-1)  # (B, H, W, 2)
        q_values = jnp.asarray(model.apply({"params": params}, input_stack))
        # L2 loss between Q-values and targets
        return jnp.mean((q_values - targets) ** 2)

    initial_loss = float(loss_fn(trainable_state.params))

    # Training loop
    losses = [initial_loss]
    for _ in range(200):
        loss, grads = jax.value_and_grad(loss_fn)(trainable_state.params)
        trainable_state = trainable_state.apply_gradients(grads=grads)
        losses.append(float(loss))

    final_loss = losses[-1]

    # Get final Q-values
    input_stack = jnp.stack([images, images], axis=-1)
    final_q = jnp.asarray(
        model.apply({"params": trainable_state.params}, input_stack)
    )

    # Check 1: Loss should decrease significantly
    assert final_loss < initial_loss * 0.1, (
        f"Loss did not decrease enough. Initial: {initial_loss:.4f}, "
        f"Final: {final_loss:.4f}, Ratio: {final_loss / initial_loss:.4f}"
    )

    # Check 2: Q-value ranking should be correct (on average across batch)
    mean_q = jnp.mean(final_q, axis=0)
    # Best estimator (lowest EPE = 0.1) should have highest Q
    assert jnp.argmax(mean_q) == 0, (
        f"Best estimator should be index 0, got {jnp.argmax(mean_q)}. "
        f"Mean Q: {mean_q}, EPE: {epe_values}"
    )


@pytest.mark.slow
def test_regression_with_varying_inputs():
    """Test regression when different inputs prefer different estimators.

    This is a crucial test for real-world scenarios where different image pairs
    may have different optimal estimators. The model should learn conditional
    Q-values based on the input.
    """
    from flowgym.nn.cnn import CNNQEstimatorModel

    num_estimators = 2
    H, W = 32, 32
    B = 8

    model = CNNQEstimatorModel(
        features_list=(16, 32),
        norm_fn="none",
        kernel_sizes=(3, 3),
        strides=(2, 2),
        num_q_values=num_estimators,
        mlp_hidden_dim=64,
        postprocess=(),
    )

    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, H, W, 2))
    params = model.init(key, dummy_input)["params"]

    tx = optax.adam(1e-3)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((B, 1))},
    )

    # Create two distinct patterns of images
    key, k1, k2 = jax.random.split(key, 3)

    # Pattern A: Low intensity images -> estimator 0 is better
    images_a = jax.random.uniform(k1, (B // 2, H, W), minval=0, maxval=100)
    epe_a = np.array([[0.1, 0.5]] * (B // 2), dtype=np.float32)

    # Pattern B: High intensity images -> estimator 1 is better
    images_b = jax.random.uniform(k2, (B // 2, H, W), minval=155, maxval=255)
    epe_b = np.array([[0.5, 0.1]] * (B // 2), dtype=np.float32)

    # Combine into one batch
    all_images = jnp.concatenate([images_a, images_b], axis=0)
    all_epe = np.concatenate([epe_a, epe_b], axis=0)
    all_targets = -all_epe

    def loss_fn(params):
        input_stack = jnp.stack([all_images, all_images], axis=-1)
        q_values = jnp.asarray(model.apply({"params": params}, input_stack))
        return jnp.mean((q_values - all_targets) ** 2)

    initial_loss = float(loss_fn(trainable_state.params))

    # Train for longer since this is a harder task
    for _ in range(500):
        _loss, grads = jax.value_and_grad(loss_fn)(trainable_state.params)
        trainable_state = trainable_state.apply_gradients(grads=grads)

    final_loss = float(loss_fn(trainable_state.params))

    # Get final Q-values for each pattern
    input_a = jnp.stack([images_a, images_a], axis=-1)
    input_b = jnp.stack([images_b, images_b], axis=-1)
    q_a = jnp.asarray(model.apply({"params": trainable_state.params}, input_a))
    q_b = jnp.asarray(model.apply({"params": trainable_state.params}, input_b))

    # Check loss decreased
    assert final_loss < initial_loss * 0.5, (
        f"Loss did not decrease enough. Initial: {initial_loss:.4f}, "
        f"Final: {final_loss:.4f}"
    )

    # For pattern A images, estimator 0 should have higher average Q
    mean_q_a = jnp.mean(q_a, axis=0)
    assert mean_q_a[0] > mean_q_a[1], (
        f"For pattern A, estimator 0 should have higher Q. Got Q: {mean_q_a}"
    )

    # For pattern B images, estimator 1 should have higher average Q
    mean_q_b = jnp.mean(q_b, axis=0)
    assert mean_q_b[1] > mean_q_b[0], (
        f"For pattern B, estimator 1 should have higher Q. Got Q: {mean_q_b}"
    )


def test_regression_loss_gradient_magnitude():
    """Test that gradients in regression mode have reasonable magnitudes.

    If gradients are too small (vanishing) or too large (exploding), training
    will not converge properly. This test verifies gradient magnitudes are
    in a reasonable range.
    """
    from flowgym.nn.cnn import CNNQEstimatorModel

    num_estimators = 3
    H, W = 32, 32
    B = 2

    model = CNNQEstimatorModel(
        features_list=(8, 16),
        norm_fn="none",
        kernel_sizes=(3, 3),
        strides=(2, 2),
        num_q_values=num_estimators,
        mlp_hidden_dim=32,
        postprocess=(),
    )

    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, H, W, 2))
    params = model.init(key, dummy_input)["params"]

    # Create typical EPE values
    epe_all = np.array(
        [[0.15, 0.25, 0.20], [0.18, 0.22, 0.19]], dtype=np.float32
    )
    targets = -epe_all

    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    def loss_fn(params):
        input_stack = jnp.stack([images, images], axis=-1)
        q_values = jnp.asarray(model.apply({"params": params}, input_stack))
        return jnp.mean((q_values - targets) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Compute gradient statistics
    grad_flat = jax.tree.leaves(grads)
    all_grads = jnp.concatenate([jnp.ravel(g) for g in grad_flat])
    grad_mean = jnp.mean(jnp.abs(all_grads))
    grad_max = jnp.max(jnp.abs(all_grads))

    # Gradients should not be vanishing (too small)
    assert grad_mean > 1e-8, (
        f"Gradients may be vanishing. Mean abs gradient: {grad_mean:.2e}"
    )

    # Gradients should not be exploding (too large)
    assert grad_max < 1e3, (
        f"Gradients may be exploding. Max abs gradient: {grad_max:.2e}"
    )

    # Loss should be a reasonable value for initial random params
    assert 0 < loss < 10, f"Initial loss seems unreasonable: {loss:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Integration test with full training step
# ──────────────────────────────────────────────────────────────────────────────


class FullMockArtOfPIV(MockArtOfPIVForTraining):
    """Mock that uses actual CNN model for q_estimate.

    This more closely simulates the real training scenario while still
    avoiding the need for actual sub-estimators.
    """

    def __init__(self, sub_models, training_mode="bandits", H=32, W=32):
        super().__init__(sub_models, training_mode)
        from flowgym.nn.cnn import CNNQEstimatorModel

        self.H = H
        self.W = W
        self.model = CNNQEstimatorModel(
            features_list=(8, 16),
            norm_fn="none",
            kernel_sizes=(3, 3),
            strides=(2, 2),
            num_q_values=len(sub_models),
            mlp_hidden_dim=32,
            postprocess=(),
        )

    def q_estimate(self, prev, curr, trainable_state, flow=None):
        """Use actual CNN model."""
        input_stack = jnp.stack([prev, curr], axis=-1)  # (B, H, W, 2)
        q_values = self.model.apply(
            {"params": trainable_state.params}, input_stack
        )
        assert isinstance(q_values, jnp.ndarray)
        assert q_values.ndim == 2
        assert q_values.shape[1] == len(self.sub_models)
        return q_values

    def init_trainable_state(self, key, tx):
        """Initialize trainable state with proper CNN parameters."""
        dummy_input = jnp.zeros((1, self.H, self.W, 2))
        params = self.model.init(key, dummy_input)["params"]
        return NNEstimatorTrainableState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            extras={"steps": jnp.zeros((1, 1))},
        )


@pytest.mark.slow
def test_full_integration_regression_convergence():
    """Integration test: regression training with real CNN in full train_step.

    This test uses the actual create_supervised_train_step method with a CNN
    model to verify the full training pipeline works correctly.
    """
    H, W = 32, 32
    B = 4

    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.5)
    model3 = MockSubModel("m3", epe_value=0.3)

    estimator = FullMockArtOfPIV(
        [model1, model2, model3], training_mode="regression", H=H, W=W
    )

    key = jax.random.PRNGKey(42)
    tx = optax.adam(1e-3)
    trainable_state = estimator.init_trainable_state(key, tx)

    # Create consistent input images (same for all training steps)
    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    # EPE values: m1=0.1, m2=0.5, m3=0.3
    epe_all = np.array([[0.1, 0.5, 0.3]] * B, dtype=np.float32)

    experience = SupervisedExperience(
        obs=(images, images),
        state={
            "images": jnp.zeros((B, 1, H, W)),
            "estimates": jnp.zeros((B, 1, H, W, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, H, W, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Track losses and metrics
    losses = []
    for _ in range(300):
        loss, trainable_state, _metrics = train_step(
            trainable_state, experience
        )
        losses.append(float(loss))

    # Get final Q-values
    final_q = jnp.asarray(estimator.q_estimate(images, images, trainable_state))
    mean_q = jnp.mean(final_q, axis=0)

    # Check 1: Loss should decrease
    assert losses[-1] < losses[0] * 0.1, (
        f"Loss did not decrease enough. Initial: {losses[0]:.6f}, "
        f"Final: {losses[-1]:.6f}"
    )

    # Check 2: Ranking should be correct
    # Lower EPE = Higher Q, so order should be: m1 > m3 > m2
    expected_ranking = [0, 2, 1]  # Indices ordered by Q (highest first)
    actual_ranking = list(jnp.argsort(-mean_q))
    assert expected_ranking == actual_ranking, (
        f"Q-value ranking incorrect.\n"
        f"Expected ranking: {expected_ranking}\n"
        f"Actual ranking: {actual_ranking}\n"
        f"Mean Q: {mean_q}"
    )


@pytest.mark.slow
def test_integration_varied_epe_across_batch():
    """Integration test: verify learning when EPE varies across batch samples.

    In real scenarios, different samples may have different optimal estimators.
    This test verifies the model learns to predict sample-specific Q-values.
    """
    H, W = 32, 32
    B = 4

    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.2)

    estimator = FullMockArtOfPIV(
        [model1, model2], training_mode="regression", H=H, W=W
    )

    key = jax.random.PRNGKey(123)
    tx = optax.adam(1e-3)
    trainable_state = estimator.init_trainable_state(key, tx)

    # Create two types of input patterns
    key, k1, k2 = jax.random.split(key, 3)
    images_low = jax.random.uniform(k1, (B // 2, H, W), minval=0, maxval=50)
    images_high = jax.random.uniform(k2, (B // 2, H, W), minval=200, maxval=255)
    images = jnp.concatenate([images_low, images_high], axis=0)

    # Different EPE profiles for different image types
    # Low intensity: m1 is better
    # High intensity: m2 is better
    epe_all = np.array(
        [
            [0.1, 0.5],  # sample 0 (low)
            [0.1, 0.5],  # sample 1 (low)
            [0.5, 0.1],  # sample 2 (high)
            [0.5, 0.1],  # sample 3 (high)
        ],
        dtype=np.float32,
    )

    experience = SupervisedExperience(
        obs=(images, images),
        state={
            "images": jnp.zeros((B, 1, H, W)),
            "estimates": jnp.zeros((B, 1, H, W, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, H, W, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Train
    for _ in range(500):
        _, trainable_state, _ = train_step(trainable_state, experience)

    # Get Q-values for each type of image
    q_low = jnp.asarray(
        estimator.q_estimate(images_low, images_low, trainable_state)
    )
    q_high = jnp.asarray(
        estimator.q_estimate(images_high, images_high, trainable_state)
    )

    # For low intensity images, m1 should have higher Q
    mean_q_low = jnp.mean(q_low, axis=0)
    assert mean_q_low[0] > mean_q_low[1], (
        f"For low intensity, m1 should have higher Q. Got: {mean_q_low}"
    )

    # For high intensity images, m2 should have higher Q
    mean_q_high = jnp.mean(q_high, axis=0)
    assert mean_q_high[1] > mean_q_high[0], (
        f"For high intensity, m2 should have higher Q. Got: {mean_q_high}"
    )


def test_integration_train_step_metrics():
    """Test that train_step returns expected metrics in regression mode."""
    H, W = 32, 32
    B = 2

    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.2)

    estimator = FullMockArtOfPIV(
        [model1, model2], training_mode="regression", H=H, W=W
    )

    key = jax.random.PRNGKey(0)
    tx = optax.adam(1e-3)
    trainable_state = estimator.init_trainable_state(key, tx)

    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    epe_all = np.array([[0.1, 0.2], [0.15, 0.25]], dtype=np.float32)

    experience = SupervisedExperience(
        obs=(images, images),
        state={
            "images": jnp.zeros((B, 1, H, W)),
            "estimates": jnp.zeros((B, 1, H, W, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, H, W, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()
    loss, new_state, metrics = train_step(trainable_state, experience)

    # Check expected metrics are present
    expected_keys = ["loss", "reward_mean", "exploration_val"]
    for key in expected_keys:
        assert key in metrics, (
            f"Missing metric: {key}. Available: {list(metrics.keys())}"
        )

    # Loss should be non-negative
    assert loss >= 0, f"Loss should be non-negative, got {loss}"

    # State should be updated
    assert new_state is not trainable_state, "State should be updated"


# ──────────────────────────────────────────────────────────────────────────────
# Tests for accuracy metric bug detection
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_greedy_accuracy_vs_explored_accuracy():
    """Test that greedy policy accuracy differs from explored accuracy.

    This test exposes a critical issue: during training, the reported accuracy
    is based on *explored* actions (with epsilon-greedy or softmax exploration),
    not on the *greedy* policy actions. When epsilon=1.0 (full exploration),
    the accuracy metric will be approximately 1/num_estimators regardless of
    how well the Q-values are learned.

    The fix should be to report GREEDY accuracy (argmax of Q-values) separately
    from explored accuracy.
    """
    from flowgym.training.exploration import epsilon_greedy_explore

    H, W = 32, 32
    B = 16  # Larger batch for statistical significance

    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.5)
    model3 = MockSubModel("m3", epe_value=0.3)
    model4 = MockSubModel("m4", epe_value=0.8)
    model5 = MockSubModel("m5", epe_value=0.6)

    estimator = FullMockArtOfPIV(
        [model1, model2, model3, model4, model5],
        training_mode="regression",
        H=H,
        W=W,
    )

    # Set exploration to fully random (epsilon=1.0)
    # This simulates the problematic configuration from the user's experiment
    estimator.schedule = lambda step: 1.0  # Always explore randomly
    estimator.explore = epsilon_greedy_explore

    key = jax.random.PRNGKey(42)
    tx = optax.adam(1e-3)
    trainable_state = estimator.init_trainable_state(key, tx)

    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    # EPE values: m1=0.1 is clearly the best
    epe_all = np.array([[0.1, 0.5, 0.3, 0.8, 0.6]] * B, dtype=np.float32)

    experience = SupervisedExperience(
        obs=(images, images),
        state={
            "images": jnp.zeros((B, 1, H, W)),
            "estimates": jnp.zeros((B, 1, H, W, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, H, W, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Train until the model learns the Q-values
    for _ in range(300):
        _, trainable_state, metrics = train_step(trainable_state, experience)

    # Get the reported "accuracy" from training (with exploration)
    # This is based on explored actions
    explored_accuracy = float(metrics.get("choices/accuracy", 0))

    # Compute GREEDY accuracy (what we actually care about)
    final_q = jnp.asarray(estimator.q_estimate(images, images, trainable_state))
    greedy_actions = jnp.argmax(final_q, axis=-1)
    best_actions = jnp.argmin(epe_all, axis=-1)
    greedy_accuracy = float(jnp.mean(greedy_actions == best_actions))

    # With epsilon=1.0 (full exploration), explored accuracy should be
    # about 1 / 5 = 0.2.
    # But greedy accuracy should be close to 1.0 if the model learned correctly
    assert explored_accuracy < 0.5, (
        f"With full exploration, explored accuracy should be low (~0.2). "
        f"Got: {explored_accuracy:.3f}"
    )

    assert greedy_accuracy > 0.9, (
        f"Greedy accuracy should be high (>0.9) after training. "
        f"Got: {greedy_accuracy:.3f}. "
        f"This indicates the Q-values are correctly learned but the reported "
        f"accuracy metric is based on explored actions, not greedy actions."
    )

    # The difference should be significant
    assert greedy_accuracy > explored_accuracy + 0.5, (
        f"Greedy accuracy ({greedy_accuracy:.3f}) should be much higher than "
        f"explored accuracy ({explored_accuracy:.3f}). "
        f"This confirms the accuracy metric bug."
    )


@pytest.mark.slow
def test_regression_with_greedy_exploration():
    """Test that accuracy is correct when exploration is disabled (epsilon=0).

    When epsilon=0, explored actions equal greedy actions, so the accuracy
    metric should correctly reflect the learned policy quality.
    """
    H, W = 32, 32
    B = 8

    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.5)
    model3 = MockSubModel("m3", epe_value=0.3)

    estimator = FullMockArtOfPIV(
        [model1, model2, model3],
        training_mode="regression",
        H=H,
        W=W,
    )

    # Set exploration to greedy (epsilon=0)
    estimator.schedule = lambda step: 0.0
    estimator.explore = lambda q, eps, key: jnp.argmax(q, axis=-1)

    key = jax.random.PRNGKey(42)
    tx = optax.adam(1e-3)
    trainable_state = estimator.init_trainable_state(key, tx)

    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    epe_all = np.array([[0.1, 0.5, 0.3]] * B, dtype=np.float32)

    experience = SupervisedExperience(
        obs=(images, images),
        state={
            "images": jnp.zeros((B, 1, H, W)),
            "estimates": jnp.zeros((B, 1, H, W, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, H, W, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Train
    for _ in range(300):
        _, trainable_state, metrics = train_step(trainable_state, experience)

    # With greedy exploration, reported accuracy should be high
    accuracy = float(metrics.get("choices/accuracy", 0))
    assert accuracy > 0.9, (
        f"With greedy exploration (epsilon=0), accuracy should be high. "
        f"Got: {accuracy:.3f}"
    )


@pytest.mark.slow
def test_regression_training_actually_learns():
    """Diagnostic test for regression updates and loss reduction.

    This test checks:
    1. Q-values change after training
    2. The change is in the correct direction (towards the targets)
    3. The loss decreases
    """
    H, W = 32, 32
    B = 4

    model1 = MockSubModel("m1", epe_value=0.1)
    model2 = MockSubModel("m2", epe_value=0.5)

    estimator = FullMockArtOfPIV(
        [model1, model2], training_mode="regression", H=H, W=W
    )

    key = jax.random.PRNGKey(42)
    tx = optax.adam(1e-3)
    trainable_state = estimator.init_trainable_state(key, tx)

    key, subkey = jax.random.split(key)
    images = jax.random.uniform(subkey, (B, H, W), minval=0, maxval=255)

    epe_all = np.array([[0.1, 0.5]] * B, dtype=np.float32)
    targets = -epe_all  # [-0.1, -0.5]

    experience = SupervisedExperience(
        obs=(images, images),
        state={
            "images": jnp.zeros((B, 1, H, W)),
            "estimates": jnp.zeros((B, 1, H, W, 2)),
            "keys": jax.random.split(jax.random.PRNGKey(0), B)[:, None, :],
            "choices": jnp.zeros((B, 1), dtype=jnp.int32),
        },
        ground_truth=jnp.zeros((B, H, W, 2)),
        cache_payload={"epe_all": epe_all},
    )

    train_step = estimator.create_supervised_train_step()

    # Record initial state
    initial_q = jnp.asarray(
        estimator.q_estimate(images, images, trainable_state)
    )
    initial_q_mean = jnp.mean(initial_q, axis=0)

    # Train
    losses = []
    for _ in range(200):
        loss, trainable_state, _ = train_step(trainable_state, experience)
        losses.append(float(loss))

    # Record final state
    final_q = estimator.q_estimate(images, images, trainable_state)
    final_q_mean = jnp.mean(final_q, axis=0)

    # Check 1: Loss decreased
    assert losses[-1] < losses[0] * 0.1, (
        f"Loss should decrease significantly. "
        f"Initial: {losses[0]:.6f}, Final: {losses[-1]:.6f}"
    )

    # Check 2: Q-values changed
    q_changed = not jnp.allclose(initial_q_mean, final_q_mean, atol=0.01)
    assert q_changed, (
        f"Q-values should change during training. "
        f"Initial: {initial_q_mean}, Final: {final_q_mean}"
    )

    # Check 3: Q-values are close to targets
    target_mean = jnp.mean(jnp.array(targets), axis=0)
    assert jnp.allclose(final_q_mean, target_mean, atol=0.1), (
        f"Q-values should be close to targets. "
        f"Final Q: {final_q_mean}, Targets: {target_mean}"
    )

    # Check 4: The ranking is correct (m1 has higher Q because lower EPE)
    assert final_q_mean[0] > final_q_mean[1], (
        f"Q[0] should be higher than Q[1] (lower EPE = higher reward). "
        f"Got Q: {final_q_mean}"
    )


def test_regression_mode_defaults_to_greedy_exploration():
    """Test that regression mode defaults to greedy exploration (epsilon=0).

    This is important because exploration doesn't affect training in regression
    mode (all actions are trained), but it DOES affect the reported accuracy
    metric. By defaulting to greedy, the accuracy metric reflects the actual
    learned policy quality.
    """
    import tempfile

    import yaml

    # Create a minimal estimator config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(
            {
                "estimators": [
                    {"name": "dummy1", "type": "mock"},
                    {"name": "dummy2", "type": "mock"},
                ]
            },
            f,
        )
        config_path = f.name

    try:
        # Create a regression estimator without an explicit exploration config.
        estimator = ArtOfPIVEstimator(
            router_config={
                "features": [8],
                "kernel_sizes": [3],
                "strides": [2],
            },
            estimators_list_path=config_path,
            training_mode="regression",
            exploration_config=None,  # Use default
            image_shape=(1, 64, 64),
            rng=0,
            initialize_estimators=False,
        )

        # Check that the schedule returns 0.0 (greedy)
        epsilon = estimator.schedule(0)
        assert epsilon == 0.0, (
            f"Regression mode should default to greedy (epsilon=0). "
            f"Got epsilon={epsilon}"
        )

        # Also verify for classification mode
        estimator_clf = ArtOfPIVEstimator(
            router_config={
                "features": [8],
                "kernel_sizes": [3],
                "strides": [2],
            },
            estimators_list_path=config_path,
            training_mode="classification",
            loss_config={"name": "softmax_cross_entropy_with_integer_labels"},
            exploration_config=None,
            image_shape=(1, 64, 64),
            rng=1,
            initialize_estimators=False,
        )
        epsilon_clf = estimator_clf.schedule(0)
        assert epsilon_clf == 0.0, (
            "Classification mode should default to greedy. "
            f"Got epsilon={epsilon_clf}"
        )

        # Bandits mode should still default to epsilon=0.1
        estimator_bandits = ArtOfPIVEstimator(
            router_config={
                "features": [8],
                "kernel_sizes": [3],
                "strides": [2],
            },
            estimators_list_path=config_path,
            training_mode="bandits",
            exploration_config=None,
            image_shape=(1, 64, 64),
            rng=2,
            initialize_estimators=False,
        )
        epsilon_bandits = estimator_bandits.schedule(0)
        assert epsilon_bandits == 0.1, (
            "Bandits mode should default to epsilon=0.1. "
            f"Got epsilon={epsilon_bandits}"
        )

    finally:
        import os

        os.unlink(config_path)
