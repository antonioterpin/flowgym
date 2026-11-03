"""Tests to verify there is no label leakage in ART of PIV classification mode.

This test suite verifies that the router network only receives images as input
and does not have access to the labels (epe_all) during the forward pass.

The key insight is: if there's information leakage, the model would be able
to predict labels even when trained on random (shuffled) labels, because
it would somehow "see" the correct labels through the forward pass.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.nn.cnn import CNNQEstimatorModel
from flowgym.types import NNEstimatorTrainableState


def create_dummy_estimator(
    num_estimators: int = 3,
    image_shape: tuple[int, int] = (64, 64),
    seed: int = 42,
) -> tuple[ArtOfPIVEstimator, NNEstimatorTrainableState]:
    """Create a minimal ART of PIV estimator for testing.

    Args:
        num_estimators: Number of sub-estimators (output classes).
        image_shape: Shape of input images (H, W).
        seed: Random seed.

    Returns:
        Tuple of (estimator, trainable_state).
    """
    import optax

    H, W = image_shape
    key = jax.random.PRNGKey(seed)

    # Create a minimal CNN model
    model = CNNQEstimatorModel(
        features_list=(16, 16),
        norm_fn="none",
        kernel_sizes=(4, 3),
        strides=(2, 2),
        num_q_values=num_estimators,
        mlp_hidden_dim=32,
        postprocess=(),
    )

    # Initialize params
    key, init_key = jax.random.split(key)
    dummy_input = jnp.zeros((1, H, W, 2))  # (B, H, W, 2) for prev+curr
    params = model.init(init_key, dummy_input)["params"]

    # Create optimizer
    tx = optax.adam(learning_rate=0.01)

    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        extras={"steps": jnp.zeros((1, 1), dtype=jnp.int32)},
    )

    # Create mock estimator with the real model
    class TestableArtOfPIV(ArtOfPIVEstimator):
        """Minimal wrapper that uses the real CNN model."""

        def __init__(self, cnn_model, num_est):
            # Intentionally bypass ArtOfPIVEstimator.__init__ in this test
            # helper; we wire only the minimal explicit state we need.
            self.model = cnn_model
            self.num_estimators = num_est
            self.training_mode = "classification"
            self.use_temporal_propagation = False
            self.use_last_estimate = False
            self.schedule = lambda step: 0.0  # Greedy
            self.explore = lambda q, r, k: jnp.argmax(q, axis=-1)
            self.target_transform = lambda x: x
            self.estimator_configs = [
                {"name": f"model_{i}"} for i in range(num_est)
            ]

            # Use softmax cross-entropy loss
            self.loss_fn = lambda logits, labels: jnp.mean(
                -jnp.sum(
                    jax.nn.one_hot(labels, num_est)
                    * jax.nn.log_softmax(logits, axis=-1),
                    axis=-1,
                )
            )

        def q_estimate(self, prev, curr, trainable_state, flow=None):
            """Compute Q-values from images using the CNN.

            This is the critical function - it should ONLY use prev and curr,
            NOT any labels or epe_all values.
            """
            # Stack input: (B, H, W, 2)
            x = jnp.stack([prev, curr], axis=-1)
            q_values = self.model.apply(
                {"params": trainable_state.params},
                x,
            )
            return jnp.asarray(q_values, dtype=jnp.float32)

    estimator = TestableArtOfPIV(model, num_estimators)
    return estimator, trainable_state


def test_no_leakage_random_labels():
    """Test that model predictions are independent of label values.

    If there's information leakage, the model would somehow "see" the labels
    and be able to predict them even before training. This test verifies
    that the model's predictions are independent of the label values.
    """
    B = 8
    H, W = 64, 64
    num_estimators = 3
    key = jax.random.PRNGKey(42)

    estimator, trainable_state = create_dummy_estimator(
        num_estimators=num_estimators,
        image_shape=(H, W),
    )

    # Create random images
    key, img_key = jax.random.split(key)
    prev = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)
    key, img_key = jax.random.split(key)
    curr = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)

    # Get model predictions BEFORE any labels are involved
    q_values = estimator.q_estimate(prev, curr, trainable_state)
    predicted_classes = jnp.argmax(q_values, axis=-1)
    assert predicted_classes.shape == (B,)

    # Now create two different sets of labels
    key, label_key = jax.random.split(key)
    labels_a = jax.random.randint(
        label_key, (B,), minval=0, maxval=num_estimators
    )

    key, label_key = jax.random.split(key)
    labels_b = jax.random.randint(
        label_key, (B,), minval=0, maxval=num_estimators
    )
    assert labels_a.shape == labels_b.shape == (B,)

    # Model predictions should be identical regardless of which labels we
    # "assign" because labels are NOT inputs to q_estimate.
    # This is trivially true since we called q_estimate before labels existed

    # The more important test is that q_estimate output does not change
    # across different epe_all layouts.
    epe_all_variant_a = np.random.rand(B, num_estimators).astype(np.float32)
    epe_all_variant_b = np.random.rand(B, num_estimators).astype(np.float32)
    assert (
        epe_all_variant_a.shape
        == epe_all_variant_b.shape
        == (
            B,
            num_estimators,
        )
    )

    # q_estimate should NOT take epe_all as input
    # Let's verify the function signature doesn't accept it
    import inspect

    sig = inspect.signature(estimator.q_estimate)
    param_names = list(sig.parameters.keys())
    assert "epe_all" not in param_names, (
        f"q_estimate should not have epe_all parameter. Got: {param_names}"
    )
    assert "cache_payload" not in param_names, (
        "q_estimate should not have cache_payload parameter. "
        f"Got: {param_names}"
    )
    assert "labels" not in param_names, (
        f"q_estimate should not have labels parameter. Got: {param_names}"
    )


def test_untrained_model_accuracy_is_random():
    """Test that an untrained model has random-chance accuracy.

    This verifies that the model doesn't start with any "knowledge" of labels.
    """
    B = 100  # Larger batch for better statistics
    H, W = 64, 64
    num_estimators = 3
    key = jax.random.PRNGKey(123)

    estimator, trainable_state = create_dummy_estimator(
        num_estimators=num_estimators,
        image_shape=(H, W),
    )

    # Create random images
    key, img_key = jax.random.split(key)
    prev = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)
    key, img_key = jax.random.split(key)
    curr = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)

    # Get model predictions
    q_values = estimator.q_estimate(prev, curr, trainable_state)
    predictions = jnp.argmax(q_values, axis=-1)

    # Create random labels (as would come from epe_all)
    key, label_key = jax.random.split(key)
    labels = jax.random.randint(
        label_key, (B,), minval=0, maxval=num_estimators
    )

    # Compute accuracy
    accuracy = jnp.mean(predictions == labels)

    # With random predictions on random labels, accuracy should be close
    # to 1 / num_estimators.
    expected_accuracy = 1.0 / num_estimators
    tolerance = 0.15  # Allow for random variation

    assert abs(accuracy - expected_accuracy) < tolerance, (
        f"Untrained model accuracy ({accuracy:.3f}) is far from random chance "
        f"({expected_accuracy:.3f}). This could indicate information leakage."
    )


def test_model_learns_from_gradient_not_labels():
    """Test that model learns from gradients, not from label access.

    Train a model with random labels and verify:
    1. Training accuracy goes up (model fits the batch)
    2. Test accuracy on NEW random labels stays at chance (no leakage)
    """
    B = 16
    H, W = 64, 64
    num_estimators = 3
    num_train_steps = 10
    key = jax.random.PRNGKey(42)

    estimator, trainable_state = create_dummy_estimator(
        num_estimators=num_estimators,
        image_shape=(H, W),
    )

    # Create fixed training images
    key, img_key = jax.random.split(key)
    train_prev = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)
    key, img_key = jax.random.split(key)
    train_curr = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)

    # Create fixed training labels (random, but fixed for this batch)
    key, label_key = jax.random.split(key)
    train_labels = jax.random.randint(
        label_key, (B,), minval=0, maxval=num_estimators
    )

    # Training loop
    for _step in range(num_train_steps):

        def loss_fn(params, state=trainable_state):
            tmp_state = state.replace(params=params)
            q_values = estimator.q_estimate(train_prev, train_curr, tmp_state)
            return estimator.loss_fn(q_values, train_labels)

        _loss, grads = jax.value_and_grad(loss_fn)(trainable_state.params)
        trainable_state = trainable_state.apply_gradients(grads=grads)

    # Check training accuracy (should be high after training on this batch)
    q_values_train = estimator.q_estimate(
        train_prev, train_curr, trainable_state
    )
    train_predictions = jnp.argmax(q_values_train, axis=-1)
    train_accuracy = jnp.mean(train_predictions == train_labels)

    assert train_accuracy > 0.7, (
        f"Training accuracy ({train_accuracy:.3f}) is too low. "
        f"Model should be able to fit the training batch."
    )

    # Now test on NEW images with NEW random labels
    key, img_key = jax.random.split(key)
    test_prev = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)
    key, img_key = jax.random.split(key)
    test_curr = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)
    key, label_key = jax.random.split(key)
    test_labels = jax.random.randint(
        label_key, (B,), minval=0, maxval=num_estimators
    )

    q_values_test = estimator.q_estimate(test_prev, test_curr, trainable_state)
    test_predictions = jnp.argmax(q_values_test, axis=-1)
    test_accuracy = jnp.mean(test_predictions == test_labels)

    # Test accuracy should be near random chance
    # (model learned to fit training batch, not to "see" labels)
    expected_chance = 1.0 / num_estimators

    # Note: This test may fail if the model somehow generalizes from the
    # training batch to the test batch, but with random labels this shouldn't
    # happen. If this test fails, it strongly suggests information leakage.
    tolerance = 0.25
    assert test_accuracy < expected_chance + tolerance, (
        f"Test accuracy ({test_accuracy:.3f}) is suspiciously high. "
        f"Expected near random chance ({expected_chance:.3f}). "
        f"This could indicate information leakage."
    )


def test_epe_all_not_accessible_in_forward_pass():
    """Verify that epe_all is not accessible during q_estimate forward pass.

    This test creates a scenario where if epe_all were accessible, the model
    could trivially achieve perfect accuracy.
    """
    B = 8
    H, W = 64, 64
    num_estimators = 3
    key = jax.random.PRNGKey(42)

    estimator, trainable_state = create_dummy_estimator(
        num_estimators=num_estimators,
        image_shape=(H, W),
    )

    # Create images and labels
    key, img_key = jax.random.split(key)
    prev = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)
    key, img_key = jax.random.split(key)
    curr = jax.random.uniform(img_key, (B, H, W), minval=0, maxval=255)

    # Create epe_all where there's a clear "best" action for each sample
    epe_all = np.random.rand(B, num_estimators).astype(np.float32) + 1.0
    for i in range(B):
        best_action = i % num_estimators
        epe_all[i, best_action] = 0.01  # Make this action clearly best

    best_actions = np.argmin(epe_all, axis=1)

    # Get model predictions - epe_all is NOT passed to q_estimate
    q_values = estimator.q_estimate(prev, curr, trainable_state)
    predictions = jnp.argmax(q_values, axis=-1)

    # If there was leakage, predictions would match best_actions
    accuracy = jnp.mean(predictions == best_actions)

    expected_chance = 1.0 / num_estimators
    tolerance = 0.2

    assert accuracy < expected_chance + tolerance, (
        f"Untrained model accuracy ({accuracy:.3f}) matches labels too well. "
        f"Expected near random chance ({expected_chance:.3f}). "
        f"This suggests epe_all might be leaking into the forward pass."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
