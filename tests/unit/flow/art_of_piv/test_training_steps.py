"""Tests for training_steps module."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from flax.core.frozen_dict import freeze

from flowgym.common.base import NNEstimatorTrainableState
from flowgym.flow.art_of_piv import training_steps
from flowgym.types import RLExperience, SupervisedExperience


def make_mock_trainable_state(batch_size=2, num_estimators=3):
    """Create a minimal trainable state for testing."""
    import optax

    # Make params affect the output so the test observes real gradients.
    params = freeze({"dummy": jnp.ones((1, num_estimators))})

    def mock_apply(variables, x):
        # Use params to ensure gradients are non-zero
        dummy_weight = variables["dummy"]
        return jnp.broadcast_to(dummy_weight, (x.shape[0], num_estimators))

    tx = optax.sgd(0.01)
    extras = {"steps": jnp.zeros((batch_size, 1), dtype=jnp.int32)}
    return NNEstimatorTrainableState.create(
        apply_fn=mock_apply,
        params=params,
        tx=tx,
        extras=extras,
    )


def make_mock_supervised_experience(batch_size=2, h=4, w=4, epe_all=None):
    """Create mock supervised experience for testing."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)

    obs = (
        jnp.zeros((batch_size, h, w)),  # prev frames
        jnp.zeros((batch_size, h, w)),  # curr frames
    )

    state = {
        "images": jnp.zeros((batch_size, 1, h, w)),
        "estimates": jnp.zeros((batch_size, 1, h, w, 2)),
        "keys": keys[:, None, :],
        "choices": jnp.zeros((batch_size, 1), dtype=jnp.int32),
    }

    ground_truth = jnp.zeros((batch_size, h, w, 2))

    cache_payload = None
    if epe_all is not None:
        cache_payload = {"epe_all": epe_all}

    return SupervisedExperience(
        obs=obs,
        state=state,
        ground_truth=ground_truth,
        cache_payload=cache_payload,
    )


def make_mock_rl_experience(batch_size=2, h=4, w=4):
    """Create mock RL experience for testing."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)

    state = {
        "images": jnp.zeros((batch_size, 1, h, w)),
        "estimates": jnp.zeros((batch_size, 1, h, w, 2)),
        "keys": keys[:, None, :],
        "choices": jnp.zeros((batch_size, 1), dtype=jnp.int32),
    }

    obs = (
        jnp.zeros((batch_size, h, w)),  # curr prev frames
        jnp.zeros((batch_size, h, w)),  # curr curr frames
    )

    old_obs = (
        jnp.zeros((batch_size, h, w)),  # old prev frames
        jnp.zeros((batch_size, h, w)),  # old curr frames
    )

    reward = jnp.ones((batch_size,)) * 0.5
    old_flow_estimate = jnp.zeros((batch_size, h, w, 2))

    return RLExperience(
        state=state,
        reward=reward,
        obs=obs,
        old_obs=old_obs,
        old_flow_estimate=old_flow_estimate,
    )


def test_supervised_bandits_loss_and_grad():
    """Test bandits mode computes finite loss and updates parameters."""
    batch_size = 2
    num_estimators = 3
    h, w = 4, 4

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    experience = make_mock_supervised_experience(batch_size, h, w)

    # Mock functions
    def mock_q_estimate(prev, curr, ts, flow=None):
        # Make Q-values depend on the parameters for meaningful gradients
        dummy_weight = ts.params["dummy"]
        return jnp.broadcast_to(dummy_weight, (batch_size, num_estimators))

    def mock_estimate(images, state, ts, cache_payload=None):
        # Return updated state with choices
        updated_state = {**state, "choices": jnp.array([[1], [0]])}
        return updated_state, {}

    def mock_run_router(prev, curr, state, ts, flow=None, train=True):
        actions = jnp.array([1, 0])
        qs = mock_q_estimate(prev, curr, ts)  # Use the actual Q-estimate
        exploration_val = jnp.array([0.1, 0.1])
        estimate_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        return actions, qs, exploration_val, estimate_keys

    def mock_loss(pred, target):
        # Make sure loss actually depends on predictions
        return jnp.mean(jnp.sum(pred**2)) + jnp.mean((pred - target) ** 2)

    def mock_target_transform(x):
        return x

    def mock_schedule(step):
        return 0.1

    train_step = training_steps.create_supervised_train_step(
        q_estimate_fn=mock_q_estimate,
        estimate_fn=mock_estimate,
        run_router_fn=mock_run_router,
        loss_fn=mock_loss,
        target_transform=mock_target_transform,
        schedule=mock_schedule,
        training_mode="bandits",
        use_temporal_propagation=False,
        num_estimators=num_estimators,
        estimator_configs=[
            {"name": f"model_{i}"} for i in range(num_estimators)
        ],
        estimator_fns=tuple(
            [lambda x: (x, {"epe": 0.1}) for _ in range(num_estimators)]
        ),
    )

    original_params = trainable_state.params
    loss, updated_state, metrics = train_step(trainable_state, experience)

    assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"
    # Parameters should change after the gradient update.
    assert not jnp.allclose(
        updated_state.params["dummy"], original_params["dummy"]
    ), "Parameters should be updated"
    assert "loss" in metrics
    assert "reward_mean" in metrics


def test_supervised_bandits_metrics_keys():
    """Test that bandits mode returns expected metric keys."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    experience = make_mock_supervised_experience(batch_size)

    def mock_q_estimate(prev, curr, ts, flow=None):
        return jnp.ones((batch_size, num_estimators))

    def mock_estimate(images, state, ts, cache_payload=None):
        updated_state = {**state, "choices": jnp.array([[1], [0]])}
        return updated_state, {}

    def mock_run_router(prev, curr, state, ts, flow=None, train=True):
        actions = jnp.array([1, 0])
        qs = jnp.ones((batch_size, num_estimators))
        exploration_val = jnp.array([0.1, 0.1])
        estimate_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        return actions, qs, exploration_val, estimate_keys

    train_step = training_steps.create_supervised_train_step(
        q_estimate_fn=mock_q_estimate,
        estimate_fn=mock_estimate,
        run_router_fn=mock_run_router,
        loss_fn=lambda p, t: jnp.mean((p - t) ** 2),
        target_transform=lambda x: x,
        schedule=lambda step: 0.1,
        training_mode="bandits",
        use_temporal_propagation=False,
        num_estimators=num_estimators,
        estimator_configs=[
            {"name": f"model_{i}"} for i in range(num_estimators)
        ],
        estimator_fns=tuple(
            [lambda x: (x, {"epe": 0.1}) for _ in range(num_estimators)]
        ),
    )

    _, _, metrics = train_step(trainable_state, experience)

    expected_keys = {
        "reward_mean",
        "epe/raw_mean",
        "target/mean",
        "exploration_val",
        "learning_rate",
        "loss",
        "choices/entropy",
    }
    assert all(key in metrics for key in expected_keys), (
        f"Missing expected metrics. Got keys: {set(metrics.keys())}"
    )


def test_supervised_bandits_steps_increment():
    """Test that training increments the steps counter."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    experience = make_mock_supervised_experience(batch_size)

    def mock_q_estimate(prev, curr, ts, flow=None):
        return jnp.ones((batch_size, num_estimators))

    def mock_estimate(images, state, ts, cache_payload=None):
        updated_state = {**state, "choices": jnp.array([[1], [0]])}
        return updated_state, {}

    def mock_run_router(prev, curr, state, ts, flow=None, train=True):
        actions = jnp.array([1, 0])
        qs = jnp.ones((batch_size, num_estimators))
        exploration_val = jnp.array([0.1, 0.1])
        estimate_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        return actions, qs, exploration_val, estimate_keys

    train_step = training_steps.create_supervised_train_step(
        q_estimate_fn=mock_q_estimate,
        estimate_fn=mock_estimate,
        run_router_fn=mock_run_router,
        loss_fn=lambda p, t: jnp.mean((p - t) ** 2),
        target_transform=lambda x: x,
        schedule=lambda step: 0.1,
        training_mode="bandits",
        use_temporal_propagation=False,
        num_estimators=num_estimators,
        estimator_configs=[
            {"name": f"model_{i}"} for i in range(num_estimators)
        ],
        estimator_fns=tuple(
            [lambda x: (x, {"epe": 0.1}) for _ in range(num_estimators)]
        ),
    )

    original_steps = trainable_state.extras["steps"]
    _, updated_state, _ = train_step(trainable_state, experience)

    assert jnp.array_equal(updated_state.extras["steps"], original_steps + 1), (
        "Steps should increment by 1"
    )


def test_supervised_regression_targets_all_actions():
    """Test that regression mode targets all actions."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    # Provide cached EPE for all actions
    epe_all = jnp.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
    experience = make_mock_supervised_experience(batch_size, epe_all=epe_all)

    loss_called_with = None

    def mock_q_estimate(prev, curr, ts, flow=None):
        return jnp.ones((batch_size, num_estimators))

    def mock_estimate(images, state, ts, cache_payload=None):
        updated_state = {**state, "choices": jnp.array([[1], [0]])}
        return updated_state, {}

    def mock_run_router(prev, curr, state, ts, flow=None, train=True):
        actions = jnp.array([1, 0])
        qs = jnp.ones((batch_size, num_estimators))
        exploration_val = jnp.array([0.1, 0.1])
        estimate_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        return actions, qs, exploration_val, estimate_keys

    def mock_loss(pred, target):
        nonlocal loss_called_with
        loss_called_with = (pred.shape, target.shape)
        return jnp.mean((pred - target) ** 2)

    train_step = training_steps.create_supervised_train_step(
        q_estimate_fn=mock_q_estimate,
        estimate_fn=mock_estimate,
        run_router_fn=mock_run_router,
        loss_fn=mock_loss,
        target_transform=lambda x: x,
        schedule=lambda step: 0.1,
        training_mode="regression",  # Regression mode
        use_temporal_propagation=False,
        num_estimators=num_estimators,
        estimator_configs=[
            {"name": f"model_{i}"} for i in range(num_estimators)
        ],
        estimator_fns=tuple(
            [lambda x: (x, {"epe": 0.1}) for _ in range(num_estimators)]
        ),
    )

    _, _, _ = train_step(trainable_state, experience)

    # In regression mode, both pred and target should span all estimators.
    assert loss_called_with == (
        (batch_size, num_estimators),
        (batch_size, num_estimators),
    ), f"Expected shapes ((2, 3), (2, 3)), got {loss_called_with}"


def test_supervised_classification_target_is_argmin():
    """Test that classification mode targets are argmin of EPE."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    epe_all = jnp.array([[0.3, 0.1, 0.2], [0.2, 0.4, 0.1]])  # Best: [1, 2]
    experience = make_mock_supervised_experience(batch_size, epe_all=epe_all)

    loss_called_with = None

    def mock_q_estimate(prev, curr, ts, flow=None):
        return jnp.ones((batch_size, num_estimators))

    def mock_estimate(images, state, ts, cache_payload=None):
        updated_state = {**state, "choices": jnp.array([[1], [2]])}
        return updated_state, {}

    def mock_run_router(prev, curr, state, ts, flow=None, train=True):
        actions = jnp.array([1, 2])
        qs = jnp.ones((batch_size, num_estimators))
        exploration_val = jnp.array([0.1, 0.1])
        estimate_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        return actions, qs, exploration_val, estimate_keys

    def mock_loss(pred, target):
        nonlocal loss_called_with
        loss_called_with = (pred, target)
        return jnp.mean(pred)  # Dummy loss

    train_step = training_steps.create_supervised_train_step(
        q_estimate_fn=mock_q_estimate,
        estimate_fn=mock_estimate,
        run_router_fn=mock_run_router,
        loss_fn=mock_loss,
        target_transform=lambda x: x,
        schedule=lambda step: 0.1,
        training_mode="classification",
        use_temporal_propagation=False,
        num_estimators=num_estimators,
        estimator_configs=[
            {"name": f"model_{i}"} for i in range(num_estimators)
        ],
        estimator_fns=tuple(
            [lambda x: (x, {"epe": 0.1}) for _ in range(num_estimators)]
        ),
    )

    _, _, _ = train_step(trainable_state, experience)

    pred_shape, target = loss_called_with
    expected_target = jnp.array([1, 2])  # argmin of epe_all
    assert jnp.array_equal(target, expected_target), (
        f"Expected target {expected_target}, got {target}"
    )
    assert pred_shape.shape == (batch_size, num_estimators), (
        f"Expected pred shape (2, 3), got {pred_shape.shape}"
    )


def test_rl_step_dqn_loss():
    """Test RL step computes DQN loss and updates gradients."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    target_state = trainable_state  # Same for simplicity
    experience = make_mock_rl_experience(batch_size)

    def mock_q_estimate(prev, curr, ts, flow=None):
        # Make Q-values depend on the parameters for meaningful gradients
        dummy_weight = ts.params["dummy"]
        return jnp.broadcast_to(dummy_weight, (batch_size, num_estimators))

    def mock_loss(pred, target):
        # Make sure loss actually depends on predictions
        return jnp.mean(jnp.sum(pred**2)) + jnp.mean((pred - target) ** 2)

    train_step = training_steps.create_rl_train_step(
        q_estimate_fn=mock_q_estimate,
        loss_fn=mock_loss,
        use_temporal_propagation=False,
        gamma=0.0,
    )

    original_params = trainable_state.params
    loss, updated_state, metrics = train_step(
        trainable_state, target_state, experience
    )

    assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"
    # Parameters should change after the gradient update.
    assert not jnp.allclose(
        updated_state.params["dummy"], original_params["dummy"]
    ), "Parameters should be updated"
    assert "reward_mean" in metrics


def test_rl_step_gamma_zero():
    """Test that RL step uses reward as target when gamma=0."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    target_state = trainable_state

    reward = jnp.array([0.5, -0.3])
    experience = make_mock_rl_experience(batch_size)
    experience = replace(experience, reward=reward)

    loss_called_with = None

    def mock_q_estimate(prev, curr, ts, flow=None):
        return jnp.ones((batch_size, num_estimators))

    def mock_loss(pred, target):
        nonlocal loss_called_with
        loss_called_with = target
        return jnp.mean((pred - target) ** 2)

    train_step = training_steps.create_rl_train_step(
        q_estimate_fn=mock_q_estimate,
        loss_fn=mock_loss,
        use_temporal_propagation=False,
        gamma=0.0,
    )

    _, _, _ = train_step(trainable_state, target_state, experience)

    assert jnp.allclose(loss_called_with, reward), (
        f"Expected target to equal reward {reward}, got {loss_called_with}"
    )


def test_rl_step_gamma_nonzero_raises():
    """Test that gamma > 0 raises NotImplementedError."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    target_state = trainable_state
    experience = make_mock_rl_experience(batch_size)

    def mock_q_estimate(prev, curr, ts, flow=None):
        return jnp.ones((batch_size, num_estimators))

    train_step = training_steps.create_rl_train_step(
        q_estimate_fn=mock_q_estimate,
        loss_fn=lambda p, t: jnp.mean((p - t) ** 2),
        use_temporal_propagation=False,
        gamma=0.9,  # Non-zero gamma
    )

    with pytest.raises(NotImplementedError, match="gamma > 0 not implemented"):
        train_step(trainable_state, target_state, experience)


def test_create_train_step_routes_supervised():
    """Test that create_train_step routes to supervised factory."""
    supervised_called = False
    rl_called = False

    def mock_supervised_factory():
        nonlocal supervised_called
        supervised_called = True
        return lambda ts, exp: (jnp.array(0.0), ts, {})

    def mock_rl_factory():
        nonlocal rl_called
        rl_called = True
        return lambda ts, ts2, exp: (jnp.array(0.0), ts, {})

    train_step = training_steps.create_train_step(
        train_type="supervised",
        create_supervised_fn=mock_supervised_factory,
        create_rl_fn=mock_rl_factory,
    )

    assert supervised_called, "Supervised factory should be called"
    assert not rl_called, "RL factory should not be called"
    assert callable(train_step), "Should return a callable training step"


def test_create_train_step_routes_rl():
    """Test that create_train_step routes to RL factory."""
    supervised_called = False
    rl_called = False

    def mock_supervised_factory():
        nonlocal supervised_called
        supervised_called = True
        return lambda ts, exp: (jnp.array(0.0), ts, {})

    def mock_rl_factory():
        nonlocal rl_called
        rl_called = True
        return lambda ts, ts2, exp: (jnp.array(0.0), ts, {})

    train_step = training_steps.create_train_step(
        train_type="rl",
        create_supervised_fn=mock_supervised_factory,
        create_rl_fn=mock_rl_factory,
    )

    assert not supervised_called, "Supervised factory should not be called"
    assert rl_called, "RL factory should be called"
    assert callable(train_step), "Should return a callable training step"


def test_prepare_experience_calls_estimate():
    """Test that prepare_experience_for_replay calls estimate_fn."""
    batch_size = 2
    num_estimators = 3

    trainable_state = make_mock_trainable_state(batch_size, num_estimators)
    experience = make_mock_supervised_experience(batch_size)

    estimate_called_with = None

    def mock_estimate(images, state, ts):
        nonlocal estimate_called_with
        estimate_called_with = (images.shape, ts)
        updated_state = {**state, "estimates": state["estimates"] + 1}
        return updated_state, {}

    result = training_steps.prepare_experience_for_replay(
        experience=experience,
        trainable_state=trainable_state,
        estimate_fn=mock_estimate,
    )

    assert estimate_called_with is not None, "estimate_fn should be called"
    images_shape, called_ts = estimate_called_with
    assert images_shape == (batch_size, 4, 4), (
        f"Expected (2, 4, 4), got {images_shape}"
    )
    assert called_ts == trainable_state, "Should pass trainable_state"

    # Check that state was updated
    assert jnp.allclose(
        result.state["estimates"], experience.state["estimates"] + 1
    ), "State should be updated with new estimates"
