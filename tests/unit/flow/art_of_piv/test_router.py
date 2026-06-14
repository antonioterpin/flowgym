"""Tests for router module."""

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze

from flowgym.common.base import NNEstimatorTrainableState
from flowgym.flow.art_of_piv import router


def make_mock_trainable_state(batch_size=2, num_estimators=3):
    """Create a minimal trainable state for testing."""
    import optax

    # Simple frozen dict for params (doesn't need to be a real model)
    params = freeze({"dummy": jnp.ones((1, 1))})

    # Mock apply function that doesn't actually use params
    def mock_apply(variables, x):
        return jnp.ones((x.shape[0], num_estimators), dtype=jnp.float32)

    tx = optax.sgd(0.01)
    extras = {"steps": jnp.zeros((batch_size, 1), dtype=jnp.int32)}
    return NNEstimatorTrainableState.create(
        apply_fn=mock_apply,
        params=params,
        tx=tx,
        extras=extras,
    )


def test_q_estimate_stacks_prev_curr():
    """q_estimate stacks prev/curr when last-estimate flow is disabled."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))
    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    # Track what shape the model receives
    received_shape = None

    def mock_apply_fn(variables, x):
        nonlocal received_shape
        received_shape = x.shape
        return jnp.ones((x.shape[0], num_estimators), dtype=jnp.float32)

    result = router.q_estimate(
        prev,
        curr,
        trainable_state,
        model_apply_fn=mock_apply_fn,
        use_last_estimate=False,
        flow=None,
    )

    assert received_shape == (batch_size, h, w, 2), (
        f"Expected input shape (B, H, W, 2), got {received_shape}"
    )
    assert result.shape == (batch_size, num_estimators)


def test_q_estimate_concatenates_flow():
    """Test that q_estimate concatenates flow when use_last_estimate=True."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))
    flow = jnp.ones((batch_size, h, w, 2))
    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    received_shape = None

    def mock_apply_fn(variables, x):
        nonlocal received_shape
        received_shape = x.shape
        return jnp.ones((x.shape[0], num_estimators), dtype=jnp.float32)

    result = router.q_estimate(
        prev,
        curr,
        trainable_state,
        model_apply_fn=mock_apply_fn,
        use_last_estimate=True,
        flow=flow,
    )

    assert received_shape == (batch_size, h, w, 4), (
        f"Expected input shape (B, H, W, 4) with flow, got {received_shape}"
    )
    assert result.shape == (batch_size, num_estimators)


def test_q_estimate_ignores_flow_when_disabled():
    """Passing flow should not affect outputs when flow usage is disabled."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))
    flow = jnp.ones((batch_size, h, w, 2))
    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    def mock_apply_fn(variables, x):
        channel_sum = jnp.sum(x, axis=(-1, -2, -3))
        return jnp.stack(
            [channel_sum, channel_sum + 1.0, channel_sum + 2.0],
            axis=-1,
        )

    result_without_flow = router.q_estimate(
        prev,
        curr,
        trainable_state,
        model_apply_fn=mock_apply_fn,
        use_last_estimate=False,
        flow=None,
    )

    result_with_flow = router.q_estimate(
        prev,
        curr,
        trainable_state,
        model_apply_fn=mock_apply_fn,
        use_last_estimate=False,
        flow=flow,
    )

    assert jnp.allclose(result_with_flow, result_without_flow), (
        "q_estimate should ignore flow when use_last_estimate=False."
    )
    assert result_with_flow.shape == (batch_size, num_estimators), (
        f"Expected output shape ({batch_size}, {num_estimators}), got "
        f"{result_with_flow.shape}"
    )


def test_run_router_zeroes_exploration_at_eval():
    """train=False should force greedy action selection via zero exploration."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))
    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    # Create keys
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    steps = trainable_state.extras["steps"]

    def mock_q_estimate_fn(prev, curr, flow=None):
        # Distinct argmax per row so we can validate greedy behavior directly.
        return jnp.array([[0.1, 0.9, 0.2], [0.8, 0.1, 0.3]], dtype=jnp.float32)

    def mock_schedule(step):
        return 0.5  # Return non-zero exploration

    def mock_explore(qs, exploration_val, key):
        greedy_actions = jnp.argmax(qs, axis=-1)
        forced_exploration_actions = jnp.full_like(greedy_actions, 2)
        return jnp.where(
            exploration_val > 0.0,
            forced_exploration_actions,
            greedy_actions,
        )

    actions_eval, _qs, exploration_val_eval, _estimate_keys = router.run_router(
        prev,
        curr,
        keys=keys,
        steps=steps,
        q_estimate_fn=mock_q_estimate_fn,
        schedule=mock_schedule,
        explore=mock_explore,
        flow=None,
        train=False,
    )

    actions_train, _qs, exploration_val_train, _estimate_keys = (
        router.run_router(
            prev,
            curr,
            keys=keys,
            steps=steps,
            q_estimate_fn=mock_q_estimate_fn,
            schedule=mock_schedule,
            explore=mock_explore,
            flow=None,
            train=True,
        )
    )

    expected_greedy = jnp.array([1, 0], dtype=jnp.int32)
    assert jnp.array_equal(actions_eval, expected_greedy), (
        f"Expected greedy actions {expected_greedy} during eval, "
        f"got {actions_eval}"
    )
    assert jnp.all(exploration_val_eval == 0.0), (
        f"Expected all-zero exploration at eval, got {exploration_val_eval}"
    )
    assert jnp.all(exploration_val_train > 0.0), (
        f"Expected positive exploration during train, got "
        f"{exploration_val_train}"
    )
    assert jnp.all(actions_train == 2), (
        f"Expected exploration-driven actions during train, got {actions_train}"
    )


def test_run_router_uses_schedule():
    """Test that run_router uses schedule to compute exploration values."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))
    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    # Create keys
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    steps = trainable_state.extras["steps"]

    def mock_q_estimate_fn(prev, curr, flow=None):
        return jnp.ones((prev.shape[0], num_estimators), dtype=jnp.float32)

    expected_exploration = 0.75

    def mock_schedule(step):
        return expected_exploration

    def mock_explore(qs, exploration_val, key):
        return jnp.zeros(qs.shape[0], dtype=jnp.int32)

    _actions, _qs, exploration_val, _estimate_keys = router.run_router(
        prev,
        curr,
        keys=keys,
        steps=steps,
        q_estimate_fn=mock_q_estimate_fn,
        schedule=mock_schedule,
        explore=mock_explore,
        flow=None,
        train=True,
    )

    assert jnp.allclose(exploration_val, expected_exploration), (
        f"Expected exploration {expected_exploration}, got {exploration_val}"
    )


def test_run_router_uses_per_sample_steps_for_1d_input():
    """Test that 1D per-sample steps produce per-sample exploration values."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    steps = jnp.array([1, 5], dtype=jnp.int32)

    def mock_q_estimate_fn(prev, curr, flow=None):
        return jnp.ones((prev.shape[0], num_estimators), dtype=jnp.float32)

    def mock_schedule(step):
        return step.astype(jnp.float32) / 10.0

    def mock_explore(qs, exploration_val, key):
        return jnp.zeros(qs.shape[0], dtype=jnp.int32)

    _actions, _qs, exploration_val, _estimate_keys = router.run_router(
        prev,
        curr,
        keys=keys,
        steps=steps,
        q_estimate_fn=mock_q_estimate_fn,
        schedule=mock_schedule,
        explore=mock_explore,
        flow=None,
        train=True,
    )

    expected = jnp.array([0.1, 0.5], dtype=jnp.float32)
    assert jnp.allclose(exploration_val, expected), (
        f"Expected per-sample exploration {expected}, got {exploration_val}"
    )


def test_run_router_returns_correct_shapes():
    """Test that run_router returns correct shapes for all outputs."""
    batch_size = 2
    h, w = 4, 4
    num_estimators = 3

    prev = jnp.ones((batch_size, h, w))
    curr = jnp.ones((batch_size, h, w))
    trainable_state = make_mock_trainable_state(batch_size, num_estimators)

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    steps = trainable_state.extras["steps"]

    def mock_q_estimate_fn(prev, curr, flow=None):
        return jnp.ones((prev.shape[0], num_estimators), dtype=jnp.float32)

    def mock_schedule(step):
        return 0.5

    def mock_explore(qs, exploration_val, key):
        return jnp.zeros(qs.shape[0], dtype=jnp.int32)

    actions, qs, exploration_val, estimate_keys = router.run_router(
        prev,
        curr,
        keys=keys,
        steps=steps,
        q_estimate_fn=mock_q_estimate_fn,
        schedule=mock_schedule,
        explore=mock_explore,
        flow=None,
        train=True,
    )

    assert actions.shape == (batch_size,), (
        f"Expected actions shape ({batch_size},), got {actions.shape}"
    )
    assert qs.shape == (batch_size, num_estimators), (
        f"Expected qs shape ({batch_size}, {num_estimators}), got {qs.shape}"
    )
    assert exploration_val.shape == (batch_size,), (
        f"Expected exploration_val shape ({batch_size},), got "
        f"{exploration_val.shape}"
    )
    assert estimate_keys.shape == (batch_size, 2), (
        f"Expected estimate_keys shape ({batch_size}, 2), got "
        f"{estimate_keys.shape}"
    )
