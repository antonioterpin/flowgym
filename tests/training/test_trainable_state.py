"""Test TrainableState."""

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

import flowgym.common.base.trainable_state as ts_mod
from flowgym.common.base.trainable_state import NNEstimatorTrainableState


def simple_apply_fn(params: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
    """Minimal apply_fn: y = w * x + b (all scalar for simplicity)."""
    w = params["w"]
    b = params["b"]
    return w * x + b


def make_simple_params() -> FrozenDict:
    return FrozenDict(
        {
            "w": jnp.array(2.0, dtype=jnp.float32),
            "b": jnp.array(1.0, dtype=jnp.float32),
        }
    )


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_sets_extras_default_empty_frozendict():
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)

    state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=None,
    )

    assert isinstance(state.extras, FrozenDict)
    assert len(state.extras) == 0


def test_create_uses_given_extras():
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)
    extras = {"alpha": jnp.array(0.5)}

    state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=extras,
    )

    assert isinstance(state.extras, FrozenDict)
    assert "alpha" in state.extras
    assert jnp.allclose(state.extras["alpha"], jnp.array(0.5))


def test_create_initializes_opt_state_from_tx():
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)

    state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=None,
    )

    expected_opt_state = tx.init(params)
    # opt_state is a PyTree; compare structure and leaves
    jax.tree_util.tree_map(
        lambda a, b: jnp.allclose(a, b),
        state.opt_state,
        expected_opt_state,
    )

    # step should start at 0 (TrainState behavior)
    assert int(state.step) == 0


def test_apply_fn_is_used_correctly():
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)

    state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=None,
    )

    x = jnp.array(3.0, dtype=jnp.float32)
    y = state.apply_fn(state.params, x)
    # y = 2 * 3 + 1 = 7
    assert jnp.allclose(y, jnp.array(7.0))


# ---------------------------------------------------------------------------
# apply_gradients (inherited from TrainState)
# ---------------------------------------------------------------------------


def test_apply_gradients_updates_params_and_step_and_keeps_extras_and_tx():
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)
    extras = {"alpha": jnp.array(0.5)}

    state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=extras,
    )

    # gradient: dL/dw = 1, dL/db = 2 (arbitrary)
    grads = FrozenDict(
        {
            "w": jnp.array(1.0, dtype=jnp.float32),
            "b": jnp.array(2.0, dtype=jnp.float32),
        }
    )

    new_state = state.apply_gradients(grads=grads)

    # SGD: new_param = old_param - lr * grad
    assert jnp.allclose(new_state.params["w"], params["w"] - 0.1 * grads["w"])
    assert jnp.allclose(new_state.params["b"], params["b"] - 0.1 * grads["b"])

    # step should increment by 1
    assert int(state.step) == 0
    assert int(new_state.step) == 1

    # extras should be unchanged
    assert isinstance(new_state.extras, FrozenDict)
    assert "alpha" in new_state.extras
    assert jnp.allclose(new_state.extras["alpha"], jnp.array(0.5))

    # tx should be the same object (static field)
    assert new_state.tx is state.tx


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_uses_build_optimizer(monkeypatch):
    params = make_simple_params()

    # fake optimizer config
    optimizer_config = {"type": "sgd", "lr": 0.123}

    # track that build_optimizer_from_config is called with the right config
    captured = {}

    def fake_build_optimizer_from_config(cfg):
        captured["cfg"] = cfg
        return optax.sgd(learning_rate=cfg["lr"])

    # Monkeypatch the symbol used inside NNEstimatorTrainableState module
    monkeypatch.setattr(
        ts_mod,
        "build_optimizer_from_config",
        fake_build_optimizer_from_config,
    )

    state = NNEstimatorTrainableState.from_config(
        apply_fn=simple_apply_fn,
        params=params,
        optimizer_config=optimizer_config,
        extras=None,
    )

    # Ensure build_optimizer_from_config got the right config
    assert "cfg" in captured
    assert captured["cfg"] == optimizer_config

    # Check that tx looks like what we expect (learning rate propagated)
    assert isinstance(state.tx, optax.GradientTransformation)

    # Check gradient step matches SGD with lr=0.123
    grads = FrozenDict(
        {
            "w": jnp.array(1.0, dtype=jnp.float32),
            "b": jnp.array(1.0, dtype=jnp.float32),
        }
    )
    new_state = state.apply_gradients(grads=grads)
    assert jnp.allclose(
        new_state.params["w"], params["w"] - optimizer_config["lr"] * grads["w"]
    )
    assert jnp.allclose(
        new_state.params["b"], params["b"] - optimizer_config["lr"] * grads["b"]
    )


# ---------------------------------------------------------------------------
# PyTree / static-field behavior
# ---------------------------------------------------------------------------


def test_trainable_state_is_a_valid_pytree_and_tx_is_static():
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)
    extras = {"alpha": jnp.array(0.5)}

    state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=extras,
    )

    leaves, _ = jax.tree_util.tree_flatten(state)
    # We expect only array-ish leaves:
    # step, params leaves, opt_state leaves, extras leaves.
    # No functions, no GradientTransformation objects, etc.
    for leaf in leaves:
        # step: scalar/int, everything else should be array-like
        assert isinstance(leaf, (jnp.ndarray, int, float, bool)), (
            f"Non-array-like leaf found in pytree: {type(leaf)}"
        )

    # A tiny sanity check that 'params' content shows up somewhere in the leaves
    assert any(jnp.allclose(leaf, params["w"]) for leaf in leaves)
    assert any(jnp.allclose(leaf, params["b"]) for leaf in leaves)

    # Make sure tx itself is not a leaf
    assert tx not in leaves
