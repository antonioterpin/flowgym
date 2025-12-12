# tests/test_trainable_state.py

import os
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
import optax
import pytest
from flax.core import FrozenDict

from flowgym.common.base.trainable_states import (
    EstimatorTrainableState,
    NNEstimatorTrainableState
)
from flowgym.make import (
    save_model,
    load_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trees_allclose(tree_a: Any, tree_b: Any, atol: float = 1e-6, rtol: float = 1e-6):
    """Assert that two PyTrees of arrays are equal up to tolerance."""
    leaves_a, treedef_a = tree_util.tree_flatten(tree_a)
    leaves_b, treedef_b = tree_util.tree_flatten(tree_b)
    assert treedef_a == treedef_b, "PyTree structures differ"
    assert len(leaves_a) == len(leaves_b), "Leaf counts differ"

    for a, b in zip(leaves_a, leaves_b):
        if isinstance(a, (jnp.ndarray, np.ndarray)):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)
        else:
            assert a == b


def _make_dummy_state(
    *,
    tx: optax.GradientTransformation | None = None,
    global_step: int = 0,
) -> NNEstimatorTrainableState:
    """Utility to build a simple NNEstimatorTrainableState for tests."""
    if tx is None:
        tx = optax.adam(optax.linear_schedule(init_value=1, end_value=0, transition_steps=2))

    params = FrozenDict({"w": jnp.array([1.0, 2.0], dtype=jnp.float32)})
    extras = FrozenDict({"global_step": jnp.array(global_step, dtype=jnp.int32)})
    opt_state = tx.init(params)
    return NNEstimatorTrainableState(
        params=params,
        opt_state=opt_state,
        tx=tx,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# EstimatorTrainableState (base)
# ---------------------------------------------------------------------------

def test_estimator_trainable_state_is_trivial_pytree():
    state = EstimatorTrainableState()
    children, aux = state.tree_flatten()
    assert children == ()
    assert aux is None

    restored = EstimatorTrainableState.tree_unflatten(aux, children)
    assert isinstance(restored, EstimatorTrainableState)


# ---------------------------------------------------------------------------
# NNEstimatorTrainableState core behaviour
# ---------------------------------------------------------------------------

def test_nn_estimator_create_initializes_opt_state_and_extras():
    params = FrozenDict({"w": jnp.ones((2,), dtype=jnp.float32)})
    tx = optax.sgd(learning_rate=0.1)
    extras = {"global_step": jnp.array(0, dtype=jnp.int32)}

    state = NNEstimatorTrainableState.create(params=params, tx=tx, extras=extras)

    assert isinstance(state, NNEstimatorTrainableState)
    assert isinstance(state.params, FrozenDict)
    assert "w" in state.params
    w = state.params["w"]
    assert isinstance(w, jnp.ndarray)
    assert w.dtype == jnp.float32
    assert w.shape == (2,)

    # opt_state should be initialized and non-empty
    assert state.opt_state is not None

    # extras should be a FrozenDict with the provided field
    assert isinstance(state.extras, FrozenDict)
    assert "global_step" in state.extras
    assert state.extras["global_step"] == extras["global_step"]


def test_nn_estimator_tree_flatten_unflatten_roundtrip():
    state = _make_dummy_state(global_step=3)
    children, aux = state.tree_flatten()

    assert len(children) == 3  # params, opt_state, extras
    assert isinstance(aux, optax.GradientTransformation)

    restored = NNEstimatorTrainableState.tree_unflatten(aux, children)
    assert isinstance(restored, NNEstimatorTrainableState)

    _trees_allclose(restored.params, state.params)
    _trees_allclose(restored.opt_state, state.opt_state)
    _trees_allclose(restored.extras, state.extras)
    # tx should be the same object we passed as aux
    assert restored.tx is aux


def test_apply_gradients_updates_params_and_preserves_tx_and_extras():
    # Simple 1D param with SGD
    params = FrozenDict({"w": jnp.array(1.0)})
    tx = optax.sgd(learning_rate=0.1)
    extras = {"global_step": jnp.array(5, dtype=jnp.int32)}
    state = NNEstimatorTrainableState.create(params=params, tx=tx, extras=extras)

    grads = FrozenDict({"w": jnp.array(2.0)})  # dL/dw = 2

    new_state = state.apply_gradients(grads)

    # SGD: w_new = w - lr * grad = 1.0 - 0.1 * 2.0 = 0.8
    assert jnp.allclose(new_state.params["w"], jnp.array(0.8)) # type: ignore

    # tx should be preserved (same object)
    assert new_state.tx is state.tx

    # extras should be preserved (same content and object identity)
    assert new_state.extras == state.extras
    assert new_state.extras is state.extras

    # We *don't* assert anything about opt_state here, because for SGD it is EmptyState()
    # and has no leaves, so there's nothing to change.



# ---------------------------------------------------------------------------
# save_model / load_model: resumability and module-name independence
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip_resumable(tmp_path):
    """Round-trip through save_model/load_model preserves dynamic state."""

    out_dir = tmp_path
    model_name = "dummy_model"
    ckpt_path = out_dir / model_name

    # Original state we want to resume later
    original_state = _make_dummy_state(global_step=7)

    # Save
    save_model(original_state, out_dir=str(out_dir), model_name=model_name)
    assert ckpt_path.exists(), "Checkpoint directory should be created"

    # Build a fresh template_state (as if in a new process)
    template_state = _make_dummy_state(global_step=0)

    # Load
    loaded_state = load_model(str(ckpt_path), template_state)
    
    assert isinstance(loaded_state, NNEstimatorTrainableState)

    # Params and opt_state and extras should match the original
    _trees_allclose(loaded_state.params, original_state.params)
    _trees_allclose(loaded_state.opt_state, original_state.opt_state)
    _trees_allclose(loaded_state.extras, original_state.extras)


def test_save_and_load_supports_training_resumption(tmp_path):
    """Simulate training, save, load, continue training with identical results."""
    out_dir = tmp_path
    model_name = "resumable_model"
    ckpt_path = out_dir / model_name

    # Initial state
    init_state = _make_dummy_state(global_step=0)

    # Fake gradients for two steps
    grads1 = FrozenDict({"w": jnp.array([0.3, -0.2], dtype=jnp.float32)})
    grads2 = FrozenDict({"w": jnp.array([-0.1, 0.4], dtype=jnp.float32)})

    # Path A: train for 2 steps continuously
    s1 = init_state.apply_gradients(grads1)
    s2_direct = s1.apply_gradients(grads2)

    # Path B: train 1 step, save, load, train another step
    s1_b = init_state.apply_gradients(grads1)
    save_model(s1_b, out_dir=str(out_dir), model_name=model_name)
    assert ckpt_path.exists()

    # Fresh template to simulate new process
    template_state = _make_dummy_state(global_step=0)
    s1_loaded = load_model(str(ckpt_path), template_state)

    assert isinstance(s1_loaded, NNEstimatorTrainableState)

    s2_resumed = s1_loaded.apply_gradients(grads2)

    # Final states should be numerically identical
    _trees_allclose(s2_direct.params, s2_resumed.params)
    _trees_allclose(s2_direct.opt_state, s2_resumed.opt_state)
    _trees_allclose(s2_direct.extras, s2_resumed.extras)


def test_load_model_uses_template_static_tx_not_checkpoint(tmp_path):
    """Static tx must come from the template, not from the checkpoint.

    This ensures checkpoints are data-only (params/opt_state/extras) and
    do not depend on pickled optimizer/transformation objects, which would
    tie them to module/class names.
    """
    out_dir = tmp_path
    model_name = "dummy_model"
    ckpt_path = out_dir / model_name

    # First state uses tx1 (e.g., Adam with lr=1e-3)
    tx1 = optax.adam(learning_rate=1e-3)
    original_state = _make_dummy_state(tx=tx1, global_step=10)

    save_model(original_state, out_dir=str(out_dir), model_name=model_name)
    assert ckpt_path.exists()

    # Now build a template with *different* tx (e.g., different LR)
    tx2 = optax.adam(learning_rate=5e-4)
    template_state = _make_dummy_state(tx=tx2, global_step=0)

    loaded_state = load_model(str(ckpt_path), template_state)
    
    assert isinstance(loaded_state, NNEstimatorTrainableState)

    # Dynamic pieces should match original
    _trees_allclose(loaded_state.params, original_state.params)
    _trees_allclose(loaded_state.opt_state, original_state.opt_state)
    _trees_allclose(loaded_state.extras, original_state.extras)

    # Static tx must be exactly the one from the template (tx2), proving
    # it is NOT coming from the checkpoint file.
    assert loaded_state.tx is tx2

def test_save_model_turns_relative_directory_into_absolute(tmp_path):
    """save_model should convert relative out_dir to absolute path."""
    relative_out_dir = "relative_dir"
    model_name = "test_model"
    ckpt_path = os.path.abspath(os.path.join(relative_out_dir, model_name))

    state = _make_dummy_state(global_step=0)

    # Save using relative path
    save_model(state, out_dir=relative_out_dir, model_name=model_name)

    # Check that the checkpoint directory was created at the absolute path
    assert os.path.exists(ckpt_path), "Checkpoint directory should be created at absolute path"

    # Clean up created directory
    if os.path.exists(relative_out_dir):
        import shutil
        shutil.rmtree(relative_out_dir)