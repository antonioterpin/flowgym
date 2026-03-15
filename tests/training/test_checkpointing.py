"""Tests for model checkpointing"""

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.core import FrozenDict

from flowgym.common.base.estimator import Estimator
from flowgym.common.base.trainable_state import NNEstimatorTrainableState
from flowgym.make import load_model, make_manager, save_model
from flowgym.training.optimizer import build_optimizer_from_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def assert_trees_allclose(a, b):
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, atol=1e-5),
        a,
        b,
    )


class MockEstimator(Estimator):
    """Minimal Mock Estimator for testing save/load with config."""

    def __init__(self, optimizer_config=None):
        self.optimizer_config = optimizer_config
        super().__init__(optimizer_config=optimizer_config)

    def create_state(self, *args, **kwargs):
        pass

    def create_train_step(self, *args, **kwargs):
        pass

    def compute_estimate_fn(self, *args, **kwargs):
        pass

    def _estimate(self, *args, **kwargs):
        pass

    def process_metrics(self, metrics):
        return metrics


# ---------------------------------------------------------------------------
# 1) Resume training: save -> load -> continue
# ---------------------------------------------------------------------------


def test_checkpoint_resume_roundtrip(clean_tmp_path):
    tmp_path = clean_tmp_path
    params = make_simple_params()
    tx = optax.sgd(learning_rate=0.1)
    extras = {"alpha": jnp.array(0.5)}

    # Initial state
    state0 = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=extras,
    )

    # Do one training step -> this is the state we will checkpoint
    grads1 = FrozenDict(
        {
            "w": jnp.array(1.0, dtype=jnp.float32),
            "b": jnp.array(-0.5, dtype=jnp.float32),
        }
    )
    state1 = state0.apply_gradients(grads=grads1)  # step should be 1
    step_to_save = int(state1.step)
    model_name = "DummyEstimator"

    # Save checkpoint
    ckpt_dir = save_model(
        state=state1,
        out_dir=tmp_path,
        step=step_to_save,
        model_name=model_name,
    )

    ckpt_dir = Path(ckpt_dir)

    time.sleep(1)
    assert ckpt_dir.exists()

    # Do another step to prove we're not using latest in-memory state
    grads2 = FrozenDict(
        {
            "w": jnp.array(-0.2, dtype=jnp.float32),
            "b": jnp.array(0.3, dtype=jnp.float32),
        }
    )
    state2 = state1.apply_gradients(grads=grads2)  # step should be 2

    # Build a template state with *same structure* but fresh contents.
    # This provides apply_fn, tx, and PyTree structure for Orbax.
    template_state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=state0.params,  # same shapes
        tx=tx,
        extras=extras,
    )

    # Restore from checkpoint
    restored_state = load_model(
        ckpt_dir=ckpt_dir,
        template_state=template_state,
        mode="resume",
    )

    # 1) Restored matches the saved state (state1), not state0 or state2
    res_any: Any = restored_state
    assert int(restored_state.step) == int(state1.step)
    assert_trees_allclose(restored_state.params, state1.params)
    assert_trees_allclose(res_any.opt_state, state1.opt_state)
    assert restored_state.extras == state1.extras

    # 2) Static fields should come from the template (not serialized)
    assert restored_state.apply_fn is template_state.apply_fn
    assert res_any.tx is template_state.tx

    # 3) Training can continue from restored_state and matches continuing
    # from state1
    resumed_after = restored_state.apply_gradients(grads=grads2)
    after_any: Any = resumed_after
    assert int(resumed_after.step) == int(state2.step)
    assert_trees_allclose(resumed_after.params, state2.params)
    assert_trees_allclose(after_any.opt_state, state2.opt_state)


# ---------------------------------------------------------------------------
# 2) Fine-tuning: load params only, new optimizer
# ---------------------------------------------------------------------------


def test_checkpoint_finetune_with_new_optimizer(clean_tmp_path):
    tmp_path = clean_tmp_path
    params = make_simple_params()
    tx_old = optax.sgd(learning_rate=0.1)
    extras = {"alpha": jnp.array(0.5)}

    # Initial state with "old" optimizer
    state0 = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx_old,
        extras=extras,
    )

    # Do one step -> this is what we checkpoint
    grads = FrozenDict(
        {
            "w": jnp.array(1.0, dtype=jnp.float32),
            "b": jnp.array(1.0, dtype=jnp.float32),
        }
    )
    trained_state = state0.apply_gradients(grads=grads)
    step_to_save = int(trained_state.step)
    model_name = "DummyEstimator"

    # Save checkpoint
    save_model(
        state=trained_state,
        out_dir=tmp_path,
        step=step_to_save,
        model_name=model_name,
    )
    ckpt_dir = tmp_path / "checkpoints" / model_name / str(step_to_save)
    time.sleep(1)
    assert ckpt_dir.exists()

    tx_new = optax.sgd(learning_rate=0.5)  # different LR to test behavior
    # Template state can use *any* optimizer; it's only for structure.
    template_state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=state0.params,
        tx=tx_new,
        extras=extras,
    )

    # Load only parameters from checkpoint
    restored_state = load_model(
        ckpt_dir=ckpt_dir,
        template_state=template_state,
        mode="params_only",
    )

    # Params should match the trained state's params
    assert_trees_allclose(restored_state.params, trained_state.params)

    # Now start fine-tuning with a *different* optimizer
    finetune_state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=restored_state.params,
        tx=tx_new,
        extras=extras,
    )

    # 1) Fine-tune state should start from the loaded params
    assert_trees_allclose(finetune_state.params, restored_state.params)

    # 2) Step should be reset (fresh optimizer)
    assert int(finetune_state.step) == 0

    # 3) opt_state should correspond to tx_new.init(restored_params)
    expected_opt_state = tx_new.init(restored_state.params)
    ft_any: Any = finetune_state
    assert_trees_allclose(ft_any.opt_state, expected_opt_state)

    # 4) Next step should use the *new* learning rate (0.5)
    grads_ft = FrozenDict(
        {
            "w": jnp.array(1.0, dtype=jnp.float32),
            "b": jnp.array(0.0, dtype=jnp.float32),
        }
    )
    after_ft = finetune_state.apply_gradients(grads=grads_ft)

    # Check one coordinate explicitly to see the LR effect
    # new_w = restored_w - lr_new * grad
    expected_w = restored_state.params["w"] - 0.5 * grads_ft["w"]
    assert jnp.allclose(after_ft.params["w"], expected_w)


def test_finetune_from_params_only_checkpoint_structure_mismatch(
    clean_tmp_path,
):
    """Reproduce RAFT-like bug: params-only checkpoint vs full state template.

    We:
      1) Save checkpoint with *only* params (no TrainState structure).
      2) Build full NNEstimatorTrainableState template
         (params+opt_state+extras).
      3) Load with load_model(..., mode='params_only'), which currently still
         builds abstract tree from *full* template_state and hands it to Orbax.
      4) Orbax complains that tree structure on disk (params-only) and the
         requested tree (full state) do not match -> ValueError, as in
         RAFT fine-tune run.
    """
    tmp_path = clean_tmp_path
    # 1) Create some params
    params = make_simple_params()

    # 2) Save a params-only checkpoint using Orbax in a step '0' directory
    ckpt_root = (tmp_path / "original").absolute()
    with make_manager(ckpt_root, keep=1) as mngr:
        mngr.save(
            0,
            args=ocp.args.Composite(
                state=ocp.args.PyTreeSave({"params": params})
            ),
        )
        mngr.wait_until_finished()

    # We pass the ROOT to load_model so it finds step 0
    ckpt_to_load = ckpt_root

    # 3) Build a full trainable-state template (params + opt_state + extras)
    tx = optax.sgd(learning_rate=0.1)
    template_state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=None,
    )

    # 4) Try to fine-tune: mode='params_only' but template_state is a full
    #    NNEstimatorTrainableState. With partial_restore=True, this now SUCCEEDS
    #    and restores what it can (params).
    restored_state = load_model(
        ckpt_dir=ckpt_to_load,
        template_state=template_state,
        mode="params_only",
    )
    assert_trees_allclose(restored_state.params, params)


def test_finetune_from_params_only_checkpoint_works(clean_tmp_path):
    tmp_path = clean_tmp_path

    # 1) Create some params
    params = make_simple_params()

    # 2) Save a params-only checkpoint using Orbax in a step '0' directory
    ckpt_root = (tmp_path / "original_ok").absolute()
    with make_manager(ckpt_root, keep=1) as mngr:
        mngr.save(
            0,
            args=ocp.args.Composite(
                state=ocp.args.PyTreeSave({"params": params})
            ),
        )
        mngr.wait_until_finished()

    # 3) Build a full trainable-state template (params + opt_state + extras)
    tx = optax.sgd(learning_rate=0.1)
    template_state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=params,
        tx=tx,
        extras=None,
    )

    # 4) Fine-tune: mode='params_only' should now succeed and give us params
    restored_state = load_model(
        ckpt_dir=ckpt_root,
        template_state=template_state,
        mode="params_only",
    )

    assert_trees_allclose(restored_state.params, params)

    # And we can create a new trainable state from them with a new optimizer
    tx_new = optax.sgd(learning_rate=0.5)
    finetune_state = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn,
        params=restored_state.params,
        tx=tx_new,
        extras=None,
    )
    assert_trees_allclose(finetune_state.params, restored_state.params)


def test_checkpoint_robust_optimizer_override(clean_tmp_path):
    """Verify checkpointed opt_config overrides template with mismatch."""
    tmp_path = clean_tmp_path
    params = make_simple_params()

    # 1) Save with Basic Adam
    adam_config = {"name": "adam", "learning_rate": 1e-3}
    tx_adam = build_optimizer_from_config(adam_config)
    model_adam = MockEstimator(optimizer_config=adam_config)

    state_save = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn, params=params, tx=tx_adam, extras=None
    )

    save_model(
        state=state_save,
        out_dir=tmp_path,
        model=model_adam,
        model_name="RobustTest",
        step=1,
    )
    ckpt_dir = tmp_path / "checkpoints" / "RobustTest" / "1"

    # 2) Restore using a template with EMA (which adds significant state/leaves)
    ema_config = {
        "name": "adam",
        "learning_rate": 1e-3,
        "chain": [{"name": "ema", "kwargs": {"decay": 0.99}}],
    }
    tx_ema = build_optimizer_from_config(ema_config)

    template_ema = NNEstimatorTrainableState.create(
        apply_fn=simple_apply_fn, params=params, tx=tx_ema, extras=None
    )

    # Verify leaf count difference
    template_any: Any = template_ema
    save_any: Any = state_save
    leaves_ema = len(jax.tree_util.tree_leaves(template_any.opt_state))
    leaves_adam = len(jax.tree_util.tree_leaves(save_any.opt_state))
    assert leaves_ema > leaves_adam, "EMA should have more leaves"

    # 3) Restore should override EMA with Adam from checkpoint
    restored = load_model(
        ckpt_dir=ckpt_dir, template_state=template_ema, mode="resume"
    )

    restored_any: Any = restored
    leaves_restored = len(jax.tree_util.tree_leaves(restored_any.opt_state))
    assert leaves_restored == leaves_adam, (
        f"Restored should have {leaves_adam} leaves (Adam), "
        f"got {leaves_restored}"
    )

    # Verify it still works for a step
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    _ = restored.apply_gradients(grads=grads)
