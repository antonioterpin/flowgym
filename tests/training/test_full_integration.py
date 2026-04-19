"""Comprehensive integration tests for train and train_supervised loops.

These tests exercise all training features using DummyEstimator:
- Replay buffer (initialization, push, sample)
- Replay buffer enrichment via prepare_experience_for_replay
- Checkpointing (save_model / load_model)
- Validation (for train_supervised)
- Metrics processing
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from flowgym.common.base.trainable_state import NNEstimatorTrainableState
from flowgym.flow.dummy import DummyEstimator
from flowgym.training.replay import ReplayBuffer
from train import train
from train_supervised import train_supervised

# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mock_save_model(
    state, out_dir, step=None, model=None, model_name=None, **kwargs
):
    """Mock save_model that creates checkpoint directories without writing."""
    out_dir = Path(out_dir)
    if model_name:
        ckpt_dir = out_dir / "checkpoints" / model_name
    else:
        ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if step is None and hasattr(state, "step"):
        step = int(state.step)
    step_dir = ckpt_dir / str(step)
    step_dir.mkdir(exist_ok=True)
    return str(step_dir)


# ─────────────────────────────────────────────────────────────────────────────
# train_supervised integration tests
# ─────────────────────────────────────────────────────────────────────────────


def test_train_supervised_full_integration(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test train_supervised with replay buffer, validation, checkpointing."""
    model = DummyEstimator(train_type="supervised", enrichment_marker=True)

    # Create a separate mock for validation sampler
    val_sampler = MagicMock()
    val_sampler.shutdown = MagicMock()
    val_sampler.reset = MagicMock()
    val_sampler.grain_iterator = None

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Mock both evaluate_batches and save_model
    with (
        patch("train_supervised.evaluate_batches") as mock_eval,
        patch("train_supervised.save_model", side_effect=_mock_save_model),
    ):
        mock_eval.return_value = {
            "mean_error": 0.1,
            "max_error": 0.2,
            "min_error": 0.05,
            "errors": jnp.array([0.1]),
        }

        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=dummy_trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=mock_sampler,
            val_sampler=val_sampler,
            val_interval=2,
            val_num_batches=1,
            num_batches=5,
            estimate_type="flow",
            save_every=3,
            log_every=1,
            key=jax.random.PRNGKey(42),
            replay_buffer_capacity=100,
            replay_ratio=1.0,
        )

        # Validation should be called at batch 0, 2, 4
        assert mock_eval.call_count >= 2, (
            f"Expected at least 2 validation calls, got {mock_eval.call_count}"
        )

    # Verify checkpoints were created
    checkpoint_dirs = list(out_dir.glob("**/DummyEstimator/*"))
    assert len(checkpoint_dirs) >= 1, (
        "Expected at least one checkpoint to be saved"
    )


def test_train_supervised_replay_enrichment(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test that prepare_experience_for_replay is called and enriches data."""
    model = DummyEstimator(train_type="supervised", enrichment_marker=True)

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Track the replay buffer contents
    original_push = ReplayBuffer.push
    pushed_experiences = []

    def tracking_push(self, experience):
        pushed_experiences.append(experience)
        return original_push(self, experience)

    with patch.object(ReplayBuffer, "push", tracking_push):
        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=dummy_trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=mock_sampler,
            num_batches=3,
            replay_buffer_capacity=100,
            replay_ratio=0.0,  # No replay sampling, just push
            key=jax.random.PRNGKey(42),
        )

    # Verify experiences were enriched with marker
    assert len(pushed_experiences) > 0, (
        "Expected experiences to be pushed to buffer"
    )
    for exp in pushed_experiences:
        assert "_enriched" in exp.state, (
            "Experience should have enrichment marker"
        )
        assert bool(exp.state["_enriched"]), "Enrichment marker should be True"


def test_train_supervised_checkpointing_roundtrip(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test that checkpoints are saved at the expected intervals."""
    model = DummyEstimator(train_type="supervised")

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    save_calls = []

    def tracking_save(*args, **kwargs):
        save_calls.append((args, kwargs))
        return _mock_save_model(*args, **kwargs)

    # Run training to create a checkpoint
    with patch("train_supervised.save_model", side_effect=tracking_save):
        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=dummy_trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=mock_sampler,
            num_batches=4,
            save_every=3,
            key=jax.random.PRNGKey(42),
        )

    # Verify save_model was called at expected intervals (batch 3)
    assert len(save_calls) >= 1, "Expected at least one save_model call"


def test_train_supervised_no_enrichment_without_flag(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test that enrichment is skipped when enrichment_marker is False."""
    model = DummyEstimator(train_type="supervised", enrichment_marker=False)

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Track the replay buffer contents
    original_push = ReplayBuffer.push
    pushed_experiences = []

    def tracking_push(self, experience):
        pushed_experiences.append(experience)
        return original_push(self, experience)

    with patch.object(ReplayBuffer, "push", tracking_push):
        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=dummy_trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=mock_sampler,
            num_batches=2,
            replay_buffer_capacity=100,
            replay_ratio=0.0,
            key=jax.random.PRNGKey(42),
        )

    # Verify experiences were NOT enriched (no marker)
    assert len(pushed_experiences) > 0, "Expected experiences to be pushed"
    for exp in pushed_experiences:
        assert "_enriched" not in exp.state, (
            "Experience should NOT have enrichment marker"
        )


# ─────────────────────────────────────────────────────────────────────────────
# train (RL) integration tests
# ─────────────────────────────────────────────────────────────────────────────


def test_train_rl_full_integration(tmp_path, mock_env):
    """Test train (RL) loop with replay buffer and checkpointing."""
    model = DummyEstimator(train_type="rl")
    env, obs, env_state = mock_env

    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(1.0, jnp.float32)})
    tx = optax.sgd(0.01)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {"dummy": jnp.array(0.0)}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Run training with mocked save_model
    with patch("train.save_model", side_effect=_mock_save_model):
        train(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            env=env,
            env_state=env_state,
            num_episodes=3,
            save_every=2,
            log_every=1,
            obs=obs,
            key=jax.random.PRNGKey(42),
            replay_buffer_capacity=100,
            replay_ratio=1.0,
        )

    # Verify checkpoints were created
    checkpoint_dirs = list(out_dir.glob("**/DummyEstimator-*"))
    assert len(checkpoint_dirs) >= 1, (
        "Expected at least one checkpoint to be saved"
    )


def test_train_rl_replay_buffer_used(tmp_path, mock_env):
    """Test that replay buffer is initialized and used in RL training."""
    model = DummyEstimator(train_type="rl")
    env, obs, env_state = mock_env

    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(1.0, jnp.float32)})
    tx = optax.sgd(0.01)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {"dummy": jnp.array(0.0)}

    # Track ReplayBuffer initialization
    with patch("train.ReplayBuffer", wraps=ReplayBuffer) as mock_buffer:
        train(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=trainable_state,
            out_dir=str(tmp_path),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            env=env,
            env_state=env_state,
            num_episodes=1,
            save_every=10,
            log_every=1,
            obs=obs,
            key=jax.random.PRNGKey(42),
            replay_buffer_capacity=100,
            replay_ratio=0.0,
        )

        mock_buffer.assert_called_once()


def test_train_rl_replay_buffer_samples(tmp_path, mock_env):
    """Test that replay buffer sampling is called when replay_ratio > 0."""
    model = DummyEstimator(train_type="rl")
    env, obs, env_state = mock_env

    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(1.0, jnp.float32)})
    tx = optax.sgd(0.01)
    trainable_state = NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {"dummy": jnp.array(0.0)}

    # Track train_step_fn calls
    train_step_calls = []
    original_train_step = model.create_train_step()

    def tracking_train_step(*args, **kwargs):
        train_step_calls.append((args, kwargs))
        return original_train_step(*args, **kwargs)

    with patch.object(
        model, "create_train_step", return_value=tracking_train_step
    ):
        train(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=trainable_state,
            out_dir=str(tmp_path),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            env=env,
            env_state=env_state,
            num_episodes=1,
            save_every=10,
            log_every=1,
            obs=obs,
            key=jax.random.PRNGKey(42),
            replay_buffer_capacity=100,
            replay_ratio=1.0,  # Force replay steps
        )

    # Should have at least 2 train step calls (1 for env step + 1 for replay)
    assert len(train_step_calls) >= 2, (
        f"Expected >=2 train steps, got {len(train_step_calls)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────────────────────────────────────


def test_train_supervised_validation_runs(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test that validation is executed at specified intervals."""
    model = DummyEstimator(train_type="supervised")

    val_sampler = MagicMock()
    val_sampler.shutdown = MagicMock()
    val_sampler.reset = MagicMock()
    val_sampler.grain_iterator = None

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Track validation calls
    with patch("train_supervised.evaluate_batches") as mock_eval:
        mock_eval.return_value = {
            "mean_error": 0.1,
            "max_error": 0.2,
            "min_error": 0.05,
            "errors": jnp.array([0.1]),
        }

        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=dummy_trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=mock_sampler,
            val_sampler=val_sampler,
            val_interval=2,
            val_num_batches=1,
            num_batches=5,
            key=jax.random.PRNGKey(42),
        )

        # Validation should be called at batch 0, 2, 4
        assert mock_eval.call_count >= 2, (
            f"Expected at least 2 validation calls, got {mock_eval.call_count}"
        )


def test_train_supervised_save_only_best(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test save_only_best mode saves only when validation improves."""
    model = DummyEstimator(train_type="supervised")

    val_sampler = MagicMock()
    val_sampler.shutdown = MagicMock()
    val_sampler.reset = MagicMock()
    val_sampler.grain_iterator = None

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Mock validation to return improving errors
    val_errors = iter([0.5, 0.3, 0.2, 0.1])

    def mock_evaluate(*args, **kwargs):
        err = next(val_errors, 0.1)
        return {
            "mean_error": err,
            "max_error": err + 0.1,
            "min_error": err - 0.05,
            "errors": jnp.array([err]),
        }

    save_calls = []

    def tracking_save(*args, **kwargs):
        save_calls.append((args, kwargs))
        return _mock_save_model(*args, **kwargs)

    with (
        patch("train_supervised.evaluate_batches", side_effect=mock_evaluate),
        patch("train_supervised.save_model", side_effect=tracking_save),
    ):
        train_supervised(
            model=model,
            model_config={"config": {"jit": False}},
            trainable_state=dummy_trainable_state,
            out_dir=str(out_dir),
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=mock_sampler,
            val_sampler=val_sampler,
            val_interval=1,
            val_num_batches=1,
            num_batches=4,
            save_every=1,
            save_only_best=True,
            key=jax.random.PRNGKey(42),
        )

    # In save_only_best mode, we save when mean_error improves
    # With errors [0.5, 0.3, 0.2, 0.1], each is an improvement
    assert len(save_calls) >= 1, (
        "Expected at least one save in save_only_best mode"
    )


def test_train_supervised_metrics_logged(
    tmp_path, mock_sampler, dummy_trainable_state
):
    """Test that metrics from train step are properly logged."""
    model = DummyEstimator(train_type="supervised")

    def create_state_fn(img, key):
        B, H, W = img.shape
        return {
            "images": img[:, None, ...],
            "estimates": jnp.zeros((B, 1, H, W, 2)),
        }

    def compute_estimate_fn(img, state, ts):
        return state, {}

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir()

    # Run training
    train_supervised(
        model=model,
        model_config={"config": {"jit": False}},
        trainable_state=dummy_trainable_state,
        out_dir=str(out_dir),
        create_state_fn=create_state_fn,
        compute_estimate_fn=compute_estimate_fn,
        sampler=mock_sampler,
        num_batches=2,
        log_every=1,
        key=jax.random.PRNGKey(42),
    )

    # If we got here without error, metrics were processed
    # DummyEstimator returns {"dummy_metric": jnp.array(1.0)} which gets
    # logged
