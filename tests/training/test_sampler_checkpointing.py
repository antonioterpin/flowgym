"""Tests for sampler checkpointing"""

import pathlib

import jax.numpy as jnp
import orbax.checkpoint as ocp

from flowgym.environment.fluid_env import FluidEnv
from flowgym.make import save_model


class DummyState:
    def __init__(self, step, params, opt_state, extras):
        self.step = step
        self.params = params
        self.opt_state = opt_state
        self.extras = extras


class DummySampler:
    def __init__(self, state, grain_iterator=None):
        self.state = state
        self.grain_iterator = grain_iterator


class MockManager:
    def __init__(self, ckpt_dir, options=None):
        self.ckpt_dir = ckpt_dir
        self.options = options
        self.save_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def save(self, step, args):
        self.save_calls.append({"step": step, "args": args})

    def wait_until_finished(self):
        pass


def test_save_model_is_atomic(monkeypatch):
    """Test that save_model is atomic"""
    manager_instances = []

    def mock_init(ckpt_dir, options=None):
        inst = MockManager(ckpt_dir, options)
        manager_instances.append(inst)
        return inst

    monkeypatch.setattr("orbax.checkpoint.CheckpointManager", mock_init)

    state = DummyState(
        step=10, params={"w": jnp.array(1.0)}, opt_state={}, extras={}
    )

    sampler = DummySampler(
        state={"foo": "bar"},
        grain_iterator={"iter_state": 123},  # Dummy grain iter
    )

    out_dir = "/tmp/test_ckpt"

    save_model(
        state=state,
        out_dir=out_dir,
        step=10,
        sampler=sampler,
        model_name="TestModel",
    )

    assert len(manager_instances) == 1
    mngr = manager_instances[0]

    expected_path = pathlib.Path(
        "/tmp/test_ckpt/checkpoints/TestModel"
    ).resolve()
    assert pathlib.Path(mngr.ckpt_dir).resolve() == expected_path

    assert len(mngr.save_calls) == 1
    call = mngr.save_calls[0]
    assert call["step"] == 10

    args = call["args"]
    assert isinstance(args, ocp.args.Composite)
    assert "state" in args._items
    assert "sampler" in args._items
    assert "grain" in args._items


def test_save_model_skips_non_grain_sampler(monkeypatch):
    """Test that save_model skips non-grain samplers"""
    manager_instances = []

    def mock_init(ckpt_dir, options=None):
        inst = MockManager(ckpt_dir, options)
        manager_instances.append(inst)
        return inst

    monkeypatch.setattr("orbax.checkpoint.CheckpointManager", mock_init)

    state = DummyState(
        step=10, params={"w": jnp.array(1.0)}, opt_state={}, extras={}
    )

    sampler = DummySampler(state={"foo": "bar"}, grain_iterator=None)

    out_dir = "/tmp/test_ckpt_no_grain"

    save_model(
        state=state,
        out_dir=out_dir,
        step=10,
        sampler=sampler,
        model_name="TestModelNoGrain",
    )

    mngr = manager_instances[0]
    assert len(mngr.save_calls) == 1
    args = mngr.save_calls[0]["args"]
    assert "state" in args._items
    assert "sampler" not in args._items
    assert "grain" not in args._items


def test_fluid_env_make_passes_load_from(monkeypatch):
    """Test that FluidEnv.make passes load_from to synthpix.make"""
    make_calls = []

    def mock_make(config, **kwargs):
        make_calls.append({"config": config, "kwargs": kwargs})
        return "dummy_sampler"

    monkeypatch.setattr("synthpix.make", mock_make)

    dataset_config = {
        "load_from": "/path/to/sampler_ckpt",
        "episode_length": 10,
        "gt_type": "flow",
    }

    FluidEnv.make(dataset_config)

    assert len(make_calls) == 1
    assert make_calls[0]["config"] == dataset_config
    assert make_calls[0]["kwargs"]["load_from"] == "/path/to/sampler_ckpt"
