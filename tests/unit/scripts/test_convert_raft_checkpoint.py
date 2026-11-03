"""Tests for ``scripts/convert_raft_checkpoint.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.raft.raft_jax import RaftJaxEstimator
from flowgym.make import load_model
from flowgym.nn.raft_torch_to_jax import convert_raft_torch_state_dict

torch = pytest.importorskip("torch")

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "convert_raft_checkpoint.py"
)
_spec = importlib.util.spec_from_file_location(
    "convert_raft_checkpoint", _SCRIPT
)
assert _spec is not None and _spec.loader is not None
convert_raft_checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_raft_checkpoint)


@pytest.mark.slow
def test_raft32_round_trip(tmp_path, monkeypatch):
    """The saved Orbax tree loads back to the converted torch weights."""
    from flowgym.nn.raft_torch_nn.flowNetsRAFT import RAFT

    torch.manual_seed(0)
    state_dict = RAFT().state_dict()
    ckpt = tmp_path / "raft32.pt"
    torch.save({"model_state_dict": state_dict, "epoch": 3}, ckpt)
    out = tmp_path / "artifact"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--variant",
            "raft32",
            "--ckpt",
            str(ckpt),
            "--out-root",
            str(out),
        ],
    )

    convert_raft_checkpoint.main()

    estimator = RaftJaxEstimator(train=False)
    template = estimator.create_trainable_state(
        jnp.zeros((1, 32, 32)), jax.random.PRNGKey(1)
    )
    restored = load_model(
        out / "jax" / "checkpoints" / "0", template, mode="params_only"
    )
    expected = convert_raft_torch_state_dict(state_dict)
    for a, b in zip(
        jax.tree_util.tree_leaves(restored.params),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    assert (out / "pytorch" / "raft32.pt").exists()
    assert json.loads((out / "metadata.json").read_text())["source_epoch"] == 3
