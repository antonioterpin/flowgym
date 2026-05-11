"""Unit tests for the training-package config-driven builders."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from flowgym.training import optimizer as optimizer_mod
from flowgym.training.exploration import build_exploration_from_config
from flowgym.training.optimizer import build_optimizer_from_config


def test_build_optimizer_from_config_none_defaults_to_set_to_zero(monkeypatch):
    """`None` config defaults to set_to_zero and emits a warning."""
    calls: list[str] = []
    monkeypatch.setattr(
        optimizer_mod.logger,
        "warning",
        lambda msg, *a, **kw: calls.append(msg),
    )

    tx = build_optimizer_from_config(None)

    assert isinstance(tx, optax.GradientTransformation)
    params = jnp.array([1.0, 2.0])
    state = tx.init(params)
    updates, _ = tx.update(jnp.array([5.0, -5.0]), state, params)
    assert jnp.all(updates == 0.0)

    assert len(calls) == 1
    assert "set_to_zero" in calls[0]


def test_build_exploration_from_config_normalizes_name():
    """Mixed-case name values resolve via the same str().lower() rule."""
    policy = build_exploration_from_config({"name": "Epsilon_Greedy"})
    actions = policy(
        jnp.array([[0.0, 1.0]]),
        jnp.array([0.0]),
        jax.random.PRNGKey(0),
    )
    assert actions.shape == (1,)


def test_build_exploration_from_config_rejects_non_dict():
    """Type error reports the offending type, not the full value."""
    with pytest.raises(TypeError, match=r"got list\."):
        build_exploration_from_config([("name", "epsilon_greedy")])  # type: ignore[arg-type]
