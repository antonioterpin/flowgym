"""Tests for Art of PIV inference sub-estimator state routing."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.art_of_piv.inference import run_subestimators


def _build_inputs():
    batch_size, height, width = 2, 3, 4
    state = {
        "images": jnp.zeros((batch_size, 2, height, width), dtype=jnp.float32),
        "estimates": jnp.zeros(
            (batch_size, 2, height, width, 2), dtype=jnp.float32
        ),
        "keys": jnp.zeros((batch_size, 2), dtype=jnp.uint32),
    }
    curr = jnp.zeros((batch_size, height, width), dtype=jnp.float32)
    choices = jnp.asarray([0, 1], dtype=jnp.int32)
    qs = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    exploration_rate = jnp.zeros((batch_size,), dtype=jnp.float32)
    router_state = SimpleNamespace(
        extras={"steps": jnp.zeros((batch_size, 1), dtype=jnp.int32)}
    )
    return state, curr, choices, qs, exploration_rate, router_state


def _estimator_fn(images, state, ts, cache_payload=None):
    _, _, height, width, _ = state["estimates"].shape
    mark = ts["mark"]
    estimated = (
        state["estimates"]
        .at[:, -1]
        .set(jnp.ones((1, height, width, 2), dtype=jnp.float32) * mark)
    )
    return {**state, "estimates": estimated}, {}


def test_run_subestimators_uses_per_estimator_state():
    state, curr, choices, qs, exploration_rate, router_state = _build_inputs()
    estimator_fns = (_estimator_fn, _estimator_fn)
    estimator_states = (
        {"mark": jnp.asarray(3.0, dtype=jnp.float32)},
        {"mark": jnp.asarray(7.0, dtype=jnp.float32)},
    )

    flows, _, _ = run_subestimators(
        state,
        router_state,
        curr,
        choices,
        {},
        qs,
        exploration_rate,
        estimator_fns=estimator_fns,
        estimator_states=estimator_states,
    )

    np.testing.assert_allclose(
        np.asarray(flows[:, 0, 0, 0]),
        np.asarray([3.0, 7.0], dtype=np.float32),
    )


def test_run_subestimators_raises_on_mismatched_state_count():
    state, curr, choices, qs, exploration_rate, router_state = _build_inputs()

    with pytest.raises(
        ValueError,
        match="estimator_fns and estimator_states must have the same length",
    ):
        run_subestimators(
            state,
            router_state,
            curr,
            choices,
            {},
            qs,
            exploration_rate,
            estimator_fns=(_estimator_fn,),
            estimator_states=(
                {"mark": jnp.asarray(3.0, dtype=jnp.float32)},
                {"mark": jnp.asarray(7.0, dtype=jnp.float32)},
            ),
        )
