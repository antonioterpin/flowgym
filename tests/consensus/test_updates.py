"""Test closed-form flow updates against numerical optimization."""

import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
import pytest

from flowgym.flow.consensus.objectives import flows_objective, weights_and_anchors
from flowgym.flow.consensus.solvers import (
    closed_form_flows_huber,
    closed_form_flows_l1,
    closed_form_flows_l2,
    optax_solve,
)


def _make_problem(seed: int = 0):
    key = jax.random.PRNGKey(seed)
    k_w, k_a, k_z, k_d, k_x = jax.random.split(key, 5)

    n_agents, height, width, channels = 3, 4, 5, 2
    rho = 1.7

    weights = jax.random.uniform(
        k_w, (n_agents, height, width), minval=0.1, maxval=3.0
    )
    weights = weights.at[0, 0, 0].set(0.0)

    anchor_flows = jax.random.normal(k_a, (n_agents, height, width, channels))
    consensus_flow = jax.random.normal(k_z, (height, width, channels))
    consensus_dual = jax.random.normal(
        k_d, (n_agents, height, width, channels)
    )
    x0 = jax.random.normal(k_x, (n_agents, height, width, channels))

    weights_and_anchors_fn = functools.partial(
        weights_and_anchors, anchor_flows, weights
    )

    return {
        "rho": rho,
        "weights": weights,
        "anchor_flows": anchor_flows,
        "consensus_flow": consensus_flow,
        "consensus_dual": consensus_dual,
        "x0": x0,
        "weights_and_anchors_fn": weights_and_anchors_fn,
    }


def _make_objective(problem: dict, objective_type: str):
    return functools.partial(
        flows_objective,
        consensus_flow=problem["consensus_flow"],
        consensus_dual=problem["consensus_dual"],
        initial_flows=problem["anchor_flows"],
        weights=problem["weights"],
        objective_type=objective_type,
        rho=problem["rho"],
    )


def _solve_with_optax(
    x0: jnp.ndarray,
    objective_fn,
    *,
    learning_rate: float,
    num_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_opt, losses = optax_solve(
        x0,
        objective_fn,
        optax.adam(learning_rate=learning_rate),
        num_steps,
    )
    assert float(losses[-1]) < float(losses[0])
    return x_opt, losses


@pytest.mark.parametrize(
    (
        "objective_type",
        "closed_form_solver",
    ),
    [
        pytest.param(
            "l2",
            closed_form_flows_l2,
            id="l2",
        ),
        pytest.param(
            "l1",
            closed_form_flows_l1,
            id="l1",
        ),
        pytest.param(
            "huber",
            closed_form_flows_huber,
            id="huber",
        ),
    ],
)
def test_closed_form_flows_match_optax(
    objective_type: str,
    closed_form_solver,
    seed: int = 0,
    learning_rate: float = 0.00001,
    num_steps: int = 1000000,
    state_rtol: float = 1e-3,
    state_atol: float = 1e-3,
    objective_atol: float = 1e-3,
):
    problem = _make_problem(seed=seed)
    objective = _make_objective(problem, objective_type)
    

    x_closed = closed_form_solver(
        problem["x0"],
        problem["consensus_flow"],
        problem["consensus_dual"],
        problem["weights_and_anchors_fn"],
        problem["rho"],
        None,
    )

    x_opt, _ = _solve_with_optax(
        problem["x0"],
        objective,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )

    max_abs_diff = float(jnp.max(jnp.abs(x_closed - x_opt)))
    obj_closed = float(objective(x_closed))
    obj_opt = float(objective(x_opt))

    assert jnp.allclose(
        x_closed, x_opt, rtol=state_rtol, atol=state_atol
    ), (
        f"{objective_type}: state mismatch. "
        f"max_abs_diff={max_abs_diff:.3e}, "
        f"obj_closed={obj_closed:.12e}, obj_opt={obj_opt:.12e}"
    )

    assert abs(obj_closed - obj_opt) <= objective_atol, (
        f"{objective_type}: objective mismatch. "
        f"|obj_closed - obj_opt|={abs(obj_closed - obj_opt):.3e}"
    )