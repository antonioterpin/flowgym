"""Solver module for flow estimation."""

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import optax


def closed_form_flows_l2(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    weights_and_anchors_fn: Callable,
    rho: float,
    _: object,
) -> jnp.ndarray:
    """Closed form x-update for flows with a l2 penalty.

    Args:
        flows: Current flow estimates from different agents (unused).
        consensus_flow: Current consensus flow estimate (H, W, 2).
        consensus_dual: Dual variable for consensus (N, H, W, 2).
        weights_and_anchors_fn: Function that returns weights and
            anchor flows.
        rho: Penalty parameter for the consensus term.

    Returns:
        Updated flow estimates.
    """
    # Use the provided function to get weights and anchor flows
    weights, anchor_flows = weights_and_anchors_fn()

    # Expand weights to match flow dimensions
    weights = weights[..., None]

    # Compute the closed form solution for the x-update
    denominator = 2.0 * weights + rho  # shape (N, H, W, 1)
    numerator = 2.0 * weights * anchor_flows + rho * (
        consensus_flow[None, ...] - consensus_dual
    )

    return numerator / denominator


def closed_form_flows_l1(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    weights_and_anchors_fn: Callable,
    rho: float,
    _: object,
) -> jnp.ndarray:
    """Closed form x-update for flows with a l1 penalty.

    Args:
        flows: Current flow estimates from different agents (unused).
        consensus_flow: Current consensus flow estimate (H, W, 2).
        consensus_dual: Dual variable for consensus (N, H, W, 2).
        weights_and_anchors_fn: Function that returns weights and
            anchor flows.
        rho: Penalty parameter for the consensus term.

    Returns:
        Updated flow estimates.
    """
    # Use the provided function to get weights and anchor flows
    weights, anchor_flows = weights_and_anchors_fn()

    # Compute the soft thresholding
    v = consensus_flow[None, ...] - consensus_dual  # (N,H,W,2)
    tau = (weights / rho)[..., None]  # (N,H,W,1)

    # Compute the closed form solution for the x-update
    y = v - anchor_flows  # (N,H,W,2)
    soft = jnp.sign(y) * jnp.maximum(jnp.abs(y) - tau, 0.0)
    x = anchor_flows + soft
    return x


def optax_solve(
    params: optax.Params | jnp.ndarray,
    objective_fn: Callable[[optax.Params], jnp.ndarray],
    optimiser: optax.GradientTransformation,
    num_steps: int,
) -> tuple[optax.Params | jnp.ndarray, jnp.ndarray]:
    """Solve an optimization problem using Optax.

    Args:
        params: Initial parameters for optimization.
        objective_fn: Objective function to minimize.
        optimiser: Optax optimizer.
        num_steps: Number of optimization steps.

    Returns:
        Tuple of optimized parameters and loss values at each step.
    """
    value_and_grad_fn = jax.value_and_grad(objective_fn)
    opt_state = optimiser.init(params)

    def step_fn(state_and_params, _):
        opt_state, params = state_and_params
        loss, grads = value_and_grad_fn(params)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (opt_state, new_params), loss

    (_, optimized_params), losses = jax.lax.scan(
        step_fn, (opt_state, params), xs=None, length=num_steps
    )

    return optimized_params, losses


def optax_consensus(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    objective_fn: Callable,
    rho: float,
    num_steps: int,
    optimiser: optax.GradientTransformation,
) -> optax.Params | jnp.ndarray:
    """Optax solver for consensus variable.

    Args:
        flows: Current flow estimates from different agents.
            shape (N, H, W, 2) where B is the batch size, N is the
            number of estimates.
        consensus_flow: Current consensus flow estimate shape (H, W, 2).
        consensus_dual: Current dual variable for consensus
            shape (N, H, W, 2).
        objective_fn: Objective function for consensus variable.
        rho: Parameter associated with the augmented Lagrangian.
        num_steps: Number of optimization steps.
        optimiser: Optax optimizer.

    Returns:
        Updated consensus flow estimate.
    """
    # Freeze flows and consensus_dual in the objective function
    consensus_flow_fn = functools.partial(
        objective_fn, flows=flows, consensus_dual=consensus_dual, rho=rho
    )

    consensus, _ = optax_solve(
        consensus_flow, consensus_flow_fn, optimiser, num_steps
    )
    return consensus


def optax_flows(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    objective_fn: Callable,
    rho: float,
    num_steps: int,
    optimiser: optax.GradientTransformation,
) -> jnp.ndarray:
    """Optax solver for flow estimates.

    Args:
        flows: Current flow estimates from different agents.
            shape (N, H, W, 2) where B is the batch size, N is the
            number of estimates.
        consensus_flow: Current consensus flow estimate shape (H, W, 2).
        consensus_dual: Current dual variable for consensus
            shape (N, H, W, 2).
        objective_fn: Objective function for flow estimates.
        rho: Parameter associated with the augmented Lagrangian.
        num_steps: Number of optimization steps.
        optimiser: Optax optimizer.

    Returns:
        Updated flow estimates.
    """
    # Freeze consensus_flow and consensus_dual in the objective function
    flows_fn = functools.partial(
        objective_fn,
        consensus_flow=consensus_flow,
        consensus_dual=consensus_dual,
        rho=rho,
    )
    flows_result, _ = optax_solve(flows, flows_fn, optimiser, num_steps)
    flows = cast(jnp.ndarray, flows_result)
    return flows


SOLVER_FLOWS_FACTORY = {
    "sgd": lambda lr: functools.partial(
        optax_flows, optimiser=optax.sgd(learning_rate=lr)
    ),
    "adam": lambda lr: functools.partial(
        optax_flows, optimiser=optax.adam(learning_rate=lr)
    ),
    "closed_form_l1": lambda lr: closed_form_flows_l1,  # LR is ignored here
    "closed_form_l2": lambda lr: closed_form_flows_l2,  # LR is ignored here
}
SOLVER_CONSENSUS_FACTORY = {
    "sgd": lambda lr: functools.partial(
        optax_consensus, optimiser=optax.sgd(learning_rate=lr)
    ),
    "adam": lambda lr: functools.partial(
        optax_consensus, optimiser=optax.adam(learning_rate=lr)
    ),
}
