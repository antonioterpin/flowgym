"""Solver module for flow estimation."""

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import optax

from flowgym.flow.consensus.types import (
    ConsensusSolver,
    ConsensusSolverFactory,
    FlowSolver,
    FlowSolverFactory,
)


def closed_form_flows_l2(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    weights_and_anchors_fn: Callable[[], tuple[jnp.ndarray, jnp.ndarray]],
    rho: float,
    unused: object,
) -> jnp.ndarray:
    """Closed form x-update for flows with a l2 penalty.

    Args:
        flows: Current flow estimates from different agents.
        consensus_flow: Current consensus flow estimate.
        consensus_dual: Dual variable for consensus.
        weights_and_anchors_fn: Function that returns weights and anchor flows.
        rho: Penalty parameter for the consensus term.
        unused: Placeholder for compatibility.

    Returns:
        Updated flow estimates as a jax numpy array.
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
    weights_and_anchors_fn: Callable[[], tuple[jnp.ndarray, jnp.ndarray]],
    rho: float,
    unused: object,
) -> jnp.ndarray:
    """Closed form x-update for flows with an L1 penalty.

    Args:
        flows: Current flow estimates from different agents.
        consensus_flow: Current consensus flow estimate.
        consensus_dual: Dual variable for consensus.
        weights_and_anchors_fn: Function that returns weights and anchor flows.
        rho: Penalty parameter for the consensus term.
        unused: Placeholder for compatibility.

    Returns:
        Updated flow estimates as a jax numpy array.
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


def closed_form_flows_huber(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    weights_and_anchors_fn: Callable[[], tuple[jnp.ndarray, jnp.ndarray]],
    rho: float,
    unused: object,
) -> jnp.ndarray:
    """Closed-form x-update for flows with a Huber penalty.

    Args:
        flows: Current flow estimates from different agents.
        consensus_flow: Current consensus flow estimate.
        consensus_dual: Dual variable for consensus.
        weights_and_anchors_fn: Callable returning (weights, anchor_flows).
        rho: ADMM penalty parameter.
        unused: Placeholder for compatibility.

    Returns:
        Updated flow estimates as a jax numpy array.

    Raises:
        ValueError: If the computed update is not a JAX array.
    """
    # W_i and z_i from the derivation
    weights, anchor_flows = weights_and_anchors_fn()  # (N,H,W), (N,H,W,2)
    delta = 1.0

    # b_i = z^k - u_i
    b = consensus_flow[jnp.newaxis, ...] - consensus_dual  # (N,H,W,2)

    # p_l, tau_l and sqrt(p_l) (broadcast over the flow components)
    p = weights  # (N,H,W)
    tau = (p / rho)[..., jnp.newaxis]  # (N,H,W,1)
    sqrt_p = jnp.sqrt(p)[..., jnp.newaxis]  # (N,H,W,1)

    # v = sqrt(p_l) * (b_l - a_l)
    v = sqrt_p * (b - anchor_flows)  # (N,H,W,2)

    # Prox of Huber: prox_{tau huber_delta}(v)
    abs_v = jnp.abs(v)
    threshold = (1.0 + tau) * delta

    prox_v = jnp.where(
        abs_v <= threshold,
        v / (1.0 + tau),
        jnp.where(v > threshold, v - tau * delta, v + tau * delta),
    )

    # (z_i[k+1])_l = a_l + (1/sqrt(p_l)) * prox_{tau_l phi_delta}(v_l)
    updated = anchor_flows + prox_v / sqrt_p

    # (Optional) handle zero weights explicitly to avoid NaNs:
    # when p = 0, the data term vanishes and the minimizer is x = b.
    zero_weight = (p == 0)[..., jnp.newaxis]
    updated = jnp.where(zero_weight, b, updated)

    if not isinstance(updated, jnp.ndarray):
        raise ValueError("Updated flows is not a jnp.ndarray")

    return updated


def optax_solve(
    params: jax.Array,
    objective_fn: Callable[[jax.Array], jax.Array],
    optimiser: optax.GradientTransformation,
    num_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve an optimization problem using Optax.

    Args:
        params: Initial parameters for optimization.
        objective_fn: Objective function to minimize.
        optimiser: Optax optimizer.
        num_steps: Number of optimization steps.

    Returns:
        optax.Params: Optimized parameters after the specified number of steps.
        jnp.ndarray: Loss values at each optimization step.
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

    return cast(jax.Array, optimized_params), losses


def optax_consensus(
    flows: jax.Array,
    consensus_flow: jax.Array,
    consensus_dual: jax.Array,
    objective_fn: Callable,
    rho: float,
    num_steps: int,
    optimiser: optax.GradientTransformation,
) -> jax.Array:
    """Optax solver for the consensus variable.

    Args:
        flows: Current flow estimates from different agents.
        consensus_flow: Current consensus flow estimate.
        consensus_dual: Current dual variable for consensus.
        objective_fn: Objective function for consensus variable.
        rho: Parameter associated with the augmented Lagrangian.
        num_steps: Number of optimization steps.
        optimiser: Optax optimizer.

    Returns:
        Updated consensus flow estimate (or optax.Params) as a jax array.
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
    flows: jax.Array,
    consensus_flow: jax.Array,
    consensus_dual: jax.Array,
    objective_fn: Callable[..., jax.Array],
    rho: float,
    num_steps: int,
    optimiser: optax.GradientTransformation,
) -> jax.Array:
    """Optax solver for flows.

    Args:
        flows: Current flow estimates from different agents.
        consensus_flow: Current consensus flow estimate.
        consensus_dual: Current dual variable for consensus.
        objective_fn: Objective function for flow estimates.
        rho: Parameter associated with the augmented Lagrangian.
        num_steps: Number of optimization steps.
        optimiser: Optax optimizer.

    Returns:
        optax.Params representing the updated flow estimates.
    """
    # Freeze consensus_flow and consensus_dual in the objective function
    flows_fn = functools.partial(
        objective_fn,
        consensus_flow=consensus_flow,
        consensus_dual=consensus_dual,
        rho=rho,
    )
    flows, _ = optax_solve(flows, flows_fn, optimiser, num_steps)
    return flows


SOLVER_FLOWS_FACTORY: dict[str, FlowSolverFactory] = {
    "sgd": lambda lr: cast(
        FlowSolver,
        functools.partial(optax_flows, optimiser=optax.sgd(learning_rate=lr)),
    ),
    "adam": lambda lr: cast(
        FlowSolver,
        functools.partial(optax_flows, optimiser=optax.adam(learning_rate=lr)),
    ),
    # LR is ignored here
    "closed_form_l1": cast(FlowSolverFactory, lambda lr: closed_form_flows_l1),
    "closed_form_l2": cast(FlowSolverFactory, lambda lr: closed_form_flows_l2),
    "closed_form_huber": cast(
        FlowSolverFactory, lambda lr: closed_form_flows_huber
    ),
}

SOLVER_CONSENSUS_FACTORY: dict[str, ConsensusSolverFactory] = {
    "sgd": lambda lr: cast(
        ConsensusSolver,
        functools.partial(
            optax_consensus, optimiser=optax.sgd(learning_rate=lr)
        ),
    ),
    "adam": lambda lr: cast(
        ConsensusSolver,
        functools.partial(
            optax_consensus, optimiser=optax.adam(learning_rate=lr)
        ),
    ),
}
