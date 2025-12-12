"""ADMM implementation for distributed flow estimation."""

import jax
import jax.numpy as jnp
from flowgym.flow.consensus.solvers import (
    SOLVER_FLOWS_FACTORY,
    SOLVER_CONSENSUS_FACTORY,
)
from flowgym.utils import DEBUG
from collections.abc import Callable


def run_admm(
    flows: jnp.ndarray,
    rho: float,
    objective_fn_flows: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    num_iterations_flows: int | None,
    solver_flows: str,
    objective_fn_z: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray
    ],
    num_iterations_consensus: int,  # TODO: should be optional when we have closed-form
    solver_consensus: str,
    max_admm_iterations: int,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    learning_rate_flows: float = 1e-3,
    learning_rate_consensus: float = 1e-3,
) -> tuple[jnp.ndarray, dict]:
    """Run ADMM algorithm for consensus-based flow estimation.

    This function implements the ADMM algorithm to find a consensus flow estimate
    from multiple flow estimates provided by different agents.

    Args:
        flows (jnp.ndarray): Array of flow estimates from different agents.
            shape (N, H, W, 2) where N is the number of agents.
        rho (float): Parameter associated with the augmented Lagrangian.
        objective_fn_flows (callable): Objective function for flow estimates.
        num_iterations_flows (int): Number of iterations for flow optimization.
        solver_flows (str): Solver to use for flow optimization.
        objective_fn_z (callable): Objective function for consensus variable.
        num_iterations_consensus (int): Number of iterations for consensus optimization.
        solver_consensus (str): Solver to use for consensus optimization.
        max_admm_iterations (int): Maximum number of ADMM iterations.
        eps_abs (float): eps absolute for stopping criterion.
        eps_rel (float): eps relative for stopping criterion.
        learning_rate_flows (float): Learning rate for flow solver.
        learning_rate_consensus (float): Learning rate for consensus solver.

    Returns:
        jnp.ndarray: Consensus flow estimate after ADMM iterations.
        dict: Metrics including final stopping time.
    """
    # TODO: this could be a different initialization
    # Initialize consensus flow and dual variable
    consensus_flow = jnp.mean(flows, axis=0)  # (H, W, 2)
    consensus_duals = flows - consensus_flow[None, ...]  # (N, H, W, 2)

    try:
        optimize_flows = SOLVER_FLOWS_FACTORY[solver_flows](learning_rate_flows)
        optimize_consensus = SOLVER_CONSENSUS_FACTORY[solver_consensus](
            learning_rate_consensus
        )
    except KeyError as e:
        raise ValueError(f"Unknown solver name: {e}")

    # Dimensions of the problem
    # Assuming flows has shape (N, H, W, 2) where
    # N is the number of agents, H and W are the height and
    # width of the flow field, and 2 is for the flow vector

    # Number of agents
    N = flows.shape[0]

    # Number of decision variables per agent (H * W * 2)
    vars_per_agent = flows.shape[-3] * flows.shape[-2] * flows.shape[-1]  # H * W * 2

    # Number of total decision variables
    total_decision_vars = N * vars_per_agent

    # Initialize active and stopping time
    active = True
    stopping_time = 0

    def admm_iteration(i, carry):
        """Perform a single ADMM iteration."""
        flows, consensus_flow, consensus_dual, active, stopping_time = carry

        # Only update active elements; others just pass their previous value

        # Update flows
        if solver_flows == "closed_form":
            # for closed-form solvers, use jnp.where
            flows_new = optimize_flows(
                flows,
                consensus_flow,
                consensus_dual,
                objective_fn_flows,
                rho,
                num_iterations_flows,
            )
            flows_new = jnp.where(active, flows_new, flows)

        else:
            # for iterative solvers, use jax.lax.cond
            flows_new = jax.lax.cond(
                active,
                lambda _: optimize_flows(
                    flows,
                    consensus_flow,
                    consensus_dual,
                    objective_fn_flows,
                    rho,
                    num_iterations_flows,
                ),
                lambda _: flows,
                operand=None,
            )

        # Update consensus
        consensus_flow_new = jax.lax.cond(
            active,
            lambda _: optimize_consensus(
                flows_new,
                consensus_flow,
                consensus_dual,
                objective_fn_z,
                rho,
                num_iterations_consensus,
            ),
            lambda _: consensus_flow,
            operand=None,
        )

        # Update dual
        dual_update = flows_new - consensus_flow_new[None, ...]
        # TODO: we should do this with a lax.cond and a lax.map
        consensus_dual_new = jnp.where(
            active,
            consensus_dual + dual_update,
            consensus_dual,
        )

        if eps_rel is not None and eps_abs is not None:
            # Apply stopping criteria based on eps_abs and eps_rel

            # Stopping conditions (per batch)
            primal_residual = jnp.linalg.norm(flows_new - consensus_flow_new[None, ...])
            dual_residual = (
                rho * jnp.sqrt(N) * jnp.linalg.norm(consensus_flow_new - consensus_flow)
            )
            flows_norm = jnp.linalg.norm(flows_new)
            consensus_norm = jnp.sqrt(N) * jnp.linalg.norm(consensus_flow_new)
            eps_pri = jnp.sqrt(total_decision_vars) * eps_abs + eps_rel * jnp.maximum(
                flows_norm, consensus_norm
            )
            eps_dual = jnp.sqrt(
                total_decision_vars
            ) * eps_abs + eps_rel * rho * jnp.linalg.norm(consensus_dual_new)

            # Update active mask (per batch item)
            still_active = (primal_residual > eps_pri) | (dual_residual > eps_dual)
            active_new = active & still_active

            # Update stopping time
            stopping_time_new = jnp.where(
                (active) & (~active_new),  # just became inactive this iter
                i,  # current loop index
                stopping_time,  # otherwise keep previous value
            )

            if DEBUG:
                jax.debug.print(
                    "Dual residual norm: {}",
                    jnp.linalg.norm(dual_residual),
                )

                jax.debug.print(
                    "Primal residual norm: {}",
                    jnp.linalg.norm(primal_residual),
                )

                jax.debug.print(
                    "Active agents changed at iteration {}: {} -> {}",
                    i,
                    active,
                    active_new,
                )

                jax.debug.print(
                    "Absolute primal residual: {}", jnp.sqrt(total_decision_vars) * 1e-4
                )

                jax.debug.print(
                    "Absolute dual residual: {}", jnp.sqrt(total_decision_vars) * 1e-4
                )

                jax.debug.print(
                    "Relative primal residual: {}",
                    1e-4 * jnp.maximum(flows_norm, consensus_norm),
                )

                jax.debug.print(
                    "Relative dual residual: {}",
                    1e-4 * jnp.linalg.norm(rho * consensus_dual_new),
                )

                jax.debug.print(
                    "Primal variables norm: {}",
                    jnp.linalg.norm(flows_new),
                )

                jax.debug.print(
                    "Consensus variables norm: {}",
                    jnp.linalg.norm(consensus_flow_new),
                )

                jax.debug.print(
                    "Dual variables norm: {}",
                    jnp.linalg.norm(consensus_dual_new),
                )

            return (
                flows_new,
                consensus_flow_new,
                consensus_dual_new,
                active_new,
                stopping_time_new,
            )
        else:
            return (
                flows_new,
                consensus_flow_new,
                consensus_dual_new,
                active,
                stopping_time,
            )

    # Initial carry
    init_carry = (
        flows,
        consensus_flow,
        consensus_duals,
        active,
        stopping_time,
    )

    final_carry = jax.lax.fori_loop(0, max_admm_iterations, admm_iteration, init_carry)

    final_flows, final_consensus_flow, _, final_actives, final_stopping_time = (
        final_carry
    )
    if DEBUG:
        jax.debug.print(
            "Final stopping time: {}, type: {}",
            final_stopping_time,
            final_stopping_time.dtype,
        )
        jax.debug.print(
            "Final active agents: {}",
            final_actives,
        )
        primal_residual = jnp.linalg.norm(final_flows - final_consensus_flow[None, ...])
        jax.debug.print(
            "Final primal residual norm: {}",
            jnp.linalg.norm(primal_residual),
        )

    metrics = {"final_stopping_time": final_stopping_time}
    return final_consensus_flow, metrics
