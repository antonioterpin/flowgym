"""Consensus algorithm for robust estimation of flow parameters."""

import functools

import jax
import jax.numpy as jnp
from goggles import get_logger

from flowgym.flow.consensus.admm import run_admm
from flowgym.flow.consensus.objectives import (
    flows_objective,
    weights_and_anchors,
    z_objective,
)
from flowgym.types import ExperimentParams

logger = get_logger(__name__)


def mean_consensus(
    flows: jnp.ndarray,
    weights: jnp.ndarray,
    _: dict | None = None,
    epsilon: float = 1e-8,
) -> tuple[jnp.ndarray, dict]:
    """Compute the mean consensus flow from multiple flow estimates.

    Args:
        flows: Array of flow estimates from different agents.
            shape (N, H, W, 2) where B is the batch size,
            N is the number of estimates,
        weights: Weights for each flow estimate, shape (N, H, W).
            The weights are used to compute a weighted mean of
            the flow estimates.
        epsilon: Small value to avoid division by zero when normalizing weights.

    Returns:
        - Mean consensus flow estimate.
        - Placeholder for additional return values, if needed.
    """
    # Detect pixels where all N weights are zero
    all_zero = jnp.all(weights == 0, axis=0, keepdims=True)  # (1,H,W), bool

    weights = jnp.where(all_zero, 1.0, weights)

    weights = weights / (jnp.sum(weights, axis=0, keepdims=True) + epsilon)

    return jnp.sum(flows * weights[..., None], axis=0), {}


def median_consensus(
    flows: jax.Array,
    _: jax.Array | None = None,
    __: dict | None = None,
) -> tuple[jax.Array, dict]:
    """Compute the median consensus flow from multiple flow estimates.

    Args:
        flows: Array of flow estimates from different agents.
            shape (N, H, W, 2) where B is the batch size,
            N is the number of estimates.

    Returns:
        jnp.ndarray: Median consensus flow estimate.
        None: Placeholder for additional return values, if needed.
    """
    return jnp.median(flows, axis=0), {}


def admm_consensus(
    flows: jax.Array,
    weights: jax.Array,
    config: dict,
) -> tuple[jax.Array, dict]:
    """ADMM-based consensus algorithm for robust flow estimation.

    This function sets the framework for the ADMM algorithm by setting
    the initial parameters and providing a constant API for
    the consensus function. The actual ADMM is defined in run_admm,
    which should be implemented separately.

    Args:
        flows: Array of flow estimates from different agents.
            shape (N, H, W, 2) where N is the number of estimates.
        weights: Weights for each flow estimate, shape (N, H, W).
            The weights are used to compute the flow objective function.
        config: Configuration parameters for ADMM.

    Returns:
        - Consensus flow estimate after ADMM iterations.
        - Metrics including final stopping time.

    Raises:
        ValueError: If any of the configuration parameters are invalid.
    """
    # Copy the configuration to avoid modifying the original
    cfg = config.copy()

    # Extract the transformation if any
    transformation = cfg.pop("transformation", None)
    if transformation is not None and transformation != "none":
        if transformation == "symlog":
            # Apply symlog transformation to flows
            flows = jnp.sign(flows) * jnp.log1p(jnp.abs(flows))
        else:
            raise ValueError(
                f"Invalid transformation: {transformation}. "
                "Expected None, or 'symlog'."
            )

    # Extract rho, the augmented Lagrangian parameter
    rho = cfg.pop("rho", 1.0)
    if not isinstance(rho, (float, int)):
        raise TypeError(
            f"Invalid rho type: {type(rho)}. Expected float or int."
        )
    if rho <= 0:
        raise ValueError(f"Rho must be positive, got {rho}.")

    # Define the objective functions for flows
    flows_objective_type = cfg.pop("flows_objective_type", "l2")
    if flows_objective_type not in ["l2", "l1", "huber"]:
        raise ValueError(
            f"Invalid flows_objective_type: {flows_objective_type}. "
            "Expected 'l2', 'l1', or 'huber'."
        )

    # Ensure that weights have the correct shape
    if weights.shape != flows.shape[:-1]:
        raise ValueError(
            f"Weights must have the same shape as flows except for "
            f"the last dimension, got {weights.shape} and {flows.shape[:-1]}."
        )

    # Extract and validate the solver for flows
    solver_flows: str = cfg.pop("solver_flows", "closed_form_l2")

    if solver_flows not in [
        "sgd",
        "adam",
        "closed_form_l2",
        "closed_form_l1",
        "closed_form_huber",
    ]:
        raise ValueError(
            f"Invalid solver_flows: {solver_flows}. "
            "Expected 'sgd', 'adam', 'closed_form_l2', 'closed_form_l1', "
            "or 'closed_form_huber'."
        )
    
    if solver_flows not in [
        "closed_form_l2",
        "closed_form_l1",
        "closed_form_huber",
    ]:
        # Create the objective function for flows
        objective_fn_flows = functools.partial(
            flows_objective,
            initial_flows=flows,
            weights=weights,
            objective_type=flows_objective_type,
            rho=rho,
        )
    else:
        if flows_objective_type != "l2" and solver_flows == "closed_form_l2":
            raise ValueError(
                "flows_objective_type must be 'l2' when using "
                "the closed_form_l2 solver."
            )
        elif flows_objective_type != "l1" and solver_flows == "closed_form_l1":
            raise ValueError(
                "flows_objective_type must be 'l1' when using "
                "the closed_form_l1 solver."
            )
        elif (
            flows_objective_type != "huber"
            and solver_flows == "closed_form_huber"
        ):
            raise ValueError(
                "flows_objective_type must be 'huber' when using "
                "the closed_form_huber solver."
            )
        # For closed form, we don't need an objective function
        # We will use the nd_anchors function directly
        objective_fn_flows = functools.partial(
            weights_and_anchors,
            weights=weights,
            anchor_flows=flows,
        )

    # Extract and validate consensus configuration parameters
    regularizer_list = cfg.pop("regularizer_list", [])
    if not isinstance(regularizer_list, list):
        raise ValueError(
            f"Invalid regularizer_list type: {type(regularizer_list)}. "
            "Expected list."
        )
    regularizer_weights = cfg.pop("regularizer_weights", {})
    if not isinstance(regularizer_weights, dict):
        raise ValueError(
            f"Invalid regularizer_weights type: {type(regularizer_weights)}. "
            "Expected dict."
        )
    if not all(
        isinstance(v, (float, int)) for v in regularizer_weights.values()
    ):
        raise ValueError(
            "All values in regularizer_weights must be float or int."
        )
    if not all(k in regularizer_weights for k in regularizer_list):
        missing_weights = [
            reg for reg in regularizer_list if reg not in regularizer_weights
        ]
        raise ValueError(
            "All regularizers in regularizer_list must have "
            "corresponding weights in regularizer_weights."
            f" Missing weights for: {missing_weights}."
        )
    if regularizer_list is None:
        regularizer_list = []
    if regularizer_weights is None:
        regularizer_weights = {}

    # Extract and validate ADMM solver parameters
    num_iterations_flows: int = cfg.pop("num_iterations_flows", 1)
    if (not isinstance(num_iterations_flows, int) or num_iterations_flows <= 0):
        raise ValueError(
            "num_iterations_flows must be a positive integer, "
            f"got {num_iterations_flows}."
        )

    # Extract and validate the learning rate for flows
    learning_rate_flows: float = cfg.pop("learning_rate_flows", 1e-2)
    if not isinstance(learning_rate_flows, float) or learning_rate_flows <= 0:
        raise ValueError(
            "learning_rate_flows must be a positive float, "
            f"got {learning_rate_flows}."
        )

    # Extract and validate the number of iterations for consensus
    num_iterations_consensus: int = cfg.pop("num_iterations_consensus", 1)
    if (
        not isinstance(num_iterations_consensus, int)
        or num_iterations_consensus <= 0
    ):
        raise ValueError(
            "num_iterations_consensus must be a positive integer, "
            f"got {num_iterations_consensus}."
        )

    # Extract and validate the consensus solver
    solver_consensus: str = cfg.pop("solver_consensus", "sgd")
    if not isinstance(solver_consensus, str):
        raise ValueError(
            f"Invalid solver_consensus type: {type(solver_consensus)}."
            " Expected str."
        )
    if solver_consensus not in ["sgd", "adam"]:
        raise ValueError(
            f"Invalid solver_consensus: {solver_consensus}. "
            "Expected 'sgd' or 'adam'."
        )

    # Create the objective function for the consensus variable z
    objective_fn_z = functools.partial(
        z_objective,
        regularizer_list=regularizer_list,
        regularizer_weights=regularizer_weights,
        rho=rho,
    )

    learning_rate_consensus = cfg.pop("learning_rate_consensus", 0.01)
    if (
        not isinstance(learning_rate_consensus, float) 
        or learning_rate_consensus <= 0
    ):
        raise ValueError(
            "learning_rate_consensus must be a positive float, "
            f"got {learning_rate_consensus}."
        )

    max_admm_iterations = cfg.pop("max_admm_iterations", 10)
    if (
        not isinstance(max_admm_iterations, int)
        or max_admm_iterations <= 0
    ):
        raise ValueError(
            "max_admm_iterations must be a positive integer, "
            f"got {max_admm_iterations}."
        )

    # Extract and validate the absolute stopping criterion
    eps_abs_stopping: float = cfg.pop("eps_abs_stopping", 0.0)
    if (not isinstance(eps_abs_stopping, float) or eps_abs_stopping < 0):
        raise ValueError(
            "eps_abs_stopping must be a non-negative float or None, "
            f"got {eps_abs_stopping}."
        )

    # Extract and validate the relative stopping criterion
    eps_rel_stopping: float = cfg.pop("eps_rel_stopping", 0.0)
    if (not isinstance(eps_rel_stopping, float) or eps_rel_stopping < 0):
        raise ValueError(
            "eps_rel_stopping must be a non-negative float or None, "
            f"got {eps_rel_stopping}."
        )

    log_metrics = cfg.pop("exp_log_metrics", {})

    runner = run_admm
    if transformation is not None and transformation != "none":
        if transformation == "symlog":
            # add a decorator to run_admm to inverse transform the
            # consensus flow at the end
            original_run_admm = run_admm

            @functools.wraps(original_run_admm)
            def run_admm_symlog(*args, **kwargs):
                result, metrics = original_run_admm(*args, **kwargs)
                if transformation == "symlog":
                    # Inverse transform the consensus flow
                    result = jnp.sign(result) * jnp.expm1(jnp.abs(result))
                return result, metrics

            runner = run_admm_symlog
        else:
            raise ValueError(
                f"Invalid transformation: {transformation}. "
                "Expected None, or 'symlog'."
            )

    # Remove keys that are not used in the consensus function
    cfg.pop("weights", None)
    cfg.pop("weights_type", None)
    cfg.pop("normalization", None)
    cfg.pop("patch_size", None)
    cfg.pop("patch_stride", None)
    cfg.pop("exp_log_path", None)
    cfg.pop("exp_baseline_performance", None)
    cfg.pop("exp_oracle_select_weights", None)

    if cfg:
        raise ValueError(f"Unknown configuration parameters: {cfg.keys()}.")

    return runner(
        flows,
        rho=rho,
        objective_fn_flows=objective_fn_flows,
        num_iterations_flows=num_iterations_flows,
        solver_flows=solver_flows,
        objective_fn_z=objective_fn_z,
        num_iterations_consensus=num_iterations_consensus,
        solver_consensus=solver_consensus,
        max_admm_iterations=max_admm_iterations,
        eps_abs=eps_abs_stopping,
        eps_rel=eps_rel_stopping,
        learning_rate_flows=learning_rate_flows,
        learning_rate_consensus=learning_rate_consensus,
        log_metrics=log_metrics,
    )


CONSENSUS_REGISTRY = {
    "mean": mean_consensus,
    "median": median_consensus,
    "admm": admm_consensus,
}


def validate_experimental_params(cfg: dict) -> ExperimentParams:
    """Validate experimental parameters for consensus algorithms.

    Args:
        cfg: Configuration dictionary containing experimental parameters.

    Raises:
        ValueError: If any of the parameters are invalid.

    Returns:
        ExperimentParams: Validated experimental parameters.
    """
    # Remove keys that are not used in the consensus function
    if cfg.get("epe_limit", None) is not None:
        epe_limit = cfg["epe_limit"]
        if not isinstance(epe_limit, (float, int)) or epe_limit <= 0:
            raise ValueError(
                f"epe_limit must be a positive number, got {epe_limit}."
            )

    # TODO: add validation for new experimental parameters as needed
    return ExperimentParams(**cfg)
