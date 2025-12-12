"""Consensus algorithm for robust estimation of flow parameters."""

import functools

# import jax
import jax.numpy as jnp
from flowgym.flow.consensus.admm import run_admm
from flowgym.flow.consensus.objectives import (
    flows_objective,
    z_objective,
    weights_and_anchors,
)

from goggles import get_logger

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
        flows (jnp.ndarray): Array of flow estimates from different agents.
            shape (N, H, W, 2) where B is the batch size, N is the number of estimates,
        weights (jnp.ndarray): Weights for each flow estimate, shape (N, H, W).
            The weights are used to compute a weighted mean of the flow estimates.
        _ (dict, optional): Unused argument, placeholder for configuration parameters.
        epsilon (float): Small value to avoid division by zero when normalizing weights.

    Returns:
        jnp.ndarray: Mean consensus flow estimate.
        None: Placeholder for additional return values, if needed.
    """
    # Detect pixels where all N weights are zero
    all_zero = jnp.all(weights == 0, axis=0, keepdims=True)  # (1,H,W), bool

    weights = jnp.where(all_zero, 1.0, weights)

    weights = weights / (jnp.sum(weights, axis=0, keepdims=True) + epsilon)

    return jnp.sum(flows * weights[..., None], axis=0), {}


def median_consensus(
    flows: jnp.ndarray,
    _: jnp.ndarray | None = None,
    __: dict | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Compute the median consensus flow from multiple flow estimates.

    Args:
        flows (jnp.ndarray): Array of flow estimates from different agents.
            shape (N, H, W, 2) where B is the batch size, N is the number of estimates.
        _ (jnp.ndarray): Unused argument, placeholder for weights.
        __ (dict, optional): Unused argument, placeholder for configuration parameters.

    Returns:
        jnp.ndarray: Median consensus flow estimate.
        None: Placeholder for additional return values, if needed.
    """
    return jnp.median(flows, axis=0), {}


def admm_consensus(
    flows: jnp.ndarray,
    weights: jnp.ndarray,
    config: dict,
) -> tuple[jnp.ndarray, dict]:
    """ADMM-based consensus algorithm for robust flow estimation.

    This function sets the framework for the ADMM algorithm by setting the initial
    parameters and providing a constant API for the consensus function. The actual ADMM
    is defined in run_admm, which should be implemented separately.

    Args:
        flows (jnp.ndarray): Array of flow estimates from different agents.
            shape (N, H, W, 2) where N is the number of estimates.
        weights (jnp.ndarray): Weights for each flow estimate, shape (N, H, W).
            The weights are used to compute the flow objective function.
        config (dict): Configuration parameters for ADMM.

    Returns:
        final_consensus_flow (jnp.ndarray): Consensus flow estimate after ADMM iterations.
        dict: Metrics including final stopping time.
    """
    # Copy the configuration to avoid modifying the original
    cfg = config.copy()

    # Extract rho, the augmented Lagrangian parameter
    rho = cfg.pop("rho", 1.0)
    if not isinstance(rho, (float, int)):
        raise ValueError(f"Invalid rho type: {type(rho)}. Expected float or int.")
    if rho <= 0:
        raise ValueError(f"Rho must be positive, got {rho}.")

    # Define the objective functions for flows
    flows_objective_type = cfg.pop("flows_objective_type", "l2")
    if flows_objective_type not in ["l2", "l1"]:
        raise ValueError(
            f"Invalid flows_objective_type: {flows_objective_type}. "
            "Expected 'l2' or 'l1'."
        )

    # Input validation for weights
    if not isinstance(weights, jnp.ndarray):
        raise ValueError(
            f"Invalid weights type: {type(weights)}. Expected jnp.ndarray."
        )

    # Ensure that weights have the correct shape
    if weights.shape != flows.shape[:-1]:
        raise ValueError(
            f"Weights must have the same shape as flows except for the last dimension, "
            f"got {weights.shape} and {flows.shape[:-1]}."
        )

    # Extract and validate the solver for flows
    if "solver_flows" in cfg:
        solver_flows = cfg.pop("solver_flows")
        if not isinstance(solver_flows, str):
            raise ValueError(
                f"Invalid solver_flows type: {type(solver_flows)}. Expected str."
            )
    else:
        logger.warning(
            "No solver_flows specified in the configuration. "
            "Using 'closed_form_l2' as default."
        )
        solver_flows = "closed_form_l2"

    if solver_flows != "closed_form_l2" and solver_flows != "closed_form_l1":
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
                "flows_objective_type must be 'l2' when using the closed_form_l2 solver."
            )
        elif flows_objective_type != "l1" and solver_flows == "closed_form_l1":
            raise ValueError(
                "flows_objective_type must be 'l1' when using the closed_form_l1 solver."
            )
        # For closed form, we don't need an objective function
        # We will use the weights_and_anchors function directly
        objective_fn_flows = functools.partial(
            weights_and_anchors,
            anchor_flows=flows,
            weights=weights,
        )

    # Extract and validate consensus configuration parameters
    regularizer_list = cfg.pop("regularizer_list", [])
    if regularizer_list is not None and not isinstance(regularizer_list, list):
        raise ValueError(
            f"Invalid regularizer_list type: {type(regularizer_list)}. Expected list."
        )
    regularizer_weights = cfg.pop("regularizer_weights", {})
    if regularizer_weights is not None and not isinstance(regularizer_weights, dict):
        raise ValueError(
            f"Invalid regularizer_weights type: {type(regularizer_weights)}. "
            "Expected dict."
        )
    if regularizer_weights is not None and not all(
        isinstance(v, (float, int)) for v in regularizer_weights.values()
    ):
        raise ValueError("All values in regularizer_weights must be float or int.")
    if regularizer_list is not None and not all(
        k in regularizer_weights for k in regularizer_list
    ):
        missing_weights = [
            reg for reg in regularizer_list if reg not in regularizer_weights
        ]
        raise ValueError(
            "All regularizers in regularizer_list must have corresponding weights "
            "in regularizer_weights."
            f" Missing weights for: {missing_weights}."
        )
    if regularizer_list is None:
        regularizer_list = []
    if regularizer_weights is None:
        regularizer_weights = {}

    # Create the objective function for the consensus variable z
    objective_fn_z = functools.partial(
        z_objective,
        regularizer_list=regularizer_list,
        regularizer_weights=regularizer_weights,
        rho=rho,
    )

    # Extract and validate ADMM solver parameters
    if (
        solver_flows != "closed_form_l2"
        and solver_flows != "closed_form_l1"
        and "num_iterations_flows" not in cfg
    ):
        logger.warning(
            "No num_iterations_flows specified in the configuration. "
            "Using 1 as default."
        )
        num_iterations_flows = 1
    elif solver_flows == "closed_form_l2" or solver_flows == "closed_form_l1":
        if "num_iterations_flows" in cfg:
            if cfg["num_iterations_flows"] is not None:
                raise ValueError(
                    "num_iterations_flows should not be specified "
                    "when using a closed_form solver."
                )
            else:
                num_iterations_flows = cfg.pop("num_iterations_flows")
        else:
            num_iterations_flows = None
    else:
        num_iterations_flows = cfg.pop("num_iterations_flows")
        if not isinstance(num_iterations_flows, int) or num_iterations_flows <= 0:
            raise ValueError(
                "num_iterations_flows must be a positive integer, "
                f"got {num_iterations_flows}."
            )

    # Extract and validate the learning rate for flows
    if (
        solver_flows != "closed_form_l1"
        and solver_flows != "closed_form_l2"
        and "learning_rate_flows" not in cfg
    ):
        logger.warning(
            "No learning_rate_flows specified in the configuration. "
            "Using 0.01 as default."
        )
        learning_rate_flows = 0.01
    elif solver_flows == "closed_form_l1" or solver_flows == "closed_form_l2":
        if "learning_rate_flows" in cfg:
            if cfg["learning_rate_flows"] is not None:
                raise ValueError(
                    "learning_rate_flows should not be specified "
                    "when using a closed_form solver."
                )
            else:
                learning_rate_flows = cfg.pop("learning_rate_flows")
        else:
            learning_rate_flows = None
    else:
        learning_rate_flows = cfg.pop("learning_rate_flows")
        if not isinstance(learning_rate_flows, float) or learning_rate_flows <= 0:
            raise ValueError(
                "learning_rate_flows must be a positive float, "
                f"got {learning_rate_flows}."
            )

    # Extract and validate the number of iterations for consensus
    if "num_iterations_consensus" not in cfg or cfg["num_iterations_consensus"] is None:
        logger.warning(
            "No num_iterations_consensus specified in the configuration. "
            "Using 1 as default."
        )
        num_iterations_consensus = 1
    else:
        num_iterations_consensus = cfg.pop("num_iterations_consensus")
        if (
            not isinstance(num_iterations_consensus, int)
            or num_iterations_consensus <= 0
        ):
            raise ValueError(
                "num_iterations_consensus must be a positive integer, "
                f"got {num_iterations_consensus}."
            )

    # Extract and validate the consensus solver
    if "solver_consensus" not in cfg or cfg["solver_consensus"] is None:
        logger.warning(
            "No solver_consensus specified in the configuration. Using 'sgd' as default."
        )
        solver_consensus = "sgd"
    else:
        solver_consensus = cfg.pop("solver_consensus")
        if not isinstance(solver_consensus, str):
            raise ValueError(
                f"Invalid solver_consensus type: {type(solver_consensus)}. Expected str."
            )

    if "learning_rate_consensus" not in cfg or cfg["learning_rate_consensus"] is None:
        logger.warning(
            "No learning_rate_consensus specified in the configuration. "
            "Using 0.01 as default."
        )
        learning_rate_consensus = 0.01
    else:
        learning_rate_consensus = cfg.pop("learning_rate_consensus")
        if (
            not isinstance(learning_rate_consensus, float)
            or learning_rate_consensus <= 0
        ):
            raise ValueError(
                "learning_rate_consensus must be a positive float, "
                f"got {learning_rate_consensus}."
            )

    # Extract and validate the maximum number of ADMM iterations
    if "max_admm_iterations" not in cfg or cfg["max_admm_iterations"] is None:
        logger.warning(
            "No max_admm_iterations specified in the configuration. Using 10 as default."
        )
        max_admm_iterations = 10
    else:
        max_admm_iterations = cfg.pop("max_admm_iterations")
        if not isinstance(max_admm_iterations, int) or max_admm_iterations <= 0:
            raise ValueError(
                "max_admm_iterations must be a positive integer, "
                f"got {max_admm_iterations}."
            )

    # Extract and validate the absolute stopping criterion
    if "eps_abs_stopping" not in cfg or cfg["eps_abs_stopping"] is None:
        logger.warning(
            "No eps_abs_stopping specified in the configuration. Using None as default."
        )
        eps_abs_stopping = None
    else:
        eps_abs_stopping = cfg.pop("eps_abs_stopping")
        if not isinstance(eps_abs_stopping, float):
            raise ValueError(
                f"Invalid eps_abs_stopping type: {type(eps_abs_stopping)}. "
                "Expected float."
            )

    # Extract and validate the relative stopping criterion
    if "eps_rel_stopping" not in cfg or cfg["eps_rel_stopping"] is None:
        logger.warning(
            "No eps_rel_stopping specified in the configuration. Using None as default."
        )
        eps_rel_stopping = None
    else:
        eps_rel_stopping = cfg.pop("eps_rel_stopping")
        if not isinstance(eps_rel_stopping, float):
            raise ValueError(
                f"Invalid eps_rel_stopping type: {type(eps_rel_stopping)}. "
                "Expected float."
            )

    if eps_abs_stopping is not None and eps_abs_stopping <= 0:
        raise ValueError(f"eps_abs_stopping must be positive, got {eps_abs_stopping}.")
    if eps_rel_stopping is not None and eps_rel_stopping <= 0:
        raise ValueError(f"eps_rel_stopping must be positive, got {eps_rel_stopping}.")
    if eps_abs_stopping is not None and eps_rel_stopping is None:
        eps_rel_stopping = 0.0
    if eps_rel_stopping is not None and eps_abs_stopping is None:
        eps_abs_stopping = 0.0

    # Remove keys that are not used in the consensus function
    cfg.pop("weights", None)
    cfg.pop("weights_type", None)
    cfg.pop("normalization", None)
    cfg.pop("patch_size", None)
    cfg.pop("patch_stride", None)

    return run_admm(
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
    )


CONSENSUS_REGISTRY = {
    "mean": mean_consensus,
    "median": median_consensus,
    "admm": admm_consensus,
}


def validate_experimental_params(cfg: dict) -> ExperimentParams:
    """Validate experimental parameters for consensus algorithms.

    Args:
        cfg (dict): Configuration dictionary containing experimental parameters.

    Raises:
        ValueError: If any of the parameters are invalid.

    Returns:
        ExperimentParams: Validated experimental parameters.
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Experimental parameters must be a dictionary, got {type(cfg)}."
        )

    # Remove keys that are not used in the consensus function
    if cfg.get("epe_limit", None) is not None:
        epe_limit = cfg["epe_limit"]
        if not isinstance(epe_limit, (float, int)) or epe_limit <= 0:
            raise ValueError(f"epe_limit must be a positive number, got {epe_limit}.")

    return ExperimentParams(**cfg)
