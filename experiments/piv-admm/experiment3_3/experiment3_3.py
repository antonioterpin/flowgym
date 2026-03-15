"""Script to replicate experiment 3.3 in https://arxiv.org/abs/2512.11695.

This script runs an ablation study over the weights of the flow weights,
in particular between uniform (here called list), photometric, photograd, and gradient.

To run:
    uv run python experiments/piv-admm/experiment3_3/experiment3_3.py

Results are stored in results/experiment3_3.csv

NOTE: This script collects the metrics for both L1 and Huber objective functions, but in the
paper we only report the results for Huber.
"""

import copy
import subprocess
from pathlib import Path

import yaml

# === USER SETTINGS ===
CONFIG_PATH = Path("experiments/piv-admm/experiment3_3/base_model.yaml")
PARAM_KEY = "config.experiment_params.epe_limit"  # dot notation for nested keys
TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0, 1000.0]

WEIGHTS_L1_LIST = {
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 5.0,
        "divergence": 600.0,
    },
    "config.consensus_config.weights_type": "list",
}

WEIGHTS_L1_PHOTOMETRIC = {
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 0.15,
        "divergence": 3.0,
    },
    "config.consensus_config.weights_type": "photometric",
}

WEIGHTS_L1_GRADIENT = {
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 10.0,
        "laplacian": 30.0,
        "divergence": 2000.0,
    },
    "config.consensus_config.weights_type": "gradient",
}

WEIGHTS_HUBER_LIST = {
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 5.0,
        "divergence": 300.0,
    },
    "config.consensus_config.weights_type": "list",
}

WEIGHTS_HUBER_PHOTOMETRIC = {
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 0.2,
        "divergence": 1.0,
    },
    "config.consensus_config.weights_type": "photometric",
}

WEIGHTS_HUBER_GRADIENT = {
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 200.0,
        "laplacian": 550.0,
        "divergence": 750.0,
    },
    "config.consensus_config.weights_type": "gradient",
}

WEIGHTS_HUBER_PHOTOGRAD = {
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 3.0,
        "laplacian": 30.0,
        "divergence": 1300.0,
    },
    "config.consensus_config.weights_type": "photograd",
}

WEIGHT_L1_PHOTOGRAD = {
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 150.0,
        "laplacian": 250.0,
        "divergence": 1100.0,
    },
    "config.consensus_config.weights_type": "photograd",
}

WEIGHTS = [
    WEIGHTS_L1_LIST,
    WEIGHTS_L1_PHOTOMETRIC,
    WEIGHTS_L1_GRADIENT,
    WEIGHTS_HUBER_PHOTOGRAD,
    WEIGHT_L1_PHOTOGRAD,
]

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(CONFIG_PATH),
    "--dataset",
    "experiments/piv-admm/experiment3_3/dataset.yaml",
    "--mode",
    "eval",
]
CUDA_VISIBLE_DEVICES = "1"
GOGGLES_PORT = "2404"
# ======================


def set_nested_value(
    d: dict, dotted_key: str, value: str | int | float
) -> None:
    """Recursively set a value in a nested dictionary given a dotted key path.

    Args:
        d: The dictionary to modify.
        dotted_key: The key path in dot notation
            (e.g., "config.experiment_params.epe_limit").
        value: The value to set at the specified path.
    """
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_sweep():
    """Run the parameter sweep by modifying the config and launching experiments."""
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    for weight in WEIGHTS:
        weights_type = weight["config.consensus_config.weights_type"]
        regularizer_weights = weight[
            "config.consensus_config.regularizer_weights"
        ]
        objective_type = weight["config.consensus_config.flows_objective_type"]
        solver_flows = weight["config.consensus_config.solver_flows"]

        print(f"\n=== Running sweep for weight={weight} ===")
        for val in TAU_VALUES:
            print(f"\n=== {PARAM_KEY} = {val} ===")

            # Deep copy config
            cfg = copy.deepcopy(base_config)

            # Update required fields
            set_nested_value(cfg, "config.experiment_params.epe_limit", val)
            set_nested_value(
                cfg, "config.consensus_config.weights_type", weights_type
            )
            set_nested_value(
                cfg,
                "config.consensus_config.regularizer_weights",
                regularizer_weights,
            )
            set_nested_value(
                cfg,
                "config.consensus_config.flows_objective_type",
                objective_type,
            )
            set_nested_value(
                cfg, "config.consensus_config.solver_flows", solver_flows
            )

            # Write temp config
            temp_path = (
                CONFIG_PATH.parent
                / f"temp_{weights_type}_{objective_type}_{val}.yaml"
            )
            with open(temp_path, "w") as f:
                yaml.safe_dump(cfg, f)

            # Build command
            cmd = RUN_CMD.copy()
            cmd[cmd.index(str(CONFIG_PATH))] = str(temp_path)

            # Run
            subprocess.run(
                [
                    "bash",
                    "-c",
                    f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} GOGGLES_PORT={GOGGLES_PORT} "
                    + " ".join(cmd),
                ],
                check=True,
            )

            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_sweep()
