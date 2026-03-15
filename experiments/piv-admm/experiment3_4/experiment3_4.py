"""Script to replicate experiment 3.4 in https://arxiv.org/abs/2512.11695.

This script runs an ablation study over the weights of the flow weights,
in particular between uniform (here called list), photometric, photograd, and gradient.

To run:
    uv run python experiments/piv-admm/experiment3_4/experiment3_4.py

Results are stored in results/experiment3_4.csv

NOTE: This script collects the metrics for both L1 and Huber objective functions, but in the
paper we only report the results for Huber.
"""

import copy
import subprocess
from pathlib import Path

import yaml

# === USER SETTINGS ===
CONFIG_PATH = Path("experiments/piv-admm/experiment3_4/base_model.yaml")
PARAM_KEY = "config.experiment_params.epe_limit"  # dot notation for nested keys
TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0, 1000.0]

L1_NO_REG_PHOTOGRAD = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 0.0,
        "divergence": 0.0,
    },
    "config.experiment_params.oracle_select_weights": False,
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "photograd",
}

L1_NO_REG_BEST = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 0.0,
        "divergence": 0.0,
    },
    "config.experiment_params.oracle_select_weights": True,
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "list",
}

L1_REG_PHOTOGRAD = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 150.0,
        "laplacian": 250.0,
        "divergence": 1100.0,
    },
    "config.experiment_params.oracle_select_weights": False,
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "photograd",
}

L1_REG_BEST = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 10.0,
        "divergence": 150.0,
    },
    "config.experiment_params.oracle_select_weights": True,
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "list",
}

HUBER_NO_REG_PHOTOGRAD = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 0.0,
        "divergence": 0.0,
    },
    "config.consensus_config.oracle_select_weights": False,
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "photograd",
}

HUBER_NO_REG_BEST = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 0.0,
        "divergence": 0.0,
    },
    "config.experiment_params.oracle_select_weights": True,
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "list",
}

HUBER_REG_PHOTOGRAD = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 3.0,
        "laplacian": 30.0,
        "divergence": 1300.0,
    },
    "config.experiment_params.oracle_select_weights": False,
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "photograd",
}

HUBER_REG_BEST = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 1.0,
        "divergence": 10.0,
    },
    "config.experiment_params.oracle_select_weights": True,
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "list",
}

COMBOS = [
    L1_NO_REG_PHOTOGRAD,
    L1_NO_REG_BEST,
    L1_REG_PHOTOGRAD,
    L1_REG_BEST,
    HUBER_NO_REG_PHOTOGRAD,
    HUBER_NO_REG_BEST,
    HUBER_REG_PHOTOGRAD,
    HUBER_REG_BEST,
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
    "experiments/piv-admm/experiment3_4/dataset.yaml",
    "--mode",
    "eval",
]
CUDA_VISIBLE_DEVICES = "0"
GOGGLES_PORT = "2403"
# ======================


def set_nested_value(
    d: dict, dotted_key: str, value: str | int | float
) -> None:
    """Recursively set a value in a nested dictionary given a dotted key path.

    Args:
        d: The dictionary to modify.
        dotted_key: Dot-separated key path (e.g., "parent.child.key").
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

    for combo in COMBOS:
        regularizer_weights = combo[
            "config.consensus_config.regularizer_weights"
        ]
        flag = combo["config.experiment_params.oracle_select_weights"]
        weights_type = combo["config.consensus_config.weights_type"]
        objective_type = combo["config.consensus_config.flows_objective_type"]
        solver_flows = combo["config.consensus_config.solver_flows"]
        print(
            f"\n=== Running sweep for weights={regularizer_weights}, flag={flag} ==="
        )
        for val in TAU_VALUES:
            print(f"\n=== {PARAM_KEY} = {val} ===")

            # Deep copy config
            cfg = copy.deepcopy(base_config)

            # Update required fields
            set_nested_value(cfg, "config.experiment_params.epe_limit", val)
            set_nested_value(
                cfg,
                "config.consensus_config.regularizer_weights",
                regularizer_weights,
            )
            set_nested_value(
                cfg, "config.experiment_params.oracle_select_weights", flag
            )
            set_nested_value(
                cfg,
                "config.consensus_config.flows_objective_type",
                objective_type,
            )
            set_nested_value(
                cfg, "config.consensus_config.solver_flows", solver_flows
            )
            set_nested_value(
                cfg, "config.consensus_config.weights_type", weights_type
            )

            regularized = (
                "reg"
                if any(v > 0.0 for v in regularizer_weights.values())
                else "noreg"
            )
            # Write temp config
            temp_path = (
                CONFIG_PATH.parent / f"temp_{regularized}_{flag}_{val}.yaml"
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
                    f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
                    + f"GOGGLES_PORT={GOGGLES_PORT} "
                    + " ".join(cmd),
                ],
                check=True,
            )

            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_sweep()
