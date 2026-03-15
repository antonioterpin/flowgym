"""Script to replicate experiment 3.2 in https://arxiv.org/abs/2512.11695.

This script runs an ablation study over the norms of the flow weights,
in particular between L1, L2 and Huber.

To run:
    uv run python experiments/piv-admm/experiment3_2/experiment3_2.py

Results are stored in results/experiment3_2.csv
"""

import copy
import subprocess
from pathlib import Path

import yaml

# === USER SETTINGS ===
CONFIG_PATH = Path("experiments/piv-admm/experiment3_2/base_model.yaml")
PARAM_KEY = "config.experiment_params.epe_limit"  # dot notation for nested keys
TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0, 10000.0]

WEIGHTS_L1 = {
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.transformation": None,
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 150.0,
        "laplacian": 250.0,
        "divergence": 1100.0,
    },
}

WEIGHTS_L2 = {
    "config.consensus_config.flows_objective_type": "l2",
    "config.consensus_config.transformation": None,
    "config.consensus_config.solver_flows": "closed_form_l2",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 5.0,
        "laplacian": 500.0,
        "divergence": 1000.0,
    },
}

WEIGHTS_HUBER = {
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.transformation": None,
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 3.0,
        "laplacian": 30.0,
        "divergence": 1300.0,
    },
}

CONFIGS = [WEIGHTS_HUBER, WEIGHTS_L2, WEIGHTS_L1]

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(CONFIG_PATH),
    "--dataset",
    "experiments/piv-admm/experiment3_2/dataset.yaml",
    "--mode",
    "eval",
]
CUDA_VISIBLE_DEVICES = "1"
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

    for config in CONFIGS:
        objective = config["config.consensus_config.flows_objective_type"]
        solver = config["config.consensus_config.solver_flows"]
        transformation = config["config.consensus_config.transformation"]
        regularizer_weights = config[
            "config.consensus_config.regularizer_weights"
        ]
        print(
            f"\n=== Running sweep for objective={objective}, solver={solver}, transformation={transformation} ==="
        )
        for val in TAU_VALUES:
            print(f"\n=== {PARAM_KEY} = {val} ===")

            # Deep copy config
            cfg = copy.deepcopy(base_config)

            # Update required fields
            set_nested_value(cfg, "config.experiment_params.epe_limit", val)
            set_nested_value(
                cfg, "config.consensus_config.transformation", transformation
            )
            set_nested_value(
                cfg, "config.consensus_config.flows_objective_type", objective
            )
            set_nested_value(
                cfg, "config.consensus_config.solver_flows", solver
            )
            set_nested_value(
                cfg,
                "config.consensus_config.regularizer_weights",
                regularizer_weights,
            )

            # Write temp config
            trans_str = transformation if transformation is not None else "none"
            temp_path = (
                CONFIG_PATH.parent
                / f"temp_{objective}_{solver}_{trans_str}_{val}.yaml"
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
                    + "GOGGLES_PORT=2402 "
                    + " ".join(cmd),  # TODO remove GOGGLES_PORT when fixed
                ],
                check=True,
            )

            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_sweep()
