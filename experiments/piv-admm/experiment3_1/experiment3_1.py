#!/usr/bin/env python3 experiment3_3.py
"""Script to replicate experiment 3.1 in https://arxiv.org/abs/2512.11695.

This script launches experiments for each combination and tau value.

To run:
    uv run python experiments/piv-admm/experiment3_1/experiment3_1.py

Results are stored in results/experiment3_1.csv

NOTE: This script collects the metrics for both L1 and Huber
objective functions, but in the paper we only report the results for Huber.
"""

import copy
import subprocess
from pathlib import Path
from typing import Any

import yaml

# === USER SETTINGS ===
CONFIG_PATH = Path("experiments/piv-admm/experiment3_1/base_model.yaml")
PARAM_KEY = "config.experiment_params.epe_limit"
TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0, 1000.0]
BASE_CONFIG_FOLDER = "src/flowgym/config/estimators/flow/piv_dataset/consensus"
USE_LEARNED_ORACLE_REJECTOR_FOR_DIS = True
LEARNED_ORACLE_TRAINED_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 1.0]
LEARNED_ORACLE_CHECKPOINTS_ROOT = Path("results/training_oracle/imgpair_0")
LEARNED_ORACLE_MASK_THRESHOLD = 0.5
LEARNED_ORACLE_FEATURES = [16, 32]
LEARNED_ORACLE_DIS_ESTIMATORS_LIST_PATH = f"{BASE_CONFIG_FOLDER}/dis_list.yaml"

COMBO_FARNEBACK_L1 = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 100.0,
        "laplacian": 10.0,
        "divergence": 1000.0,
    },
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "photograd",
    "config.estimators_list_path": (
        "src/flowgym/config/estimators/flow/piv_dataset/"
        + "consensus/farneback_list.yaml"
    ),
    "config.experiment_params.baseline_performance": {
        "min_epe": 0.10687,
        "max_epe": 0.32995,
        "mean_epe": 0.20248,
        "min_relative_epe": 0.23998,
        "mean_relative_epe": 12.03773,
        "max_relative_epe": 41.13775,
    },
}

COMBO_MIX_L1 = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 10.0,
        "laplacian": 25.0,
        "divergence": 1800.0,
    },
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "photograd",
    "config.estimators_list_path": f"{BASE_CONFIG_FOLDER}/mix_list.yaml",
    "config.experiment_params.baseline_performance": {
        "min_epe": 0.08859,
        "max_epe": 0.24117,
        "mean_epe": 0.15316,
        "min_relative_epe": 0.11474,
        "mean_relative_epe": 12.70115,
        "max_relative_epe": 44.74119,
    },
}

COMBO_FARNEBACK_HUBER = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 15.0,
        "laplacian": 20.0,
        "divergence": 700.0,
    },
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "photograd",
    "config.estimators_list_path": f"{BASE_CONFIG_FOLDER}/farneback_list.yaml",
    "config.experiment_params.baseline_performance": {
        "min_epe": 0.10687,
        "max_epe": 0.32995,
        "mean_epe": 0.20248,
        "min_relative_epe": 0.23998,
        "mean_relative_epe": 12.03773,
        "max_relative_epe": 41.13775,
    },
}

COMBO_MIX_HUBER = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 0.0,
        "laplacian": 25.0,
        "divergence": 1200.0,
    },
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "photograd",
    "config.estimators_list_path": f"{BASE_CONFIG_FOLDER}/mix_list.yaml",
    "config.experiment_params.baseline_performance": {
        "min_epe": 0.08859,
        "max_epe": 0.24117,
        "mean_epe": 0.15316,
        "min_relative_epe": 0.11474,
        "mean_relative_epe": 12.70115,
        "max_relative_epe": 44.74119,
    },
}

COMBO_DIS_L1 = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 150.0,
        "laplacian": 250.0,
        "divergence": 1100.0,
    },
    "config.consensus_config.flows_objective_type": "l1",
    "config.consensus_config.solver_flows": "closed_form_l1",
    "config.consensus_config.weights_type": "photograd",
    "config.estimators_list_path": LEARNED_ORACLE_DIS_ESTIMATORS_LIST_PATH,
    "config.experiment_params.baseline_performance": {
        "min_epe": 0.08859,
        "max_epe": 0.24117,
        "mean_epe": 0.15316,
        "min_relative_epe": 0.11474,
        "mean_relative_epe": 12.70115,
        "max_relative_epe": 44.74119,
    },
}

COMBO_DIS_HUBER = {
    "config.consensus_config.regularizer_weights": {
        "smoothness": 3.0,
        "laplacian": 30.0,
        "divergence": 1300.0,
    },
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "photograd",
    "config.estimators_list_path": LEARNED_ORACLE_DIS_ESTIMATORS_LIST_PATH,
    "config.experiment_params.baseline_performance": {
        "min_epe": 0.08859,
        "max_epe": 0.24117,
        "mean_epe": 0.15316,
        "min_relative_epe": 0.11474,
        "mean_relative_epe": 12.70115,
        "max_relative_epe": 44.74119,
    },
}

COMBOS = [
    COMBO_FARNEBACK_L1,
    COMBO_FARNEBACK_HUBER,
    COMBO_MIX_L1,
    COMBO_MIX_HUBER,
    COMBO_DIS_L1,
    COMBO_DIS_HUBER,
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
    "experiments/piv-admm/experiment3_1/dataset.yaml",
    "--mode",
    "eval",
]
CUDA_VISIBLE_DEVICES = "1"
GOGGLES_PORT = "2402"
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


def _tau_tag(tau: float) -> str:
    """Format tau for path names."""
    return str(tau).replace(".", "_")


def _checkpoint_has_saved_steps(checkpoint_dir: Path) -> bool:
    """Return True when an Orbax checkpoint directory has at least one step."""
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return False
    return any(
        entry.is_dir() and entry.name.isdigit()
        for entry in checkpoint_dir.iterdir()
    )


def inject_dis_learned_oracle_postprocess(
    estimators_cfg: dict[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Attach learned-oracle postprocess to each DIS sub-estimator."""
    out = copy.deepcopy(estimators_cfg)
    estimators = out.get("estimators", [])
    if not isinstance(estimators, list):
        raise ValueError(
            "estimators list must be present in estimators config."
        )

    dis_estimators_count = sum(
        1 for estimator in estimators if estimator.get("estimator") == "dis_jax"
    )
    dis_idx = 0

    for estimator_cfg in estimators:
        if estimator_cfg.get("estimator") != "dis_jax":
            continue
        est_config = estimator_cfg.setdefault("config", {})
        if not isinstance(est_config, dict):
            raise ValueError(
                f"Estimator config must be a dict, got {type(est_config)}."
            )

        learned_step = {
            "name": "learned_oracle_threshold",
            "threshold_value": LEARNED_ORACLE_MASK_THRESHOLD,
            "load_from": str(checkpoint_dir),
            "features": LEARNED_ORACLE_FEATURES,
            "estimator_index": dis_idx,
            "estimator_count": dis_estimators_count,
        }

        existing_postprocess = est_config.get("postprocess", [])
        if isinstance(existing_postprocess, list):
            est_config["postprocess"] = [learned_step, *existing_postprocess]
        else:
            est_config["postprocess"] = [learned_step]
        dis_idx += 1

    return out


def run_sweep():
    """Run the parameter sweep by modifying the config dynamically."""
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    for combo in COMBOS:
        regularizer_weights = combo[
            "config.consensus_config.regularizer_weights"
        ]
        weights_type = combo["config.consensus_config.weights_type"]
        objective_type = combo["config.consensus_config.flows_objective_type"]
        solver_flows = combo["config.consensus_config.solver_flows"]
        path = combo["config.estimators_list_path"]
        baseline_performance = combo[
            "config.experiment_params.baseline_performance"
        ]

        print(f"\n=== Running sweep for algos={combo} ===")
        for val in TAU_VALUES:
            print(f"\n=== {PARAM_KEY} = {val} ===")
            temp_estimators_path: Path | None = None

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
            estimators_path = path

            should_use_learned_dis_rejector = (
                USE_LEARNED_ORACLE_REJECTOR_FOR_DIS
                and Path(path).name
                == Path(LEARNED_ORACLE_DIS_ESTIMATORS_LIST_PATH).name
                and val in LEARNED_ORACLE_TRAINED_THRESHOLDS
            )
            if should_use_learned_dis_rejector:
                checkpoint_dir = (
                    LEARNED_ORACLE_CHECKPOINTS_ROOT
                    / f"tau_{_tau_tag(val)}"
                    / "learned_oracle_threshold"
                    / "0"
                    / "checkpoints"
                    / "LearnedOracleThresholdEstimator"
                )
                if _checkpoint_has_saved_steps(checkpoint_dir):
                    with open(path) as f:
                        base_estimators_cfg = yaml.safe_load(f)
                    dis_estimators_cfg = inject_dis_learned_oracle_postprocess(
                        base_estimators_cfg, checkpoint_dir
                    )
                    temp_estimators_path = (
                        CONFIG_PATH.parent
                        / (
                            "temp_estimators_dis_learned_oracle_"
                            f"{objective_type}_{_tau_tag(val)}.yaml"
                        )
                    )
                    with open(temp_estimators_path, "w") as f:
                        yaml.safe_dump(dis_estimators_cfg, f)
                    estimators_path = str(temp_estimators_path)
                    print(
                        "Using learned-oracle outlier rejector for DIS "
                        f"(tau={val}, checkpoint={checkpoint_dir})."
                    )
                else:
                    print(
                        "Learned-oracle checkpoint not found for "
                        f"tau={val} at {checkpoint_dir}; "
                        "falling back to baseline DIS estimators."
                    )

            set_nested_value(
                cfg, "config.estimators_list_path", estimators_path
            )
            set_nested_value(
                cfg,
                "config.experiment_params.baseline_performance",
                baseline_performance,
            )

            final_part_path = (
                Path(estimators_path).name.replace(".yaml", "")
            )
            # Write temp config
            temp_path = (
                CONFIG_PATH.parent
                / f"temp_{final_part_path}_{objective_type}_{val}.yaml"
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
            if temp_estimators_path is not None:
                temp_estimators_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_sweep()
