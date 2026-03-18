"""Experiment 3.5: non-oracle outlier rejection for PIV-ADMM consensus.

This script runs consensus with classical outlier-rejection postprocessing
pipelines already implemented in the repository.

To run:
    uv run python experiments/piv-admm/experiment3_5/experiment3_5.py
"""

import copy
import subprocess
from pathlib import Path
from typing import Any

import yaml

# === USER SETTINGS ===
CONFIG_PATH = Path("experiments/piv-admm/experiment3_5/base_model.yaml")
DATASET_PATH = Path("experiments/piv-admm/experiment3_5/dataset.yaml")
RESULTS_DIR = Path("results/experiment3_5")

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(CONFIG_PATH),
    "--dataset",
    str(DATASET_PATH),
    "--mode",
    "eval",
]
CUDA_VISIBLE_DEVICES = "1"
GOGGLES_PORT = "2405"
# ======================

BASE_CONSENSUS = {
    "name": "dis_huber",
    "config.estimators_list_path": (
        "src/flowgym/config/estimators/flow/piv_dataset/consensus/dis_list.yaml"
    ),
    "config.consensus_config.flows_objective_type": "huber",
    "config.consensus_config.solver_flows": "closed_form_huber",
    "config.consensus_config.weights_type": "photograd",
    "config.consensus_config.regularizer_weights": {
        "smoothness": 3.0,
        "laplacian": 30.0,
        "divergence": 1300.0,
    },
}

OUTLIER_SCHEMES = [
    {
        "name": "none",
        "config.postprocess": (
            "src/flowgym/config/estimators/post-processing/common.yaml"
        ),
    },
    {
        "name": "umt_relaxed",
        "config.postprocess": [
            {
                "name": "universal_median_test",
                "r_threshold": 2.5,
                "epsilon": 0.01,
                "radius": 1,
            },
            {"name": "tile_average_interpolation", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
        ],
    },
    {
        "name": "umt_default",
        "config.postprocess": [
            {
                "name": "universal_median_test",
                "r_threshold": 2.0,
                "epsilon": 0.001,
                "radius": 2,
            },
            {"name": "laplace_interpolation", "num_iter": 256},
            {"name": "median_smoothing", "radius": 2},
            {"name": "average_smoothing", "radius": 1},
        ],
    },
    {
        "name": "umt_strict",
        "config.postprocess": [
            {
                "name": "universal_median_test",
                "r_threshold": 1.5,
                "epsilon": 0.001,
                "radius": 2,
            },
            {"name": "laplace_interpolation", "num_iter": 512},
            {"name": "median_smoothing", "radius": 2},
            {"name": "median_smoothing", "radius": 2},
        ],
    },
    {
        "name": "adaptive_local_relaxed",
        "config.postprocess": [
            {"name": "adaptive_local_filter", "n_sigma": 3.0, "radius": 1},
            {"name": "tile_average_interpolation", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
        ],
    },
    {
        "name": "adaptive_local_default",
        "config.postprocess": [
            {"name": "adaptive_local_filter", "n_sigma": 2.0, "radius": 2},
            {"name": "laplace_interpolation", "num_iter": 256},
            {"name": "median_smoothing", "radius": 2},
        ],
    },
    {
        "name": "adaptive_local_strict",
        "config.postprocess": [
            {"name": "adaptive_local_filter", "n_sigma": 1.5, "radius": 2},
            {"name": "laplace_interpolation", "num_iter": 512},
            {"name": "median_smoothing", "radius": 2},
        ],
    },
    {
        "name": "adaptive_global_relaxed",
        "config.postprocess": [
            {"name": "adaptive_global_filter", "n_sigma": 3.0},
            {"name": "tile_average_interpolation", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
        ],
    },
    {
        "name": "adaptive_global_default",
        "config.postprocess": [
            {"name": "adaptive_global_filter", "n_sigma": 2.0},
            {"name": "laplace_interpolation", "num_iter": 256},
            {"name": "median_smoothing", "radius": 2},
        ],
    },
    {
        "name": "adaptive_global_strict",
        "config.postprocess": [
            {"name": "adaptive_global_filter", "n_sigma": 1.5},
            {"name": "laplace_interpolation", "num_iter": 512},
            {"name": "median_smoothing", "radius": 2},
        ],
    },
    {
        "name": "constant_threshold_wide",
        "config.postprocess": [
            {
                "name": "constant_threshold_filter",
                "vel_min": 0.0,
                "vel_max": 3.0,
            },
            {"name": "tile_average_interpolation", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
        ],
    },
    {
        "name": "constant_threshold_tight",
        "config.postprocess": [
            {
                "name": "constant_threshold_filter",
                "vel_min": 0.0,
                "vel_max": 1.5,
            },
            {"name": "laplace_interpolation", "num_iter": 256},
            {"name": "median_smoothing", "radius": 2},
        ],
    },
]


def set_nested_value(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dotted path."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def run_sweep() -> None:
    """Run all non-oracle outlier-rejection configurations."""
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for outlier in OUTLIER_SCHEMES:
        cfg = copy.deepcopy(base_config)

        for key, value in BASE_CONSENSUS.items():
            if key == "name":
                continue
            set_nested_value(cfg, key, value)

        # Disable oracle-based rejection/selection explicitly.
        set_nested_value(cfg, "config.experiment_params.epe_limit", None)
        set_nested_value(
            cfg,
            "config.experiment_params.oracle_select_weights",
            False,
        )
        set_nested_value(
            cfg,
            "config.postprocess",
            outlier["config.postprocess"],
        )

        result_path = (
            RESULTS_DIR
            / f"{BASE_CONSENSUS['name']}__{outlier['name']}.csv"
        )
        set_nested_value(
            cfg,
            "config.experiment_params.log_path",
            str(result_path),
        )

        temp_path = (
            CONFIG_PATH.parent
            / f"temp_{BASE_CONSENSUS['name']}__{outlier['name']}.yaml"
        )
        with open(temp_path, "w") as f:
            yaml.safe_dump(cfg, f)

        cmd = RUN_CMD.copy()
        cmd[cmd.index(str(CONFIG_PATH))] = str(temp_path)

        print(f"\n=== Running outlier scheme: {outlier['name']} ===")
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
