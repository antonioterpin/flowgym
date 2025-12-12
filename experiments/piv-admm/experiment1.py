#!/usr/bin/env python3
"""Run a sweep over a parameter list by editing a config and launching experiments."""

import subprocess
import yaml
from pathlib import Path

# === USER SETTINGS ===
CONFIG_PATH = Path("src/flowgym/config/estimators/flow/piv_dataset/consensus/test.yaml")
PARAM_KEY = "config.experiment_params.epe_limit"  # dot notation for nested keys
PARAM_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 10000.0]

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(CONFIG_PATH),
    "--dataset",
    "src/flowgym/config/piv_dataset_class1_eval_original.yaml",
    "--mode",
    "eval",
]
CUDA_VISIBLE_DEVICES = "0"
# ======================


def set_nested_value(d: dict, dotted_key: str, value):
    """Recursively set a value in a nested dictionary given a dotted key path."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_sweep():
    """Run the parameter sweep by modifying the config and launching experiments."""
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    for val in PARAM_VALUES:
        print(f"\n=== Running sweep for {PARAM_KEY} = {val} ===")

        # Update config in memory
        cfg = base_config.copy()
        set_nested_value(cfg, PARAM_KEY, val)

        # Write temporary config
        temp_path = CONFIG_PATH.parent / f"temp_{val}.yaml"
        with open(temp_path, "w") as f:
            yaml.safe_dump(cfg, f)

        # Build command with modified config
        cmd = RUN_CMD.copy()
        cmd[cmd.index(str(CONFIG_PATH))] = str(temp_path)

        # Run the experiment
        subprocess.run(
            [
                "bash",
                "-c",
                f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} " + " ".join(cmd),
            ],
            check=True,
        )

        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_sweep()
