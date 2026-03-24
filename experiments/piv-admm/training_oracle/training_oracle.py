"""Train the learned oracle threshold estimator on PIV dataset class1.

This launches `src.main` in `train-supervised` mode, which internally calls
`src/train_supervised.py`.

Training split: `.../train`
Validation split: `.../val`

To run:
    uv run python experiments/piv-admm/training_oracle/training_oracle.py
"""

import copy
import subprocess
from pathlib import Path
from typing import Any

import yaml

# === USER SETTINGS ===
MODEL_PATH = Path("experiments/piv-admm/training_oracle/base_model.yaml")
DATASET_PATH = Path("experiments/piv-admm/training_oracle/dataset.yaml")
CUDA_VISIBLE_DEVICES = "0"
GOGGLES_PORT = "2406"
# Oracle-threshold axis used in previous PIV-ADMM experiments.
ORACLE_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 1.0]
# ======================

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(MODEL_PATH),
    "--dataset",
    str(DATASET_PATH),
    "--mode",
    "train-supervised",
]


def set_nested_value(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dotted path."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _tau_tag(tau: float) -> str:
    return str(tau).replace(".", "_")


def run_training_oracle() -> None:
    """Run supervised training for all oracle-threshold values."""
    with open(MODEL_PATH) as f:
        base_model_cfg = yaml.safe_load(f)

    for tau in ORACLE_THRESHOLDS:
        cfg = copy.deepcopy(base_model_cfg)
        set_nested_value(cfg, "config.oracle_epe_threshold", tau)

        tag = _tau_tag(tau)
        set_nested_value(cfg, "out_dir", f"results/training_oracle/tau_{tag}")
        set_nested_value(cfg, "run_name", f"piv_admm_training_oracle_tau_{tag}")

        temp_model_path = (
            MODEL_PATH.parent / f"temp_model_oracle_threshold_{tag}.yaml"
        )
        with open(temp_model_path, "w") as f:
            yaml.safe_dump(cfg, f)

        cmd = RUN_CMD.copy()
        cmd[cmd.index(str(MODEL_PATH))] = str(temp_model_path)

        print(f"=== Running oracle_epe_threshold={tau} ===")
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

        temp_model_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_training_oracle()
