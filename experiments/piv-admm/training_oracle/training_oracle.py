"""Train the learned oracle threshold estimator on PIV dataset class1.

This launches `src.main` in `train-supervised` mode, which internally calls
`src/train_supervised.py`.

Training split: `.../train`
Validation split: `.../val`

To run:
    uv run python experiments/piv-admm/training_oracle/training_oracle.py
"""

import subprocess
from pathlib import Path

# === USER SETTINGS ===
MODEL_PATH = Path("experiments/piv-admm/training_oracle/base_model.yaml")
DATASET_PATH = Path("experiments/piv-admm/training_oracle/dataset.yaml")
CUDA_VISIBLE_DEVICES = "0"
GOGGLES_PORT = "2406"
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


def run_training_oracle() -> None:
    """Run supervised training for the learned oracle estimator."""
    subprocess.run(
        [
            "bash",
            "-c",
            f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
            + f"GOGGLES_PORT={GOGGLES_PORT} "
            + " ".join(RUN_CMD),
        ],
        check=True,
    )


if __name__ == "__main__":
    run_training_oracle()
