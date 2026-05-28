"""Run a tiny supervised-training workflow with validation and checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

from example_support import (
    make_flow_files,
    make_synthetic_dataset_config,
    run_main,
    write_yaml,
)


def run_example() -> None:
    """Create temp configs for train-supervised and verify checkpoint output."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        train_files = make_flow_files(temp_dir / "train_flows", num_files=3)
        val_files = make_flow_files(temp_dir / "val_flows", num_files=1)

        val_config = make_synthetic_dataset_config(
            file_list=val_files,
            batch_size=1,
            num_batches=1,
            loop=False,
        )
        val_path = write_yaml(temp_dir / "val_dataset.yaml", val_config)

        dataset_config = make_synthetic_dataset_config(
            file_list=train_files,
            batch_size=1,
            num_batches=2,
            loop=False,
        )
        dataset_config["validation"] = {
            "dataset": str(val_path),
            "interval": 1,
            "num_batches": 1,
        }
        dataset_config["save_every"] = 1
        dataset_config["log_every"] = 1

        estimator_config = {
            "estimator": "dummy",
            "estimate_type": "flow",
            "out_dir": str(temp_dir / "training_output"),
            "config": {
                "jit": False,
                "train_type": "supervised",
            },
        }

        dataset_path = write_yaml(temp_dir / "dataset.yaml", dataset_config)
        estimator_path = write_yaml(
            temp_dir / "estimator.yaml", estimator_config
        )
        result = run_main(
            mode="train-supervised",
            estimator_path=estimator_path,
            dataset_path=dataset_path,
        )

        run_root = temp_dir / "training_output" / "dummy" / "0"
        checkpoint_root = run_root / "checkpoints" / "DummyEstimator"
        checkpoints = (
            sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
            if checkpoint_root.exists()
            else []
        )

        print("Training dataset:", dataset_path)
        print("Validation dataset:", val_path)
        print("Checkpoint root:", checkpoint_root)
        print("Checkpoint steps:", [path.name for path in checkpoints])
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())


if __name__ == "__main__":
    run_example()
