"""Run a tiny density-estimation evaluation workflow end to end."""

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
    """Create local configs and run density evaluation through the CLI."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        flow_files = make_flow_files(temp_dir / "flows", num_files=2)

        dataset_config = make_synthetic_dataset_config(
            file_list=flow_files,
            batch_size=1,
            num_batches=1,
            loop=False,
        )
        estimator_config = {
            "estimator": "simple",
            "estimate_type": "density",
            "estimate_shape": [1],
            "config": {
                "jit": False,
                "threshold": 100,
            },
        }

        dataset_path = write_yaml(temp_dir / "dataset.yaml", dataset_config)
        estimator_path = write_yaml(temp_dir / "estimator.yaml", estimator_config)
        result = run_main(
            mode="eval",
            estimator_path=estimator_path,
            dataset_path=dataset_path,
        )

        print("Dataset config:", dataset_path)
        print("Estimator config:", estimator_path)
        print("CLI command: uv run python src/main.py --mode eval ...")
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())


if __name__ == "__main__":
    run_example()
