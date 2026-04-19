"""Example of how to use caching with RAFT model and temporary .mat files."""

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml


def setup_configs(temp_dir: str) -> tuple[Path, Path]:
    """Create temporary .mat files and matching configs in temp directory.

    Args:
        temp_dir: Temporary directory path for data and configs.

    Returns:
        Tuple of (dataset_path, model_path) for the created configs.
    """

    data_dir = Path(temp_dir) / "test"
    data_dir.mkdir()

    mat_files: list[str] = []
    for i in range(10):
        mat_path = data_dir / f"flow_{i:03d}.mat"
        with h5py.File(mat_path, "w", libver="latest", userblock_size=512) as f:
            f.create_dataset(
                "I0",
                data=np.random.randint(0, 255, size=(256, 256), dtype=np.uint8),
            )
            f.create_dataset(
                "I1",
                data=np.random.randint(0, 255, size=(256, 256), dtype=np.uint8),
            )
            f.create_dataset(
                "V",
                data=np.random.randn(256, 256, 2).astype(np.float32),
            )

        # Write a minimal MATLAB 7.3-like header so readers detect .mat format.
        header = (
            (
                f"MATLAB 7.3 MAT-file, Platform: Python-h5py, "
                f"Created on {datetime.now():%c}"
            )
            .encode("ascii")
            .ljust(116, b" ")
        )
        header += b" " * (512 - 116)
        with open(mat_path, "r+b") as fp:
            fp.write(header)

        mat_files.append(str(mat_path))

    # 1. Load template config
    template_path = Path(
        "src/flowgym/config/piv_dataset_class1_eval_original.yaml"
    )
    with open(template_path) as f:
        dataset_config = yaml.safe_load(f)

    # 2. Override for example
    dataset_config["batch_size"] = 2  # Smaller batch for RAFT memory
    dataset_config["num_batches"] = 5
    dataset_config["loop"] = False
    dataset_config["randomize"] = False
    dataset_config["include_images"] = True
    dataset_config["flow_fields_per_batch"] = 1
    dataset_config["batches_per_flow_batch"] = 1
    dataset_config["image_shape"] = [256, 256]
    dataset_config["eval_gt"] = False

    # Point to generated .mat files.
    dataset_config["file_list"] = mat_files
    dataset_config["scheduler_class"] = ".mat"

    # Caching Configuration
    dataset_config["caching"] = {
        "root_dir": str(temp_dir),
        "cache_id": "raft_example_cache",
        "spec": {"epe": ["float32", []], "relative_epe": ["float32", []]},
        "warm_start": "index",
    }

    dataset_path = Path(temp_dir) / "dataset.yaml"
    with open(dataset_path, "w") as f:
        yaml.dump(dataset_config, f)

    # 2. Model Config
    model_config = {
        "estimator": "raft_jax",
        "estimate_type": "flow",
        "config": {
            "iters": 4,  # Low iters for speed in example
            "patch_size": 32,
            "patch_stride": 32,  # Increased stride for speed/memory
            "hidden_dim": 64,  # Small model
            "context_dim": 64,
            "corr_levels": 2,
            "corr_radius": 2,
            "train": False,
            "optimizer_config": {
                "name": "adam",
                "learning_rate": 0.0001,
            },
        },
    }

    model_path = Path(temp_dir) / "model.yaml"
    with open(model_path, "w") as f:
        yaml.dump(model_config, f)

    return dataset_path, model_path


def run_main(dataset_path: Path, model_path: Path, description: str):
    """Run src/main.py via subprocess.

    Args:
        dataset_path: Path to dataset configuration YAML.
        model_path: Path to model configuration YAML.
        description: Description of this run for logging.

    Returns:
        Duration of the run in seconds.
    """
    print(f"\n--- {description} ---")

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.main",
        "--model",
        str(model_path),
        "--dataset",
        str(dataset_path),
        "--mode",
        "eval",
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    start_time = time.time()
    result = subprocess.run(
        cmd,
        check=False,
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start_time

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"Error running {description}:")
        sys.exit(1)

    print(f"Completed in {duration:.2f} seconds.")

    # Verify that cache ID was updated
    for line in result.stderr.splitlines() + result.stdout.splitlines():
        if "Updated cache_id with model suffix" in line:
            print(f"Verified: {line.strip()}")

    return duration


def run_example():
    print("=== RAFT Model Caching Example ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        dataset_path, model_path = setup_configs(temp_dir)

        # Run 1: Cold Start
        duration_1 = run_main(dataset_path, model_path, "Run 1: Cold Start")

        # Modify config to use warm_start="all" for Run 2
        with open(dataset_path) as f:
            config = yaml.safe_load(f)

        # Enforce using loaded cache
        config["caching"]["warm_start"] = "all"

        with open(dataset_path, "w") as f:
            yaml.dump(config, f)

        # Run 2: Warm Start
        duration_2 = run_main(dataset_path, model_path, "Run 2: Warm Start")

        # Analysis
        if duration_2 < duration_1:
            speedup = duration_1 / duration_2
            print(f"\nSUCCESS: Run 2 was {speedup:.2f}x faster!")
        else:
            print(
                f"\nNote: Run 2 was not faster. "
                f"(Time 1: {duration_1:.2f}s, Time 2: {duration_2:.2f}s)"
            )


if __name__ == "__main__":
    run_example()
