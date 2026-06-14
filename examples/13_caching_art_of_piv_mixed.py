"""Example of Art of PIV caching with mixed RAFT and DIS sub-estimators.

This example demonstrates Art of PIV's ability to learn which estimator
(RAFT or DIS) works best for different flow scenarios, while using
caching to avoid redundant computation across sub-models.
"""

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


def setup_configs(temp_dir: str) -> tuple[Path, Path, Path]:
    """Create temporary .mat files and matching configs in temp directory.

    Args:
        temp_dir: Temporary directory path for data and configs.

    Returns:
        Tuple of `(dataset_path, model_path, estimators_list_path)` for
        created configs.
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
    dataset_config["batch_size"] = 2  # Small batch for mixed estimators
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
    # For mixed RAFT+DIS, we have 5 sub-estimators total
    dataset_config["caching"] = {
        "root_dir": str(temp_dir),
        "cache_id": "art_of_piv_mixed_cache",
        "spec": {
            "epe_all": ["float32", [5]],  # 2 DIS + 3 RAFT estimators
            "epe_mask": ["bool", [5]],
        },
        "warm_start": "index",
    }

    dataset_path = Path(temp_dir) / "dataset.yaml"
    with open(dataset_path, "w") as f:
        yaml.dump(dataset_config, f)

    # 2. Create mixed sub-estimators list
    # Combine DIS (fast, less accurate) and RAFT (slower, more accurate)
    estimators_list = {
        "estimators": [
            # DIS Estimators - Fast and lightweight
            {
                "name": "DIS_Fast",
                "estimator": "dis_jax",
                "config": {
                    "patch_size": 7,
                    "patch_stride": 3,
                    "grad_desc_iters": 2,
                    "start_level": 0,
                    "levels": 2,
                    "use_temporal_propagation": False,
                    "jit": True,
                    "output_full_res": True,
                },
            },
            {
                "name": "DIS_Accurate",
                "estimator": "dis_jax",
                "config": {
                    "patch_size": 11,
                    "patch_stride": 2,
                    "grad_desc_iters": 4,
                    "start_level": 0,
                    "levels": 3,
                    "use_temporal_propagation": False,
                    "jit": True,
                    "output_full_res": True,
                },
            },
            # RAFT Estimators - More accurate but slower
            {
                "name": "RAFT_Fast",
                "estimator": "raft_jax",
                "config": {
                    "iters": 3,
                    "patch_size": 32,
                    "patch_stride": 32,
                    "hidden_dim": 64,
                    "context_dim": 64,
                    "corr_levels": 2,
                    "corr_radius": 2,
                    "train": False,
                    "jit": True,
                },
            },
            {
                "name": "RAFT_Medium",
                "estimator": "raft_jax",
                "config": {
                    "iters": 6,
                    "patch_size": 32,
                    "patch_stride": 24,
                    "hidden_dim": 96,
                    "context_dim": 96,
                    "corr_levels": 3,
                    "corr_radius": 3,
                    "train": False,
                    "jit": True,
                },
            },
            {
                "name": "RAFT_Accurate",
                "estimator": "raft_jax",
                "config": {
                    "iters": 8,
                    "patch_size": 32,
                    "patch_stride": 16,
                    "hidden_dim": 128,
                    "context_dim": 128,
                    "corr_levels": 4,
                    "corr_radius": 4,
                    "train": False,
                    "jit": True,
                },
            },
        ]
    }

    estimators_list_path = Path(temp_dir) / "estimators_list.yaml"
    with open(estimators_list_path, "w") as f:
        yaml.dump(estimators_list, f)

    # 3. Model Config
    model_config = {
        "estimator": "art_of_piv",
        "estimate_type": "flow",
        "config": {
            "router_config": {
                "features": [32, 64, 128],
                "kernel_sizes": [3, 3, 3],
                "strides": [2, 2, 2],
                "norm_fn": "none",
                "mlp_hidden_dim": 256,
            },
            "estimators_list_path": str(estimators_list_path),
            "use_temporal_propagation": False,
            "output_full_res": True,
            "use_last_estimate": False,
            "train_type": "supervised",
            "training_mode": "bandits",
            "optimizer_config": {
                "name": "adam",
                "learning_rate": 0.0001,
            },
            "loss_config": {
                "name": "l2",
            },
            "gamma": 0.0,
            "exploration_config": {
                "name": "epsilon_greedy",
                "schedule": {
                    "name": "constant",
                    "value": 0.1,
                },
            },
            "jit": True,
        },
    }

    model_path = Path(temp_dir) / "model.yaml"
    with open(model_path, "w") as f:
        yaml.dump(model_config, f)

    return dataset_path, model_path, estimators_list_path


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
    )
    duration = time.time() - start_time

    if result.returncode != 0:
        print(
            f"\nError running {description}: process exited with code "
            f"{result.returncode}"
        )
        sys.exit(1)

    print(f"\n✓ Completed in {duration:.2f} seconds.")

    return duration


def run_example():
    print("=== Art of PIV Mixed RAFT+DIS Caching Example ===")
    print(
        "\nThis example demonstrates Art of PIV with heterogeneous "
        "sub-estimators:\n"
        "  - 2 DIS variants (fast, lightweight dense inverse "
        "search)\n"
        "  - 3 RAFT variants (accurate, recurrent all-pairs "
        "field transforms)\n"
        "\n"
        "The router learns to select the best estimator based on "
        "image features,\n"
        "trading off between speed (DIS) and accuracy (RAFT) adaptively."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        dataset_path, model_path, estimators_list_path = setup_configs(temp_dir)

        print(f"\nEstimators list created at: {estimators_list_path}")
        print("Sub-estimators:")
        print("  1. DIS_Fast       - patch_size=7, 2 iterations")
        print("  2. DIS_Accurate   - patch_size=11, 4 iterations")
        print("  3. RAFT_Fast      - 3 refinement iters, hidden_dim=64")
        print("  4. RAFT_Medium    - 6 refinement iters, hidden_dim=96")
        print("  5. RAFT_Accurate  - 8 refinement iters, hidden_dim=128")

        # Run 1: Cold Start
        # This run will:
        # 1. Initialize all 5 sub-estimators (2 DIS + 3 RAFT)
        # 2. Compute EPEs for each sub-estimator on each sample
        # 3. Cache the EPE results for each sub-estimator separately
        # 4. Aggregate EPEs into the meta-cache (epe_all, epe_mask)
        print("\n" + "=" * 60)
        print("PHASE 1: Cold Start (Build Caches)")
        print("=" * 60)
        duration_1 = run_main(dataset_path, model_path, "Run 1: Cold Start")

        # Modify config to use warm_start="all" for Run 2
        with open(dataset_path) as f:
            config = yaml.safe_load(f)

        # Enforce using loaded cache
        config["caching"]["warm_start"] = "all"

        with open(dataset_path, "w") as f:
            yaml.dump(config, f)

        # Run 2: Warm Start
        # This run will:
        # 1. Pre-load all 5 sub-estimator caches into memory
        # 2. Build unified index for O(1) hash lookups across all sub-models
        # 3. Use cached EPEs to train router without running inference
        # 4. Demonstrate multi-algorithm caching efficiency
        print("\n" + "=" * 60)
        print("PHASE 2: Warm Start (Load from Cache)")
        print("=" * 60)
        duration_2 = run_main(dataset_path, model_path, "Run 2: Warm Start")

        # Analysis
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        if duration_2 < duration_1:
            speedup = duration_1 / duration_2
            print(f"✓ SUCCESS: Run 2 was {speedup:.2f}x faster!")
            print(
                f"\nCaching effectiveness for mixed RAFT+DIS meta-estimator:\n"
                f"  - Cold start: {duration_1:.2f}s "
                f"(computed EPEs for 5 sub-models)\n"
                f"  - Warm start: {duration_2:.2f}s (loaded from cache)\n"
                f"  - Speedup: {speedup:.2f}x\n"
                f"\n"
                f"Benefits of mixed architecture:\n"
                f"  • Router learns algorithm-specific strengths\n"
                f"  • DIS provides fast baseline estimates\n"
                f"  • RAFT handles challenging flow patterns\n"
                f"  • Caching avoids redundant computation across all methods\n"
                f"  • Hierarchical cache structure (sub-model → meta-level)\n"
            )
        else:
            print(
                f"Note: Run 2 was not faster. "
                f"(Time 1: {duration_1:.2f}s, Time 2: {duration_2:.2f}s)"
            )
            print(
                "This may happen for very small examples where cache overhead "
                "dominates."
            )


if __name__ == "__main__":
    run_example()
