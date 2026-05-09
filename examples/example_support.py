"""Shared helpers for self-contained FlowGym example scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write a YAML file and return its path.

    Args:
        path: Destination path for the YAML file.
        data: Dictionary to serialize.

    Returns:
        The path the file was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return path


def make_flow_files(
    directory: Path,
    *,
    num_files: int = 2,
    shape: tuple[int, int, int] = (48, 48, 2),
) -> list[str]:
    """Create small synthetic flow fields for synthpix-backed examples.

    Args:
        directory: Directory the .npy files are written to.
        num_files: How many flow fields to generate.
        shape: ``(height, width, channels)`` of each flow field; channels
            must be 2.

    Returns:
        Paths of the written .npy files, as strings.

    Raises:
        ValueError: If ``shape`` does not end in 2 channels.
    """
    directory.mkdir(parents=True, exist_ok=True)
    height, width, channels = shape
    if channels != 2:
        raise ValueError("Flow field shape must end with 2 channels.")

    ys, xs = np.mgrid[:height, :width]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    radius = radius / max(radius.max(), 1.0)

    files: list[str] = []
    for idx in range(num_files):
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[..., 0] = 1.0 + 0.15 * idx + 0.25 * np.sin(xs / 10.0)
        flow[..., 1] = 0.15 * np.cos(ys / 12.0) * (1.0 - radius)
        file_path = directory / f"flow_{idx:03d}.npy"
        np.save(file_path, flow)
        files.append(str(file_path))
    return files


def make_synthetic_dataset_config(
    *,
    file_list: list[str],
    batch_size: int = 1,
    image_shape: tuple[int, int] = (32, 32),
    num_batches: int | None = None,
    loop: bool = False,
    randomize: bool = False,
) -> dict[str, Any]:
    """Return a small synthetic dataset config suitable for local examples.

    Args:
        file_list: Paths of the flow .npy files to feed the sampler.
        batch_size: Number of samples per batch.
        image_shape: ``(height, width)`` of the synthetic images.
        num_batches: Optional cap on the number of batches; omitted from
            the config when ``None``.
        loop: Whether the sampler should loop the file list.
        randomize: Whether the sampler should randomize file ordering.

    Returns:
        A dataset configuration dict ready to serialize to YAML.
    """
    height, width = image_shape
    return {
        "seed": 0,
        "batch_size": batch_size,
        "flow_fields_per_batch": 1,
        "batches_per_flow_batch": batch_size,
        "loop": loop,
        "randomize": randomize,
        "include_images": False,
        "episode_length": 0,
        "buffer_size": 8,
        "image_shape": [height, width],
        "dt": 1.0,
        "seeding_density_range": [0.02, 0.02],
        "p_hide_img1": 0.0,
        "p_hide_img2": 0.0,
        "diameter_ranges": [[1.4, 1.8]],
        "diameter_var": 0.05,
        "intensity_ranges": [[180, 220]],
        "intensity_var": 0.01,
        "rho_ranges": [[0.4, 0.6]],
        "rho_var": 0.0,
        "noise_uniform": 0.0,
        "noise_gaussian_mean": 0.0,
        "noise_gaussian_std": 0.0,
        "velocities_per_pixel": 1.0,
        "resolution": 1.0,
        "flow_field_size": [48, 48],
        "img_offset": [8, 8],
        "min_speed_x": -2.0,
        "max_speed_x": 2.0,
        "min_speed_y": -1.0,
        "max_speed_y": 1.0,
        "output_units": "pixels",
        "scheduler_class": ".npy",
        "file_list": file_list,
        "eval_gt": False,
        **({"num_batches": num_batches} if num_batches is not None else {}),
    }


def run_main(
    *,
    mode: str,
    estimator_path: Path,
    dataset_path: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the repository CLI entrypoint for an example.

    Args:
        mode: ``--mode`` value passed to ``src/main.py``.
        estimator_path: Path to the estimator YAML config.
        dataset_path: Path to the dataset YAML config.
        extra_env: Additional environment variables to set for the call.

    Returns:
        The completed subprocess result.

    Raises:
        RuntimeError: If the CLI exits with a non-zero status.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "src/main.py",
        "--mode",
        mode,
        "--estimator",
        str(estimator_path),
        "--dataset",
        str(dataset_path),
    ]
    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_API_KEY": "example",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"FlowGym CLI example failed with exit code {result.returncode}."
        )
    return result
