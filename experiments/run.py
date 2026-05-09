"""FlowGym experiment launcher.

Usage:
  uv run python experiments/run.py --exp <experiment_name>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from flowgym.experiments import (
    validate_model_study,
    validate_scalar_tags,
)
from flowgym.utils import dump_yaml, load_configuration

DEFAULT_PROJECT = "flowgym"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = load_configuration(str(path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict YAML at {path}, got {type(data)!r}.")
    return data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment launcher."""
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--exp-root", default="experiments")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in the underlying main.py runs.",
    )
    return p.parse_args()


def main() -> None:  # noqa: PLR0912, PLR0915
    """Expand the experiment matrix and shell out to the entrypoint per cell."""
    args = parse_args()
    exp_dir = (Path(args.exp_root) / args.exp).resolve()
    spec_path = exp_dir / "exp.yaml"
    spec = _load_yaml(spec_path)

    project = spec.get("project", DEFAULT_PROJECT)
    mode = spec["mode"]
    seeds = [int(seed) for seed in spec.get("seeds", [0])]

    if not isinstance(project, str) or not project:
        raise ValueError("exp.yaml: project must be a non-empty string.")
    if not isinstance(mode, str) or not mode:
        raise ValueError("exp.yaml: mode must be a non-empty string.")

    if "dataset" in spec and "datasets" in spec:
        raise ValueError(
            "exp.yaml: Cannot specify both 'dataset' and 'datasets'."
        )
    if "dataset" in spec:
        dataset_specs = [spec["dataset"]]
    elif "datasets" in spec:
        dataset_specs = spec["datasets"]
    else:
        dataset_specs = None

    dataset_infos: list[tuple[Path, dict[str, Any]]] = []
    if dataset_specs is not None:
        for ds_spec in dataset_specs:
            if isinstance(ds_spec, str):
                ds_path = (exp_dir / ds_spec).resolve()
                ds_tags: dict[str, Any] = {}
            elif isinstance(ds_spec, dict):
                if "path" not in ds_spec:
                    raise ValueError("Dataset dict must contain 'path' field.")
                ds_path = (exp_dir / ds_spec["path"]).resolve()
                raw_tags = ds_spec.get("tags", {})
                ds_tags = (
                    validate_scalar_tags(raw_tags, "dataset.tags")
                    if raw_tags
                    else {}
                )
            else:
                raise ValueError(
                    "Dataset must be a string path or dict with 'path'."
                )

            if not ds_path.exists():
                raise FileNotFoundError(ds_path)
            dataset_infos.append((ds_path, ds_tags))

    base_datasets = {
        idx: _load_yaml(ds_path)
        for idx, (ds_path, _) in enumerate(dataset_infos)
    }
    dataset_tags = {idx: tags for idx, (_, tags) in enumerate(dataset_infos)}

    model_rel_paths = [Path(path) for path in spec["models"]]
    model_paths = [(exp_dir / path).resolve() for path in model_rel_paths]
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(model_path)

    generated_dir = exp_dir / "_generated"
    generated_models = generated_dir / "models"
    generated_datasets = generated_dir / "datasets"

    patched_models: list[tuple[Path, Path]] = []
    model_dataset_overrides: dict[Path, str] = {}
    for model_rel_path, model_path in zip(
        model_rel_paths, model_paths, strict=True
    ):
        model_cfg = _load_yaml(model_path)
        validate_model_study(model_cfg, args.exp)
        model_cfg["project"] = project

        if "dataset" in model_cfg:
            model_dataset_overrides[model_rel_path] = model_cfg.pop("dataset")

        out_model_path = generated_models / model_rel_path
        dump_yaml(out_model_path, model_cfg)
        patched_models.append((out_model_path, model_rel_path))

    per_seed_datasets: dict[int, dict[int, Path]] = {}
    if dataset_infos:
        for seed in seeds:
            per_seed_datasets[seed] = {}
            for ds_idx in range(len(dataset_infos)):
                dataset_cfg = dict(base_datasets[ds_idx])
                dataset_cfg["seed"] = seed
                if dataset_tags[ds_idx]:
                    dataset_cfg["tags"] = dataset_tags[ds_idx]
                out_dataset_path = (
                    generated_datasets / f"dataset{ds_idx}_seed{seed}.yaml"
                )
                dump_yaml(out_dataset_path, dataset_cfg)
                per_seed_datasets[seed][ds_idx] = out_dataset_path

    entry = spec.get("entrypoint", "src/main.py")
    entry_path = Path.cwd() / entry
    if not entry_path.exists():
        raise FileNotFoundError(f"Entrypoint not found at {entry_path}")

    commands: list[list[str]] = []
    for seed in seeds:
        for model_path, model_rel_path in patched_models:
            if model_rel_path in model_dataset_overrides:
                override_path = (
                    exp_dir / model_dataset_overrides[model_rel_path]
                ).resolve()
                override_dataset = dict(_load_yaml(override_path))
                override_dataset["seed"] = seed
                out_dataset_path = (
                    generated_datasets
                    / model_rel_path.parent
                    / f"{model_rel_path.stem}_seed{seed}.yaml"
                )
                dump_yaml(out_dataset_path, override_dataset)
                datasets_to_use = [out_dataset_path]
            elif dataset_infos:
                datasets_to_use = [
                    per_seed_datasets[seed][idx]
                    for idx in range(len(dataset_infos))
                ]
            else:
                raise ValueError(
                    f"Model {model_rel_path} does not specify a "
                    "'dataset' field, "
                    "and exp.yaml does not provide a default dataset."
                )

            for dataset_path in datasets_to_use:
                cmd = [
                    sys.executable,
                    str(entry_path),
                    "--mode",
                    mode,
                    "--model",
                    str(model_path),
                    "--dataset",
                    str(dataset_path),
                ]
                if args.debug:
                    cmd.append("--debug")
                commands.append(cmd)

    for command in commands:
        subprocess.check_call(command)


if __name__ == "__main__":
    main()
