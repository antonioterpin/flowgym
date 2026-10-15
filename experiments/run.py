"""FlowGym experiment launcher.

Usage:
  uv run python experiments/run.py --exp <experiment_name>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from flowgym.experiments import validate_model_study
from flowgym.utils import deep_merge, dump_yaml, load_configuration

DEFAULT_PROJECT = "flowgym"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = load_configuration(str(path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict YAML at {path}, got {type(data)!r}.")
    return data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment launcher.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--exp-root", default="experiments")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in the underlying main.py runs.",
    )
    return p.parse_args()


def main() -> None:
    """Run every model of the experiment once per seed, then collect results.

    Raises:
        ValueError: If ``exp.yaml`` or a model config is malformed.
        FileNotFoundError: If the entrypoint is missing.
    """
    args = parse_args()
    exp_dir = (Path(args.exp_root) / args.exp).resolve()
    spec = _load_yaml(exp_dir / "exp.yaml")
    project = spec.get("project", DEFAULT_PROJECT)
    seeds = [int(seed) for seed in spec.get("seeds", [0])]
    metrics = list(spec.get("metrics", ["mean_epe"]))
    entry_path = Path.cwd() / spec.get("entrypoint", "src/main.py")
    if not entry_path.exists():
        raise FileNotFoundError(f"Entrypoint not found at {entry_path}")
    generated_dir = exp_dir / "_generated"
    # Runs are separate processes: let them reuse compiled XLA programs
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/flowgym/jax")
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    # Per model: its mode, and its dataset with any overrides applied
    models: list[tuple[str, str, Path, dict[str, Any]]] = []
    for model in spec["models"]:
        model_cfg = _load_yaml(exp_dir / model)
        validate_model_study(model_cfg, args.exp)
        model_cfg["project"] = project
        model_mode = model_cfg.pop("mode", spec["mode"])
        dataset = model_cfg.pop("dataset", spec.get("dataset"))
        if dataset is None:
            raise ValueError(
                f"{model} sets no dataset and exp.yaml has no default."
            )
        dataset_cfg = deep_merge(
            _load_yaml(exp_dir / dataset),
            model_cfg.pop("dataset_overrides", {}),
        )
        model_path = generated_dir / "models" / model
        dump_yaml(model_path, model_cfg)
        models.append((model, model_mode, model_path, dataset_cfg))

    def summary_path(model: str, seed: int) -> Path:
        return generated_dir / "summaries" / model / f"seed{seed}.json"

    results_path = exp_dir / "results.json"
    results_path.unlink(missing_ok=True)
    for seed in seeds:
        for model, model_mode, model_path, dataset_cfg in models:
            dataset_path = (
                generated_dir / "datasets" / model / f"seed{seed}.yaml"
            )
            dump_yaml(dataset_path, {**dataset_cfg, "seed": seed})
            summary = summary_path(model, seed)
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.unlink(missing_ok=True)
            cmd = [
                sys.executable,
                str(entry_path),
                "--mode",
                model_mode,
                "--estimator",
                str(model_path),
                "--dataset",
                str(dataset_path),
                "--summary-out",
                str(summary),
            ]
            if args.debug:
                cmd.append("--debug")
            subprocess.check_call(cmd)

    # Mean and population std over seeds of each listed metric
    results: dict[str, dict[str, Any]] = {}
    for model, *_ in models:
        missing = [s for s in seeds if not summary_path(model, s).exists()]
        if missing:
            # e.g. a train-supervised run without a ``test:`` block
            print(f"run.py: no summary for {model} seeds {missing}; skipped.")
            continue
        summaries = {
            str(s): json.loads(summary_path(model, s).read_text())
            for s in seeds
        }
        results[model] = {}
        for metric in metrics:
            values = {
                s: summary.get(metric) for s, summary in summaries.items()
            }
            if all(isinstance(v, int | float) for v in values.values()):
                results[model][metric] = {
                    "mean": statistics.mean(values.values()),
                    "std": statistics.pstdev(values.values()),
                    "seeds": values,
                }
    if results:
        results_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
