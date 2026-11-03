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
import tempfile
from pathlib import Path
from typing import Any

from flowgym.experiments import update_readme_results, validate_model_study
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
    p.add_argument(
        "--models",
        nargs="+",
        help="Run only these entries of exp.yaml's models list.",
    )
    return p.parse_args()


def run_experiment(
    exp: str,
    exp_root: str | Path = "experiments",
    *,
    selected_models: list[str] | None = None,
    debug: bool = False,
    spec_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
    dataset_overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> None:
    """Run an experiment, optionally isolating a composed sweep job.

    Args:
        exp: Experiment folder name.
        exp_root: Parent directory of experiment folders.
        selected_models: Optional subset of the model entries.
        debug: Enable debug mode in child processes.
        spec_overrides: Overrides for exp.yaml, such as seeds or mode.
        model_overrides: Nested overrides applied to each model config.
        dataset_overrides: Nested overrides applied after model-specific ones.
        output_dir: Isolated job directory, or the experiment folder by default.
        project_root: Root for relative dataset and entrypoint paths.

    Raises:
        ValueError: If the experiment or a model config is malformed.
        FileNotFoundError: If the entrypoint is missing.
    """
    root = Path.cwd() if project_root is None else project_root.resolve()
    exp_dir = (root / exp_root / exp).resolve()
    spec = deep_merge(_load_yaml(exp_dir / "exp.yaml"), spec_overrides or {})
    destination = exp_dir if output_dir is None else output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project = spec.get("project", DEFAULT_PROJECT)
    seeds = [int(seed) for seed in spec.get("seeds", [0])]
    metrics = list(spec.get("metrics", ["mean_epe"]))
    selected = selected_models or spec["models"]
    unknown = sorted(set(selected) - set(spec["models"]))
    if unknown:
        raise ValueError(f"--models not in exp.yaml: {unknown}")
    entry_path = root / spec.get("entrypoint", "src/main.py")
    if not entry_path.exists():
        raise FileNotFoundError(f"Entrypoint not found at {entry_path}")
    generated_dir = destination / "_generated"
    # Runs are separate processes: let them reuse compiled XLA programs
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/flowgym/jax")
    )
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    # Per model: its mode and config, and its dataset (path from the
    # repository root) with any overrides
    models: list[tuple[str, str, Path, dict[str, Any], dict[str, Any]]] = []
    for model in spec["models"]:
        model_cfg = deep_merge(
            _load_yaml(exp_dir / model), model_overrides or {}
        )
        validate_model_study(model_cfg, exp)
        model_cfg["project"] = project
        model_mode = model_cfg.pop("mode", spec["mode"])
        dataset = model_cfg.pop("dataset", spec.get("dataset"))
        if dataset is None:
            raise ValueError(
                f"{model} sets no dataset and exp.yaml has no default."
            )
        dataset_cfg = deep_merge(
            deep_merge(
                _load_yaml(root / dataset),
                model_cfg.pop("dataset_overrides", {}),
            ),
            dataset_overrides or {},
        )
        if output_dir is not None:
            model_cfg["out_dir"] = str(destination / "runs" / model)
        model_path = generated_dir / "models" / model
        if model in selected:
            dump_yaml(model_path, model_cfg)
        models.append((model, model_mode, model_path, model_cfg, dataset_cfg))

    def summary_path(model: str, seed: int) -> Path:
        return generated_dir / "summaries" / model / f"seed{seed}.json"

    def dataset_path(model: str, seed: int) -> Path:
        return generated_dir / "datasets" / model / f"seed{seed}.yaml"

    results_path = destination / "results.json"
    results_path.unlink(missing_ok=True)
    for seed in seeds:
        for model, model_mode, model_path, _, dataset_cfg in models:
            if model not in selected:
                continue
            dump_yaml(dataset_path(model, seed), {**dataset_cfg, "seed": seed})
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
                str(dataset_path(model, seed)),
                "--summary-out",
                str(summary),
            ]
            if debug:
                cmd.append("--debug")
            if output_dir is None:
                subprocess.check_call(cmd)
            else:
                # Every job and child owns a logging socket, including when
                # a Hydra launcher schedules several workers concurrently.
                with tempfile.TemporaryDirectory(
                    prefix="flowgym-goggles-"
                ) as log_dir:
                    env = {
                        **os.environ,
                        "GOGGLES_SOCKET": str(Path(log_dir) / "log.sock"),
                    }
                    subprocess.check_call(cmd, cwd=root, env=env)

    # Mean and population std over seeds of each listed metric, for every
    # model with all its summaries (also those run by another launcher)
    results: dict[str, dict[str, Any]] = {}
    for model, _, model_path, model_cfg, dataset_cfg in models:
        missing = [s for s in seeds if not summary_path(model, s).exists()]
        if missing:
            # e.g. a train-supervised run without a ``test:`` block
            print(f"run.py: no summary for {model} seeds {missing}; skipped.")
            continue
        # Summaries this launcher did not write must come from the same
        # configs, not from an earlier version of the experiment
        stale = model not in selected and (
            _load_yaml(model_path) != model_cfg
            or any(
                _load_yaml(dataset_path(model, s)) != {**dataset_cfg, "seed": s}
                for s in seeds
            )
        )
        if stale:
            print(f"run.py: summaries for {model} are stale; skipped.")
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
        if output_dir is None:
            update_readme_results(exp_dir / "README.md")


def main() -> None:
    """Run the original experiment CLI using the shared launcher."""
    args = parse_args()
    run_experiment(
        args.exp, args.exp_root, selected_models=args.models, debug=args.debug
    )


if __name__ == "__main__":
    main()
