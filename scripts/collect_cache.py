"""Fill per-image error caches for many model configs over one dataset.

Estimator-agnostic sweep producer. For each candidate model config it runs
``src/main.py --mode eval`` with a shared ``--cache-root``; the dataset
config supplies the caching ``spec`` (e.g. ``{epe: [float32, []]}``), and
each estimator's ``get_cache_id_suffix`` keeps the per-config caches in
separate ``<cache-root>/<cache_id>/`` directories. Nothing here is specific
to any algorithm -- DIS, RAFT, openpiv, art_of_piv all flow through the same
path.

Runs are independent subprocesses (so JIT/GPU memory is released between
configs) and the sweep continues past a failing config unless
``--halt-on-error`` is set. A one-line summary is printed at the end.

Example:

    uv run python scripts/collect_cache.py \\
        --models experiments/.../models/dis_models_*.yaml \\
        --dataset tune.yaml \\
        --cache-root caches/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

DEFAULT_RUNNER = "uv run python"


def load_estimators_list(path: Path) -> list[dict]:
    """Load the ``estimators:`` entries from an estimators_list YAML.

    This is the multi-estimator format consumed by ``art_of_piv`` (e.g.
    ``dis_models_20.yaml``): a top-level ``estimators:`` key holding a list
    of ``{name, estimator, estimate_type, config}`` dicts.

    Args:
        path: Path to the estimators_list YAML.

    Returns:
        The list of estimator-entry dicts.

    Raises:
        ValueError: If the file has no non-empty ``estimators:`` list.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    estimators = data.get("estimators") if isinstance(data, dict) else None
    if not estimators:
        raise ValueError(f"{path} has no non-empty `estimators:` list")
    return estimators


def materialize_model_configs(
    entries: list[dict],
    workdir: Path,
) -> list[Path]:
    """Write estimators_list entries as standalone model config YAMLs.

    Each entry is emitted as a ``{estimator, estimate_type, config}`` YAML
    (the ``name`` key, which is list-only metadata, is dropped) so it can be
    passed to ``src/main.py --model``.

    Args:
        entries: Estimator-entry dicts from :func:`load_estimators_list`.
        workdir: Directory to write the per-entry YAMLs into.

    Returns:
        Paths to the written model config YAMLs, in entry order.
    """
    paths: list[Path] = []
    for idx, entry in enumerate(entries):
        model = {k: v for k, v in entry.items() if k != "name"}
        raw_stem = str(entry.get("name", f"model_{idx}"))
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_stem)
        out = workdir / f"{stem}.yaml"
        out.write_text(yaml.dump(model), encoding="utf-8")
        paths.append(out)
    return paths


def build_eval_command(
    runner: tuple[str, ...],
    model: Path,
    dataset: Path,
    cache_root: Path,
    cache_id: str | None = None,
) -> list[str]:
    """Build the ``src/main.py --mode eval`` command for one model config.

    Args:
        runner: Interpreter prefix tokens (e.g. ``("uv", "run", "python")``).
        model: Path to the model config YAML.
        dataset: Path to the dataset config YAML (carries the caching spec).
        cache_root: Shared cache root for the sweep.
        cache_id: Optional base cache id forwarded to main.py.

    Returns:
        The argument list to hand to :func:`subprocess.run`.
    """
    cmd = [
        *runner,
        "src/main.py",
        "--mode",
        "eval",
        "--estimator",
        str(model),
        "--dataset",
        str(dataset),
        "--cache-root",
        str(cache_root),
    ]
    if cache_id is not None:
        cmd += ["--cache-id", cache_id]
    return cmd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument list (for testing).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fill error caches for many model configs over a dataset.",
    )
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        default=None,
        help="Standalone model config YAMLs to evaluate (one cache each).",
    )
    parser.add_argument(
        "--estimators-list",
        type=Path,
        nargs="+",
        default=None,
        help="estimators_list YAMLs (a top-level `estimators:` list, e.g. "
        "dis_models_20.yaml). Every sub-estimator is collected as its own "
        "cache; nested lists dedupe by cache_id, so the superset suffices.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset config YAML; must carry a caching block with a spec.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Shared output root for the per-config caches.",
    )
    parser.add_argument(
        "--cache-id-base",
        type=str,
        default=None,
        help="Base cache id shared by all configs (the per-config suffix is "
        "still appended). Defaults to the dataset config's cache_id.",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default=DEFAULT_RUNNER,
        help="Interpreter prefix for src/main.py "
        f"(default: {DEFAULT_RUNNER!r}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N model configs.",
    )
    parser.add_argument(
        "--halt-on-error",
        action="store_true",
        help="Stop at the first failing config instead of continuing.",
    )
    args = parser.parse_args(argv)
    if not args.models and not args.estimators_list:
        parser.error("provide --models and/or --estimators-list")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the sweep.

    Args:
        argv: Optional explicit argument list (for testing).

    Returns:
        Process exit code (0 if every config succeeded, 1 otherwise).
    """
    args = _parse_args(argv)

    models: list[Path] = list(args.models or [])
    if args.estimators_list:
        workdir = Path(tempfile.mkdtemp(prefix="collect_cache_"))
        for list_path in args.estimators_list:
            entries = load_estimators_list(list_path)
            models.extend(materialize_model_configs(entries, workdir))

    if args.limit is not None:
        models = models[: args.limit]
    runner = tuple(args.runner.split())

    n_ok = 0
    n_fail = 0
    for idx, model in enumerate(models, start=1):
        cmd = build_eval_command(
            runner,
            model,
            args.dataset,
            args.cache_root,
            args.cache_id_base,
        )
        print(
            f"[{idx}/{len(models)}] {model} -> {args.cache_root}",
            flush=True,
        )
        returncode = subprocess.run(cmd, check=False).returncode
        if returncode == 0:
            n_ok += 1
        else:
            n_fail += 1
            print(f"  FAILED ({model}, exit {returncode})", file=sys.stderr)
            if args.halt_on_error:
                break

    print(f"done: {n_ok} ok, {n_fail} failed, {len(models)} total")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
