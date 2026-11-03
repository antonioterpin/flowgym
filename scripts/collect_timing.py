"""Collect a per-config ``timing.json`` for a sweep of model configs.

Sibling of ``collect_cache.py``. Where that wrapper fills the per-image
**error** cache (device-independent, reusable anywhere), this one measures the
**device-dependent** inference time and writes ``timing.json`` into each
config's cache dir -- the field ``select_ensemble.py --time-limit`` consumes.
The two are deliberately separate: error is reproducible across machines and
cached once; timing depends on the GPU/driver/load and is collected per device.

For every model config it builds the estimator with ``make_estimator`` (exactly
as ``src/main.py --mode eval`` does, so the timed forward is the one a consumer
pays for), then times the JIT-compiled forward the way ``tests/test_dis_jax.py``
speed tests do:

- a random input pair of the dataset's image shape is made **device-resident**
  (on GPU when one is available) *before* timing -- PIV/optical-flow compute
  cost is data-independent, so no data loading is needed;
- the forward is JIT-compiled and warmed (compile excluded from the result);
- timing is ``timeit.repeat(number=iters, repeat=repeat)`` and the record
  reports mean/median/min/max per-call milliseconds.

The cache dir is ``<cache-root>/<cache_id_base><suffix>/`` where ``suffix`` is
the estimator's ``get_cache_id_suffix`` -- the same dir ``collect_cache.py``
fills -- so timing and error land together.

Runs in one process (no per-config subprocess), skipping configs whose
``timing.json`` already exists; continues past a failing config unless
``--halt-on-error``.

NOTE: on a shared GPU, export ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so JAX
does not grab 75% of the device (which collides with other jobs and can fail
cuBLAS init). For trustworthy numbers, run when the GPU is otherwise idle.

Example:

    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/collect_timing.py \\
        --models 'sweep/models/*.yaml' --dataset sweep/dataset.yaml \\
        --cache-root caches/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import time
import timeit
from pathlib import Path
from typing import Any

import yaml

# Reuse the estimators_list helpers from the sibling collector (same dir).
_CC_PATH = Path(__file__).resolve().parent / "collect_cache.py"
_cc_spec = importlib.util.spec_from_file_location("collect_cache", _CC_PATH)
assert _cc_spec is not None and _cc_spec.loader is not None
_collect_cache = importlib.util.module_from_spec(_cc_spec)
_cc_spec.loader.exec_module(_collect_cache)
load_estimators_list = _collect_cache.load_estimators_list
materialize_model_configs = _collect_cache.materialize_model_configs


def build_timing_record(
    per_call_ms: list[float],
    *,
    compile_ms: float,
    warmup: int,
    iters: int,
    repeat: int,
    batch_size: int,
    device: str | None = None,
    config: dict | None = None,
    cache_id: str | None = None,
) -> dict[str, Any]:
    """Assemble a ``timing.json`` record from per-call millisecond samples.

    Args:
        per_call_ms: One per-call wall time (ms) per ``timeit`` repeat block.
        compile_ms: One-time JIT compile cost (excluded from the stats).
        warmup: Untimed warmup calls performed before measuring.
        iters: ``timeit`` ``number`` (calls per block).
        repeat: ``timeit`` ``repeat`` (number of blocks).
        batch_size: Forward batch size the times correspond to.
        device: Optional device string the run executed on.
        config: Optional estimator config dict (for traceability).
        cache_id: Optional cache id the record belongs to.

    Returns:
        A JSON-serializable record. ``mean_ms``/``median_ms``/``min_ms``/
        ``max_ms`` are the fields ``select_ensemble.py --time-stat`` selects.
    """
    record: dict[str, Any] = {
        "mean_ms": statistics.fmean(per_call_ms),
        "median_ms": statistics.median(per_call_ms),
        "min_ms": min(per_call_ms),
        "max_ms": max(per_call_ms),
        "std_ms": (
            statistics.pstdev(per_call_ms) if len(per_call_ms) > 1 else 0.0
        ),
        "compile_ms": compile_ms,
        "n_warmup": warmup,
        "n_iters": iters,
        "n_repeat": repeat,
        "batch_size": batch_size,
    }
    if device is not None:
        record["device"] = device
    if config is not None:
        record["config"] = config
    if cache_id is not None:
        record["cache_id"] = cache_id
    return record


def resolve_cache_id_base(dataset_config: dict, override: str | None) -> str:
    """Resolve the base cache id (override wins, else dataset caching block).

    Args:
        dataset_config: Parsed dataset config.
        override: Value of ``--cache-id-base`` (or None).

    Returns:
        The base cache id the per-config suffix is appended to.

    Raises:
        ValueError: If neither an override nor a dataset ``caching.cache_id``
            is available, so the cache cannot be named.
    """
    base = override or (dataset_config.get("caching") or {}).get("cache_id")
    if not base:
        raise ValueError(
            "no --cache-id-base given and the dataset config has no "
            "`caching.cache_id` to derive it from."
        )
    return base


def time_forward(
    create_state_fn: Any,
    compute_estimate_fn: Any,
    trainable_state: Any,
    img1: Any,
    img2: Any,
    *,
    warmup: int,
    iters: int,
    repeat: int,
) -> tuple[list[float], float]:
    """Time the compiled forward at the given (already on-device) inputs.

    The estimation state is built once outside the timed region, so only the
    forward (``compute_estimate_fn``) is measured. ``block_until_ready`` is
    inside the timed callable so async dispatch cannot hide device work.

    Args:
        create_state_fn: JIT state-init from ``make_estimator``.
        compute_estimate_fn: JIT forward from ``make_estimator``.
        trainable_state: Trainable state from ``make_estimator``.
        img1: First-frame batch, already device-resident.
        img2: Second-frame batch, already device-resident.
        warmup: Untimed warmup calls (the first triggers compilation).
        iters: ``timeit`` ``number``.
        repeat: ``timeit`` ``repeat``.

    Returns:
        ``(per_call_ms, compile_ms)``: one per-call ms per repeat block, and
        the one-time compile cost (first call) in ms.
    """
    import jax  # noqa: PLC0415

    state0 = create_state_fn(img1, jax.random.PRNGKey(0))
    jax.block_until_ready(state0)

    def run() -> None:
        state, _ = compute_estimate_fn(
            img2, state0, trainable_state, cache_payload=None
        )
        jax.block_until_ready(state["estimates"])

    t0 = time.perf_counter()
    run()  # first call compiles
    compile_ms = (time.perf_counter() - t0) * 1000.0
    for _ in range(max(0, warmup - 1)):
        run()

    blocks = timeit.repeat(stmt=run, number=iters, repeat=repeat)
    per_call_ms = [(block / iters) * 1000.0 for block in blocks]
    return per_call_ms, compile_ms


def time_model(
    model_path: Path,
    dataset_config: dict,
    cache_root: Path,
    cache_id_base: str,
    *,
    batch_size: int,
    warmup: int,
    iters: int,
    repeat: int,
) -> tuple[Path, dict[str, Any]]:
    """Build one estimator, time its forward, write ``timing.json``.

    Args:
        model_path: Standalone model config YAML.
        dataset_config: Parsed dataset config (supplies ``image_shape``).
        cache_root: Shared cache root.
        cache_id_base: Base cache id (suffix appended per estimator).
        batch_size: Forward batch size (timing is typically bs=1).
        warmup: Untimed warmup calls.
        iters: ``timeit`` ``number``.
        repeat: ``timeit`` ``repeat``.

    Returns:
        ``(cache_dir, record)`` for the timed config.

    Raises:
        ValueError: If the estimator exposes no state-init (no
            ``estimate_shape``), so its forward cannot be timed.
    """
    import jax  # noqa: PLC0415

    from flowgym.make import make_estimator  # noqa: PLC0415

    doc = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    height, width = dataset_config["image_shape"]
    image_shape = (batch_size, int(height), int(width))
    estimate_shape = (batch_size, int(height), int(width), 2)
    rng = jax.random.PRNGKey(0)

    trainable_state, create_state_fn, compute_estimate_fn, model = (
        make_estimator(
            doc,
            image_shape=image_shape,
            estimate_shape=estimate_shape,
            load_from=doc.get("load_from"),
            rng=rng,
        )
    )
    if create_state_fn is None:
        raise ValueError(
            f"{model_path}: estimator has no state-init (estimate_shape "
            "required for timing)."
        )

    suffix = model.get_cache_id_suffix(trainable_state)
    cache_id = f"{cache_id_base}{suffix}"
    cache_dir = cache_root / cache_id

    # Device-resident random input pair (lands on GPU when available).
    k1, k2 = jax.random.split(rng)
    img1 = jax.device_put(jax.random.uniform(k1, image_shape))
    img2 = jax.device_put(jax.random.uniform(k2, image_shape))
    jax.block_until_ready((img1, img2))

    per_call_ms, compile_ms = time_forward(
        create_state_fn,
        compute_estimate_fn,
        trainable_state,
        img1,
        img2,
        warmup=warmup,
        iters=iters,
        repeat=repeat,
    )
    record = build_timing_record(
        per_call_ms,
        compile_ms=compile_ms,
        warmup=warmup,
        iters=iters,
        repeat=repeat,
        batch_size=batch_size,
        device=str(jax.devices()[0]),
        config=doc.get("config"),
        cache_id=cache_id,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "timing.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return cache_dir, record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument list (for testing).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Collect per-config timing.json for many model configs.",
    )
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        default=None,
        help="Standalone model config YAMLs to time (one timing.json each).",
    )
    parser.add_argument(
        "--estimators-list",
        type=Path,
        nargs="+",
        default=None,
        help="estimators_list YAMLs (top-level `estimators:`); each "
        "sub-estimator is timed as its own config.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset config YAML; supplies image_shape and (optionally) the "
        "caching.cache_id used as the base cache id.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Shared cache root (same root collect_cache.py fills).",
    )
    parser.add_argument(
        "--cache-id-base",
        type=str,
        default=None,
        help="Base cache id; defaults to the dataset config's cache_id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Forward batch size to time (default: 1).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup calls before measuring (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="timeit `number`: calls per repeat block (default: 200).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="timeit `repeat`: number of blocks (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N model configs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-time configs that already have a timing.json.",
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
    """Run the timing sweep.

    Args:
        argv: Optional explicit argument list (for testing).

    Returns:
        Process exit code (0 if every config timed cleanly, 1 otherwise).
    """
    args = parse_args(argv)

    models: list[Path] = list(args.models or [])
    if args.estimators_list:
        workdir = Path(tempfile.mkdtemp(prefix="collect_timing_"))
        for list_path in args.estimators_list:
            entries = load_estimators_list(list_path)
            models.extend(materialize_model_configs(entries, workdir))
    if args.limit is not None:
        models = models[: args.limit]

    dataset_config = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    cache_id_base = resolve_cache_id_base(dataset_config, args.cache_id_base)

    import jax  # noqa: PLC0415

    print(f"[timing] device: {jax.devices()}  configs: {len(models)}")

    n_ok = 0
    n_fail = 0
    n_skip = 0
    for idx, model in enumerate(models, start=1):
        try:
            doc = yaml.safe_load(model.read_text(encoding="utf-8"))
            # Cheap skip: recompute the suffix without timing.
            from flowgym.make import make_estimator  # noqa: PLC0415

            height, width = dataset_config["image_shape"]
            ts, _, _, probe = make_estimator(
                doc,
                image_shape=(args.batch_size, int(height), int(width)),
                estimate_shape=(args.batch_size, int(height), int(width), 2),
                load_from=doc.get("load_from"),
                rng=jax.random.PRNGKey(0),
            )
            cache_dir = args.cache_root / (
                cache_id_base + probe.get_cache_id_suffix(ts)
            )
            del ts, probe
            jax.clear_caches()
            if (cache_dir / "timing.json").exists() and not args.overwrite:
                n_skip += 1
                continue

            _, record = time_model(
                model,
                dataset_config,
                args.cache_root,
                cache_id_base,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iters=args.iters,
                repeat=args.repeat,
            )
            n_ok += 1
            print(
                f"[timing] [{idx}/{len(models)}] {model.stem} "
                f"{record['mean_ms']:.4f} ms (compile "
                f"{record['compile_ms']:.0f} ms)",
                flush=True,
            )
        except Exception as exc:
            n_fail += 1
            print(
                f"[timing] [{idx}/{len(models)}] FAILED ({model}): {exc!r}",
                file=sys.stderr,
            )
            if args.halt_on_error:
                break
        finally:
            jax.clear_caches()

    print(
        f"done: {n_ok} ok, {n_skip} skipped, {n_fail} failed, "
        f"{len(models)} total"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
