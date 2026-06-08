"""Select a K-subset of candidates minimizing an aggregate per-image error.

Algorithm-agnostic ensemble selection over a sweep cache: any cache whose
per-candidate subdirs hold ``data/part-*.parquet`` rows with a ``key``
column and a scalar error column (``--metric``, default ``epe``) works,
regardless of which estimator produced it. Each subdir is one candidate.

The script picks a subset ``S`` (|S| = K) that minimizes the aggregator
(mean or median) over the per-image best error,

    cost(S) = aggregator_i( min_{a in S} e_{a, i} ),

optionally subject to a per-candidate inference-time bound. A greedy
solver runs always; an exact MILP (HiGHS via :func:`scipy.optimize.milp`)
is available for the mean aggregator with ``--exact``.

Timing is optional: a candidate only needs a ``timing.json`` when a
finite ``--time-limit`` is requested (timing is device-dependent and
collected separately); error is device-independent and always present.

Example:

    uv run python scripts/select_ensemble.py \\
        --cache-root dis_sweep_caches --K 3 --exact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

# DIS serializes its `preset` enum by name in get_config()/timing.json, but
# the estimator constructor only accepts an int (or PresetType). Map known
# names back to ints so exported configs re-load directly.
_PRESET_NAME_TO_INT = {
    "ULTRAFAST": 0,
    "FAST": 1,
    "MEDIUM": 2,
    "HIGH_QUALITY": 3,
}


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Coerce serialized enum values back to a re-loadable form.

    Args:
        config: A candidate's stored estimator config.

    Returns:
        A shallow copy with a string ``preset`` mapped to its int value
        (other values are left untouched).
    """
    cfg = dict(config)
    preset = cfg.get("preset")
    if isinstance(preset, str) and preset in _PRESET_NAME_TO_INT:
        cfg["preset"] = _PRESET_NAME_TO_INT[preset]
    return cfg


def _export_models(
    summary: dict[str, Any],
    path: Path,
    estimator: str,
    estimate_type: str,
) -> tuple[int, int]:
    """Write a selection's configs as a collect-ready estimators_list YAML.

    The output is the ``estimators:`` format consumed by
    ``collect_cache.py --estimators-list``, so a chosen subset can be
    re-collected on other splits (train/val/test) directly. Candidates
    whose cache stored no ``config`` (e.g. caches written with only a
    ``meta.json``) are skipped.

    Args:
        summary: A selection summary from :func:`_summarize_selection`.
        path: Output YAML path.
        estimator: ``estimator`` field for each emitted entry.
        estimate_type: ``estimate_type`` field for each emitted entry.

    Returns:
        A pair ``(written, skipped)`` counting emitted and config-less
        entries.
    """
    entries: list[dict[str, Any]] = []
    skipped = 0
    for entry in summary["selected"]:
        config = entry.get("config") or {}
        if not config:
            skipped += 1
            continue
        entries.append(
            {
                "name": entry["cache_id"],
                "estimator": estimator,
                "estimate_type": estimate_type,
                "config": _normalize_config(config),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump({"estimators": entries}, fh, sort_keys=False)
    return len(entries), skipped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument list (for testing).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Select a K-subset of candidates minimizing per-image "
        "error.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Root directory of the sweep cache (one subdir per candidate).",
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Subset size.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="epe",
        help="Parquet column holding the per-image scalar error to "
        "minimize (default: epe).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=math.inf,
        help="Per-candidate time bound (same units as --time-stat). When "
        "finite, candidates without a timing.json are dropped.",
    )
    parser.add_argument(
        "--aggregator",
        choices=("mean", "median"),
        default="mean",
        help="How to aggregate the per-image best error across images.",
    )
    parser.add_argument(
        "--time-stat",
        choices=("mean_ms", "median_ms", "min_ms", "max_ms"),
        default="mean_ms",
        help="Which timing.json field to compare against --time-limit.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Run exact MILP after greedy (only supported for mean).",
    )
    parser.add_argument(
        "--milp-timeout",
        type=float,
        default=300.0,
        help="HiGHS wall-time cap in seconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to dump the full result JSON.",
    )
    parser.add_argument(
        "--export-models",
        type=Path,
        default=None,
        help="Write the selected subset's configs as an estimators_list "
        "YAML (consumable by collect_cache.py --estimators-list) to "
        "re-collect on other splits.",
    )
    parser.add_argument(
        "--export-estimator",
        type=str,
        default="dis_jax",
        help="`estimator` field for --export-models entries (default: "
        "dis_jax).",
    )
    parser.add_argument(
        "--export-estimate-type",
        type=str,
        default="flow",
        help="`estimate_type` field for --export-models entries.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Extra logging.",
    )
    return parser.parse_args(argv)


def _load_candidate(
    subdir: Path,
    metric: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load a single candidate's metadata and per-image error rows.

    Args:
        subdir: Path to the candidate subdirectory.
        metric: Parquet column holding the per-image scalar error.

    Returns:
        A triple ``(record, keys, errors)`` where ``record`` merges the
        candidate's ``meta.json`` and (optional) ``timing.json`` payloads,
        ``keys`` is a 1-D ``uint64`` array of image keys, and ``errors``
        is a 1-D ``float32`` array in the same order as ``keys``.

    Raises:
        FileNotFoundError: If no data shards are present.
    """
    record: dict[str, Any] = {}
    meta_path = subdir / "meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as fh:
            record.update(json.load(fh))
    timing_path = subdir / "timing.json"
    if timing_path.is_file():
        with timing_path.open("r", encoding="utf-8") as fh:
            record.update(json.load(fh))

    data_dir = subdir / "data"
    shards = sorted(data_dir.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {data_dir}")

    key_chunks: list[np.ndarray] = []
    err_chunks: list[np.ndarray] = []
    for shard in shards:
        table = pq.read_table(shard, columns=["key", metric])
        key_chunks.append(table.column("key").to_numpy())
        err_chunks.append(
            table.column(metric).to_numpy().astype(np.float32, copy=False),
        )
    keys = np.concatenate(key_chunks)
    errors = np.concatenate(err_chunks).astype(np.float32, copy=False)
    return record, keys, errors


def _load_cache(
    cache_root: Path,
    metric: str,
    verbose: bool,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Load the entire sweep cache into aligned arrays.

    A subdirectory is a candidate iff it holds ``data/part-*.parquet``
    shards; ``timing.json`` is optional (only needed for a finite time
    bound) and ``meta.json`` supplies the cache id when present.

    Args:
        cache_root: Root directory of the sweep cache.
        metric: Parquet column holding the per-image scalar error.
        verbose: Whether to print per-candidate progress.

    Returns:
        A tuple ``(E, cache_ids, records)`` where ``E`` has shape
        ``(N, M)`` of ``float32`` errors, ``cache_ids`` is the list of
        cache ids in row order, and ``records`` holds the merged
        meta/timing payload per candidate. The caller picks the timing
        stat to filter on.

    Raises:
        RuntimeError: If no candidates with data shards were found.
    """
    all_subdirs = sorted(p for p in cache_root.iterdir() if p.is_dir())
    subdirs = [p for p in all_subdirs if list((p / "data").glob("part-*"))]
    skipped = [p for p in all_subdirs if p not in subdirs]
    if skipped:
        print(
            f"warning: skipped {len(skipped)} subdir(s) under {cache_root} "
            "without data shards (likely partial / interrupted runs)",
            file=sys.stderr,
        )
        if verbose:
            for p in skipped[:20]:
                print(f"  - {p.name}", file=sys.stderr)
            if len(skipped) > 20:
                print(f"  ... ({len(skipped) - 20} more)", file=sys.stderr)
    if not subdirs:
        raise RuntimeError(f"no candidates with data shards under {cache_root}")

    # Load every candidate, then keep only those sharing the majority key
    # set. A union of caches can contain partial / interrupted ones (fewer
    # rows); rather than aborting the whole selection, those are dropped
    # with a warning so the consistent majority is still usable.
    loaded: list[tuple[Path, dict[str, Any], np.ndarray, np.ndarray]] = []
    sigs: list[str] = []
    for idx, subdir in enumerate(subdirs):
        record, keys, errors = _load_candidate(subdir, metric)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        loaded.append((subdir, record, sorted_keys, errors[order]))
        sigs.append(hashlib.md5(sorted_keys.tobytes()).hexdigest())
        if verbose and (idx + 1) % 100 == 0:
            print(f"loaded {idx + 1}/{len(subdirs)} candidates", flush=True)

    majority_sig = Counter(sigs).most_common(1)[0][0]
    dropped = [
        (sd, sk)
        for (sd, _, sk, _), sig in zip(loaded, sigs, strict=True)
        if sig != majority_sig
    ]
    kept = [
        item
        for item, sig in zip(loaded, sigs, strict=True)
        if sig == majority_sig
    ]
    if dropped:
        print(
            f"warning: dropped {len(dropped)} candidate(s) whose key set "
            f"differs from the majority ({len(kept)} share it); likely "
            "incomplete caches:",
            file=sys.stderr,
        )
        for sd, sk in dropped[:20]:
            print(f"  - {sd.name} ({sk.size} keys)", file=sys.stderr)
        if len(dropped) > 20:
            print(f"  ... ({len(dropped) - 20} more)", file=sys.stderr)

    cache_ids = [rec.get("cache_id", sd.name) for (sd, rec, _, _) in kept]
    records = [rec for (_, rec, _, _) in kept]
    E = np.stack([er for (_, _, _, er) in kept], axis=0).astype(
        np.float32, copy=False
    )
    return E, cache_ids, records


def _aggregate(values: np.ndarray, aggregator: str) -> float:
    """Apply the configured aggregator to a 1-D EPE vector.

    Args:
        values: Per-image EPE values.
        aggregator: ``"mean"`` or ``"median"``.

    Returns:
        The aggregated scalar cost.

    Raises:
        ValueError: If ``aggregator`` is not recognized.
    """
    if aggregator == "mean":
        return float(np.mean(values))
    if aggregator == "median":
        return float(np.median(values))
    raise ValueError(f"unknown aggregator {aggregator!r}")


def _greedy_select(
    E: np.ndarray,
    K: int,
    aggregator: str,
    verbose: bool,
) -> list[int]:
    """Greedy subset selection minimizing the aggregated per-image best EPE.

    Brute-force scan of all remaining candidates at every step. The
    earlier draft used a Minoux-style lazy heap; that variant requires
    a max-heap of marginal gains, not a min-heap of absolute costs, so
    it was incorrect and has been removed. At the scales we care about
    (N x M on the order of 10^6), the O(K * N * M) scan finishes in
    well under a second and is obviously correct, regardless of the
    aggregator's submodularity.

    Args:
        E: Algorithm x image EPE matrix of shape ``(N, M)``.
        K: Desired subset size.
        aggregator: ``"mean"`` or ``"median"``.
        verbose: Whether to print per-step progress.

    Returns:
        A list of selected row indices in selection order.

    Raises:
        ValueError: If ``K`` exceeds ``N`` or is non-positive.
        RuntimeError: If the greedy state becomes inconsistent (should
            not happen in practice).
    """
    n_algos, n_images = E.shape
    if K <= 0 or K > n_algos:
        raise ValueError(
            f"K={K} out of range for N={n_algos} surviving algorithms",
        )

    current_min = np.full(n_images, np.inf, dtype=np.float32)
    selected: list[int] = []
    remaining = set(range(n_algos))

    while len(selected) < K:
        best_a = -1
        best_cost = math.inf
        for a in sorted(remaining):
            cand = np.minimum(current_min, E[a])
            cost = _aggregate(cand, aggregator)
            if cost < best_cost:
                best_cost = cost
                best_a = a
        if best_a < 0:
            raise RuntimeError("greedy scan failed to pick an element")
        current_min = np.minimum(current_min, E[best_a])
        selected.append(best_a)
        remaining.discard(best_a)
        if verbose:
            print(
                f"  step {len(selected)}: picked {best_a} cost={best_cost:.6f}",
                flush=True,
            )

    return selected


def _build_milp_constraints(
    n_algos: int,
    n_images: int,
    K: int,
) -> tuple[LinearConstraint, ...]:
    """Build the LP/MILP constraints for the exact mean-aggregator solver.

    Variables are laid out as ``[x_0, ..., x_{N-1}, y_{0,0}, y_{0,1},
    ..., y_{N-1, M-1}]`` (algorithm indicators followed by per-pair
    assignments, row-major over algorithms).

    The three constraint blocks are:

    * one equality fixing ``sum_a x_a == K``,
    * ``M`` equalities ``sum_a y_{a,i} == 1`` (each image gets exactly
      one assigned algorithm),
    * ``N*M`` inequalities ``y_{a,i} - x_a <= 0`` linking assignment
      to selection.

    Args:
        n_algos: Number of surviving algorithms ``N``.
        n_images: Number of images ``M``.
        K: Subset size.

    Returns:
        Three :class:`scipy.optimize.LinearConstraint` objects in the
        order described above.
    """
    n_y = n_algos * n_images
    n_vars = n_algos + n_y

    # Block 1: sum x_a == K.
    row1 = csr_matrix(
        (
            np.ones(n_algos, dtype=np.float64),
            (np.zeros(n_algos, dtype=np.int64), np.arange(n_algos)),
        ),
        shape=(1, n_vars),
    )
    c1 = LinearConstraint(row1, lb=float(K), ub=float(K))

    # Block 2: for each image i, sum_a y_{a,i} == 1.
    rows: list[int] = []
    cols: list[int] = []
    for a in range(n_algos):
        base = n_algos + a * n_images
        rows.extend(range(n_images))
        cols.extend(range(base, base + n_images))
    data = np.ones(len(rows), dtype=np.float64)
    img_mat = csr_matrix(
        (data, (np.asarray(rows), np.asarray(cols))),
        shape=(n_images, n_vars),
    )
    c2 = LinearConstraint(img_mat, lb=1.0, ub=1.0)

    # Block 3: y_{a,i} - x_a <= 0 for every (a, i).
    pair_rows = np.arange(n_y, dtype=np.int64)
    pair_cols_y = n_algos + pair_rows
    pair_cols_x = np.repeat(np.arange(n_algos, dtype=np.int64), n_images)
    rows_all = np.concatenate([pair_rows, pair_rows])
    cols_all = np.concatenate([pair_cols_y, pair_cols_x])
    data_all = np.concatenate(
        [np.ones(n_y, dtype=np.float64), -np.ones(n_y, dtype=np.float64)],
    )
    pair_mat = csr_matrix(
        (data_all, (rows_all, cols_all)),
        shape=(n_y, n_vars),
    )
    c3 = LinearConstraint(pair_mat, lb=-math.inf, ub=0.0)
    return c1, c2, c3


def _exact_milp(
    E: np.ndarray,
    K: int,
    timeout_s: float,
    verbose: bool,
) -> tuple[list[int] | None, str, float | None]:
    """Solve the exact mean-aggregator subset problem via HiGHS MILP.

    Args:
        E: Algorithm x image EPE matrix (``float32`` or ``float64``).
        K: Subset size.
        timeout_s: Wall-time cap passed to HiGHS.
        verbose: Whether to print solver diagnostics.

    Returns:
        A tuple ``(selected, status_message, sum_objective)``. The
        selection is ``None`` when HiGHS failed to return an integer
        solution; the sum objective is the LP objective ``sum_{a,i}
        e_{a,i} y_{a,i}`` (divide by ``M`` for mean EPE).
    """
    n_algos, n_images = E.shape
    n_y = n_algos * n_images

    c = np.concatenate(
        [
            np.zeros(n_algos, dtype=np.float64),
            E.astype(np.float64, copy=False).reshape(-1),
        ],
    )

    bounds = Bounds(lb=0.0, ub=1.0)
    integrality = np.concatenate(
        [
            np.ones(n_algos, dtype=np.int64),
            np.zeros(n_y, dtype=np.int64),
        ],
    )

    constraints = _build_milp_constraints(n_algos, n_images, K)

    result = milp(
        c=c,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
        options={"time_limit": float(timeout_s), "disp": bool(verbose)},
    )

    status_message = f"status={result.status} message={result.message!r}"
    if result.x is None:
        return None, status_message, None

    x_part = np.asarray(result.x[:n_algos])
    selected_arr = np.flatnonzero(x_part > 0.5)
    if selected_arr.size != K:
        return None, status_message, float(result.fun)
    return selected_arr.tolist(), status_message, float(result.fun)


def _summarize_selection(
    selected: list[int],
    E: np.ndarray,
    cache_ids: list[str],
    configs: list[dict[str, Any]],
    times: np.ndarray,
) -> dict[str, Any]:
    """Build a JSON-serialisable description of a chosen subset.

    Args:
        selected: Row indices into ``E`` of the selected algorithms.
        E: Full EPE matrix of the surviving algorithms.
        cache_ids: Cache ids aligned with ``E`` rows.
        configs: Full timing-record payloads aligned with ``E`` rows.
        times: Per-algorithm time stat used for filtering.

    Returns:
        A dictionary with cost statistics, per-image winners, and a
        per-selection-entry breakdown.
    """
    sub_E = E[selected]
    per_image_min = sub_E.min(axis=0)
    per_image_argmin_local = sub_E.argmin(axis=0)
    per_image_winner_global = np.asarray(selected)[per_image_argmin_local]

    win_counts = np.bincount(per_image_argmin_local, minlength=len(selected))

    entries: list[dict[str, Any]] = []
    for local_idx, a in enumerate(selected):
        entries.append(
            {
                "cache_id": cache_ids[a],
                "config": configs[a].get("config", {}),
                "time_ms": float(times[a]),
                "win_count": int(win_counts[local_idx]),
                "mean_epe_solo": float(np.mean(E[a])),
                "median_epe_solo": float(np.median(E[a])),
            },
        )

    return {
        "selected_indices": [int(a) for a in selected],
        "selected": entries,
        "cost_mean": float(np.mean(per_image_min)),
        "cost_median": float(np.median(per_image_min)),
        "per_image_winner": [
            cache_ids[int(a)] for a in per_image_winner_global
        ],
        "per_image_min_epe": [float(v) for v in per_image_min],
    }


def _print_selection(label: str, summary: dict[str, Any]) -> None:
    """Print one solver's selection block.

    Args:
        label: Header label (e.g. ``"greedy"``).
        summary: Output of :func:`_summarize_selection`.
    """
    print(
        f"[{label}] cost_mean={summary['cost_mean']:.6f} "
        f"cost_median={summary['cost_median']:.6f}"
    )
    for rank, entry in enumerate(summary["selected"], start=1):
        cfg = entry["config"]
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(
            f"  #{rank} {entry['cache_id']} "
            f"t={entry['time_ms']:.3f}ms "
            f"wins={entry['win_count']} "
            f"mean_solo={entry['mean_epe_solo']:.4f}  [{cfg_str}]",
        )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Optional explicit argument list (for testing).

    Returns:
        Process exit code (0 on success).

    Raises:
        RuntimeError: If too few algorithms survive filtering or if
            non-finite EPE values remain after the filter step.
    """
    args = _parse_args(argv)

    if args.verbose:
        print(f"loading cache from {args.cache_root}", flush=True)
    E_all, cache_ids_all, records_all = _load_cache(
        args.cache_root,
        args.metric,
        args.verbose,
    )

    # Pick the requested time stat per candidate.
    times_all = np.array(
        [float(t.get(args.time_stat, float("nan"))) for t in records_all],
        dtype=np.float64,
    )

    n0 = E_all.shape[0]
    finite_mask = np.isfinite(E_all).all(axis=1)
    n_dropped_nonfinite = int((~finite_mask).sum())

    # The time bound is opt-in: with an infinite limit, candidates without
    # a timing.json (NaN stat) are kept. A finite limit drops them.
    if math.isfinite(args.time_limit):
        time_mask = times_all <= args.time_limit
    else:
        time_mask = np.ones_like(finite_mask)
    keep_mask = finite_mask & time_mask
    n_dropped_time = int(((~time_mask) & finite_mask).sum())

    print(f"loaded {n0} candidates from {args.cache_root}")
    print(f"  dropped {n_dropped_nonfinite} with non-finite {args.metric}")
    print(
        f"  dropped {n_dropped_time} above --time-limit={args.time_limit} "
        f"on {args.time_stat}",
    )
    n_kept = int(keep_mask.sum())
    print(f"  kept {n_kept} surviving candidates")

    if n_kept < args.K:
        raise RuntimeError(
            f"only {n_kept} candidates survived, cannot pick K={args.K}",
        )

    kept_idx = np.flatnonzero(keep_mask)
    E = E_all[kept_idx]
    times = times_all[kept_idx]
    cache_ids = [cache_ids_all[i] for i in kept_idx]
    configs = [records_all[i] for i in kept_idx]

    if not np.isfinite(E).all():
        raise RuntimeError(
            f"non-finite {args.metric} survived filtering; "
            "this indicates a bug",
        )

    print(
        f"K={args.K} metric={args.metric} aggregator={args.aggregator} "
        f"time_stat={args.time_stat} time_limit={args.time_limit}",
    )

    print("running greedy...")
    greedy_local = _greedy_select(E, args.K, args.aggregator, args.verbose)
    greedy_summary = _summarize_selection(
        greedy_local,
        E,
        cache_ids,
        configs,
        times,
    )
    _print_selection("greedy", greedy_summary)

    exact_summary: dict[str, Any] | None = None
    exact_status: str | None = None
    if args.exact:
        if args.aggregator != "mean":
            print(
                "note: --exact is only supported for --aggregator mean in v1; "
                "skipping MILP and using greedy result.",
            )
        else:
            print(
                f"running exact MILP (HiGHS, timeout={args.milp_timeout}s)...",
            )
            exact_local, status_msg, sum_obj = _exact_milp(
                E,
                args.K,
                args.milp_timeout,
                args.verbose,
            )
            exact_status = status_msg
            print(f"  HiGHS {status_msg}")
            if exact_local is None:
                print(
                    "WARNING: MILP did not return an integer solution; "
                    "falling back to greedy.",
                )
            else:
                exact_summary = _summarize_selection(
                    exact_local,
                    E,
                    cache_ids,
                    configs,
                    times,
                )
                _print_selection("exact", exact_summary)
                if sum_obj is not None:
                    obj_mean = sum_obj / float(E.shape[1])
                    print(
                        f"  MILP sum-objective={sum_obj:.6f}  "
                        f"(mean={obj_mean:.6f})"
                    )

    if exact_summary is not None:
        agg_key = "cost_mean" if args.aggregator == "mean" else "cost_median"
        g_cost = greedy_summary[agg_key]
        e_cost = exact_summary[agg_key]
        if e_cost > 0:
            gap = (g_cost - e_cost) / e_cost * 100.0
        else:
            gap = float("nan")
        print(
            f"gap[{args.aggregator}] greedy={g_cost:.6f} "
            f"exact={e_cost:.6f}  -> {gap:.4f}%",
        )

    if args.out is not None:
        payload: dict[str, Any] = {
            "solver": "exact+greedy" if exact_summary is not None else "greedy",
            "K": args.K,
            "time_limit": args.time_limit,
            "time_stat": args.time_stat,
            "aggregator": args.aggregator,
            "kept_n": n_kept,
            "dropped_nonfinite": n_dropped_nonfinite,
            "dropped_time": n_dropped_time,
            "greedy": greedy_summary,
        }
        if exact_summary is not None:
            payload["exact"] = exact_summary
            payload["exact_status"] = exact_status
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")

    if args.export_models is not None:
        best = exact_summary if exact_summary is not None else greedy_summary
        written, skipped = _export_models(
            best,
            args.export_models,
            args.export_estimator,
            args.export_estimate_type,
        )
        msg = f"exported {written} model config(s) to {args.export_models}"
        if skipped:
            msg += f" ({skipped} skipped: no stored config)"
        print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
