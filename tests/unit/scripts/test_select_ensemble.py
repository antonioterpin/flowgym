"""Tests for the generic ensemble selector script."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "select_ensemble.py"

_spec = importlib.util.spec_from_file_location("select_ensemble", _SCRIPT)
assert _spec is not None and _spec.loader is not None
select_ensemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_ensemble)


def _write_candidate(
    cache_root: Path,
    cache_id: str,
    keys: list[int],
    values: list[float],
    metric: str = "epe",
    timing: dict | None = None,
) -> None:
    """Write a single candidate's cache directory under ``cache_root``.

    Args:
        cache_root: Sweep cache root.
        cache_id: Candidate subdirectory name and recorded cache id.
        keys: Per-image integer keys.
        values: Per-image scalar errors aligned with ``keys``.
        metric: Parquet column name holding the errors.
        timing: Optional timing payload; when given a ``timing.json`` is
            written alongside the data shards.
    """
    subdir = cache_root / cache_id
    data_dir = subdir / "data"
    data_dir.mkdir(parents=True)
    (subdir / "meta.json").write_text(
        json.dumps({"cache_id": cache_id, "version": "1.0"}),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "key": pa.array(np.asarray(keys, dtype=np.uint64)),
            metric: pa.array(np.asarray(values, dtype=np.float32)),
        }
    )
    pq.write_table(table, data_dir / "part-0.parquet")
    if timing is not None:
        (subdir / "timing.json").write_text(
            json.dumps(timing), encoding="utf-8"
        )


@pytest.fixture
def tiny_cache(tmp_path: Path) -> Path:
    """Build a 3-candidate, 4-image cache with a clear complementary pair.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The cache root path.
    """
    keys = [10, 20, 30, 40]
    _write_candidate(tmp_path, "cand_a", keys, [1.0, 9.0, 9.0, 9.0])
    _write_candidate(tmp_path, "cand_b", keys, [9.0, 1.0, 9.0, 9.0])
    _write_candidate(tmp_path, "cand_c", keys, [5.0, 5.0, 1.0, 1.0])
    return tmp_path


def test_greedy_selects_complementary_subset(
    tiny_cache: Path, tmp_path: Path
) -> None:
    """Greedy picks the pair minimizing mean per-image best error."""
    out = tmp_path / "result.json"
    rc = select_ensemble.main(
        ["--cache-root", str(tiny_cache), "--K", "2", "--out", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    chosen = {e["cache_id"] for e in payload["greedy"]["selected"]}
    # cand_c covers images 3,4 (cost 1); cand_a covers image 1 (cost 1).
    # {a,c} -> mean([1,5,1,1]) = 2.0 beats {a,b} (5.0) and {b,c} (2.0 tie,
    # but greedy seeds with the best solo = cand_c then adds cand_a).
    assert chosen == {"cand_a", "cand_c"}
    assert payload["greedy"]["cost_mean"] == pytest.approx(2.0)


def test_exact_matches_known_optimum(tiny_cache: Path, tmp_path: Path) -> None:
    """--exact returns the known optimum; greedy is already optimal (gap 0)."""
    out = tmp_path / "result.json"
    rc = select_ensemble.main(
        [
            "--cache-root",
            str(tiny_cache),
            "--K",
            "2",
            "--exact",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "exact" in payload
    # {a,c} and {b,c} both reach the optimum mean 2.0.
    assert payload["exact"]["cost_mean"] == pytest.approx(2.0)
    assert {e["cache_id"] for e in payload["exact"]["selected"]} <= {
        "cand_a",
        "cand_b",
        "cand_c",
    }
    # Greedy already finds the optimum here, so the greedy/exact gap is 0.
    assert payload["greedy"]["cost_mean"] == pytest.approx(
        payload["exact"]["cost_mean"]
    )


@pytest.fixture
def greedy_suboptimal_cache(tmp_path: Path) -> Path:
    """Build a cache where greedy is strictly worse than the exact optimum.

    Solo means: cand_a=2.5 (best, greedy's seed), cand_b=cand_c=3.5. After
    seeding cand_a, greedy adds cand_b for mean 1.75, but the true optimum
    {cand_b, cand_c} has mean 1.0 -- so the MILP must refine greedy.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The cache root path.
    """
    keys = [1, 2, 3, 4]
    _write_candidate(tmp_path, "cand_a", keys, [1.0, 1.0, 4.0, 4.0])
    _write_candidate(tmp_path, "cand_b", keys, [1.0, 6.0, 1.0, 6.0])
    _write_candidate(tmp_path, "cand_c", keys, [6.0, 1.0, 6.0, 1.0])
    return tmp_path


def test_exact_milp_refines_greedy(
    greedy_suboptimal_cache: Path, tmp_path: Path
) -> None:
    """Greedy init + MILP refinement: the exact pass strictly improves."""
    out = tmp_path / "result.json"
    rc = select_ensemble.main(
        [
            "--cache-root",
            str(greedy_suboptimal_cache),
            "--K",
            "2",
            "--exact",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    greedy_cost = payload["greedy"]["cost_mean"]
    exact_cost = payload["exact"]["cost_mean"]
    assert greedy_cost == pytest.approx(1.75)  # greedy seeds the trap cand_a
    assert exact_cost == pytest.approx(1.0)  # true optimum {cand_b, cand_c}
    assert exact_cost < greedy_cost  # MILP refines past greedy
    assert {e["cache_id"] for e in payload["exact"]["selected"]} == {
        "cand_b",
        "cand_c",
    }


def test_custom_metric_column(tmp_path: Path) -> None:
    """A non-default metric column is honoured via --metric."""
    keys = [1, 2, 3]
    _write_candidate(tmp_path, "x", keys, [0.0, 5.0, 5.0], metric="score")
    _write_candidate(tmp_path, "y", keys, [5.0, 0.0, 0.0], metric="score")
    rc = select_ensemble.main(
        ["--cache-root", str(tmp_path), "--K", "1", "--metric", "score"]
    )
    assert rc == 0


def test_works_without_timing_json(tiny_cache: Path) -> None:
    """Candidates without timing.json are usable when no time bound is set."""
    E, cache_ids, records = select_ensemble._load_cache(
        tiny_cache, metric="epe", verbose=False
    )
    assert E.shape == (3, 4)
    assert set(cache_ids) == {"cand_a", "cand_b", "cand_c"}
    assert all(r.get("mean_ms") is None for r in records)


def test_partial_cache_is_dropped_not_fatal(tiny_cache: Path) -> None:
    """A cache with a minority (incomplete) key set is dropped, not fatal."""
    # tiny_cache has 3 candidates over keys [10,20,30,40]; add a partial one.
    _write_candidate(tiny_cache, "cand_partial", [10, 20], [0.0, 0.0])
    E, cache_ids, _ = select_ensemble._load_cache(
        tiny_cache, metric="epe", verbose=False
    )
    assert E.shape == (3, 4)  # the 4-key majority survives
    assert "cand_partial" not in cache_ids


def test_export_models_writes_collectable_estimators_list(
    tmp_path: Path,
) -> None:
    """Selected configs are exported as a collect_cache estimators_list."""
    summary = {
        "selected": [
            {"cache_id": "a", "config": {"patch_size": 7, "preset": 1}},
            {"cache_id": "b", "config": {}},  # no stored config -> skipped
        ]
    }
    out = tmp_path / "chosen_models.yaml"
    written, skipped = select_ensemble._export_models(
        summary, out, "dis_jax", "flow"
    )
    assert (written, skipped) == (1, 1)
    data = yaml.safe_load(out.read_text())
    assert data["estimators"] == [
        {
            "name": "a",
            "estimator": "dis_jax",
            "estimate_type": "flow",
            "config": {"patch_size": 7, "preset": 1},
        }
    ]


def test_export_preserves_config_verbatim(tmp_path: Path) -> None:
    """Configs are exported untouched (the selector is algorithm-agnostic).

    A serialized preset name (e.g. FAST) is emitted as-is; re-loading it is
    the estimator's job (the DIS constructor accepts the enum name).
    """
    summary = {"selected": [{"cache_id": "a", "config": {"preset": "FAST"}}]}
    out = tmp_path / "m.yaml"
    select_ensemble._export_models(summary, out, "dis_jax", "flow")
    data = yaml.safe_load(out.read_text())
    assert data["estimators"][0]["config"]["preset"] == "FAST"


def test_export_models_end_to_end(tmp_path: Path) -> None:
    """--export-models writes the chosen subset using timing.json configs."""
    keys = [1, 2, 3]
    _write_candidate(
        tmp_path,
        "x",
        keys,
        [0.0, 5.0, 5.0],
        timing={"config": {"patch_size": 7}},
    )
    _write_candidate(
        tmp_path,
        "y",
        keys,
        [5.0, 0.0, 0.0],
        timing={"config": {"patch_size": 9}},
    )
    out = tmp_path / "models.yaml"
    rc = select_ensemble.main(
        [
            "--cache-root",
            str(tmp_path),
            "--K",
            "2",
            "--export-models",
            str(out),
        ]
    )
    assert rc == 0
    data = yaml.safe_load(out.read_text())
    sizes = {e["config"]["patch_size"] for e in data["estimators"]}
    assert sizes == {7, 9}
    assert all(e["estimator"] == "dis_jax" for e in data["estimators"])


def test_finite_time_limit_without_timing_drops_all(tiny_cache: Path) -> None:
    """A finite time bound drops candidates lacking a timing stat."""
    with pytest.raises(RuntimeError, match="survived"):
        select_ensemble.main(
            ["--cache-root", str(tiny_cache), "--K", "1", "--time-limit", "1.0"]
        )


# ---------------------------------------------------------------------------
# Tests for --keys-from and --cache-id-prefix
# ---------------------------------------------------------------------------


def _expected_key(filename: str) -> int:
    """Expected uint64 key for a filename (matches compute_batch_keys)."""
    h = hashlib.md5(filename.encode()).hexdigest()[:16]
    return int(np.uint64(int(h, 16)))


def test_keys_from_dir_matches_compute_batch_keys_formula(
    tmp_path: Path,
) -> None:
    """_keys_from_dir returns the file-based-formula keys."""
    names = ["flow_001.mat", "flow_002.mat", "flow_003.mat"]
    for name in names:
        (tmp_path / name).write_bytes(b"")

    result = select_ensemble._keys_from_dir(tmp_path)
    expected = {_expected_key(n) for n in names}
    assert result == expected


def test_keys_from_dir_recursive(tmp_path: Path) -> None:
    """_keys_from_dir recurses into subdirectories."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "a.mat").write_bytes(b"")
    (sub / "b.mat").write_bytes(b"")

    result = select_ensemble._keys_from_dir(tmp_path)
    assert result == {_expected_key("a.mat"), _expected_key("b.mat")}


def test_restrict_candidate_keeps_requested_subset() -> None:
    """_restrict_candidate_to_keys filters to the requested key subset."""
    keys = np.array([10, 20, 30, 40], dtype=np.uint64)
    errors = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    requested = {10, 30}

    result = select_ensemble._restrict_candidate_to_keys(
        keys, errors, requested
    )
    assert result is not None
    rk, re = result
    assert set(rk.tolist()) == requested
    assert re.tolist() == pytest.approx([1.0, 3.0])


def test_restrict_candidate_drops_when_key_missing() -> None:
    """_restrict_candidate_to_keys returns None on a missing key."""
    keys = np.array([10, 20], dtype=np.uint64)
    errors = np.array([1.0, 2.0], dtype=np.float32)
    requested = {10, 20, 99}  # 99 not in candidate

    result = select_ensemble._restrict_candidate_to_keys(
        keys, errors, requested
    )
    assert result is None


def test_restrict_candidate_dedups_duplicate_keys() -> None:
    """Duplicate keys (e.g. a twice-collected cache) yield one row per key.

    Without de-duplication the restricted array length would vary across
    candidates and break the stack into the algorithm x image matrix.
    """
    keys = np.array([10, 10, 20, 30, 30], dtype=np.uint64)  # sorted, dups
    errors = np.array([1.0, 9.0, 2.0, 3.0, 9.0], dtype=np.float32)
    requested = {10, 20, 30}

    result = select_ensemble._restrict_candidate_to_keys(
        keys, errors, requested
    )
    assert result is not None
    rk, re = result
    assert rk.tolist() == [10, 20, 30]  # exactly one row per requested key
    assert re.tolist() == pytest.approx([1.0, 2.0, 3.0])  # first occurrence


def test_keys_from_restricts_selection(tmp_path: Path) -> None:
    """--keys-from scopes selection to the named split's keys only."""
    # Build a 3-candidate cache with keys [10, 20, 30, 40].
    all_keys = [10, 20, 30, 40]
    _write_candidate(
        tmp_path / "cache", "cand_a", all_keys, [1.0, 9.0, 9.0, 9.0]
    )
    _write_candidate(
        tmp_path / "cache", "cand_b", all_keys, [9.0, 1.0, 9.0, 9.0]
    )
    _write_candidate(
        tmp_path / "cache", "cand_c", all_keys, [5.0, 5.0, 1.0, 1.0]
    )

    # Split dir: two .mat files mapping to cache keys 10 and 20.
    # We need filenames whose md5 hashes produce those exact uint64 values, so
    # instead we use _keys_from_dir indirectly: write the actual split dir and
    # compare what _load_cache produces with what _keys_from_dir computes.
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    # Use filename strings that we can verify independently.
    mat_a = "img_001.mat"
    mat_b = "img_002.mat"
    (split_dir / mat_a).write_bytes(b"")
    (split_dir / mat_b).write_bytes(b"")

    split_keys = select_ensemble._keys_from_dir(split_dir)
    assert len(split_keys) == 2

    # Build a cache where only split_keys are valid image keys.
    sk_sorted = sorted(split_keys)
    cache2 = tmp_path / "cache2"
    _write_candidate(cache2, "cand_x", sk_sorted, [1.0, 2.0])
    _write_candidate(cache2, "cand_y", sk_sorted, [2.0, 1.0])

    E, cache_ids, _ = select_ensemble._load_cache(
        cache2, "epe", verbose=False, keys_from=split_dir
    )
    # Both candidates have all requested keys; E should have shape (2, 2).
    assert E.shape == (2, 2)
    assert set(cache_ids) == {"cand_x", "cand_y"}


def test_keys_from_drops_candidate_missing_keys(tmp_path: Path) -> None:
    """A candidate missing some requested keys is dropped with a warning."""
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "a.mat").write_bytes(b"")
    (split_dir / "b.mat").write_bytes(b"")

    split_keys = sorted(select_ensemble._keys_from_dir(split_dir))
    assert len(split_keys) == 2

    cache = tmp_path / "cache"
    # cand_full has both keys; cand_partial has only one.
    _write_candidate(cache, "cand_full", split_keys, [1.0, 2.0])
    _write_candidate(cache, "cand_partial", [split_keys[0]], [1.0])

    E, cache_ids, _ = select_ensemble._load_cache(
        cache, "epe", verbose=False, keys_from=split_dir
    )
    assert "cand_partial" not in cache_ids
    assert "cand_full" in cache_ids
    assert E.shape == (1, 2)


def test_cache_id_prefix_filters_subdirs(tmp_path: Path) -> None:
    """--cache-id-prefix excludes subdirs not starting with the prefix."""
    keys = [1, 2, 3]
    _write_candidate(tmp_path, "dis_alpha", keys, [1.0, 2.0, 3.0])
    _write_candidate(tmp_path, "dis_beta", keys, [4.0, 5.0, 6.0])
    _write_candidate(tmp_path, "raft_gamma", keys, [7.0, 8.0, 9.0])

    E, cache_ids, _ = select_ensemble._load_cache(
        tmp_path, "epe", verbose=False, cache_id_prefix="dis_"
    )
    assert set(cache_ids) == {"dis_alpha", "dis_beta"}
    assert "raft_gamma" not in cache_ids
    assert E.shape == (2, 3)
