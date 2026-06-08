"""Tests for the generic ensemble selector script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def test_finite_time_limit_without_timing_drops_all(tiny_cache: Path) -> None:
    """A finite time bound drops candidates lacking a timing stat."""
    with pytest.raises(RuntimeError, match="survived"):
        select_ensemble.main(
            ["--cache-root", str(tiny_cache), "--K", "1", "--time-limit", "1.0"]
        )
