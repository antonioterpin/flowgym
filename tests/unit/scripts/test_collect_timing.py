"""Tests for the generic timing-collection sweep wrapper.

These exercise the pure, JAX-free parts (record assembly, cache-id-base
resolution, arg parsing). The timing pass itself needs a built estimator and
a device, so it is not unit-tested here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "collect_timing.py"

_spec = importlib.util.spec_from_file_location("collect_timing", _SCRIPT)
assert _spec is not None and _spec.loader is not None
collect_timing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_timing)


def test_build_timing_record_has_select_ensemble_fields() -> None:
    """Record carries the four fields select_ensemble --time-stat keys on."""
    rec = collect_timing.build_timing_record(
        [1.0, 2.0, 3.0],
        compile_ms=50.0,
        warmup=5,
        iters=100,
        repeat=3,
        batch_size=1,
    )
    for key in ("mean_ms", "median_ms", "min_ms", "max_ms"):
        assert key in rec
    assert rec["min_ms"] == 1.0
    assert rec["max_ms"] == 3.0
    assert rec["mean_ms"] == pytest.approx(2.0)
    assert rec["median_ms"] == 2.0
    assert rec["n_iters"] == 100
    assert rec["n_repeat"] == 3
    assert rec["batch_size"] == 1


def test_build_timing_record_optional_metadata() -> None:
    """Device/config/cache_id attach when provided; std=0 for one sample."""
    rec = collect_timing.build_timing_record(
        [1.5],
        compile_ms=0.0,
        warmup=1,
        iters=1,
        repeat=1,
        batch_size=2,
        device="cuda:0",
        config={"window_size": 16},
        cache_id="ds_cabc1234",
    )
    assert rec["device"] == "cuda:0"
    assert rec["config"] == {"window_size": 16}
    assert rec["cache_id"] == "ds_cabc1234"
    assert rec["batch_size"] == 2
    assert rec["std_ms"] == 0.0


def test_resolve_cache_id_base_prefers_override() -> None:
    """An explicit base wins over the dataset's caching.cache_id."""
    cfg = {"caching": {"cache_id": "from-dataset"}}
    assert collect_timing.resolve_cache_id_base(cfg, "override") == "override"
    assert collect_timing.resolve_cache_id_base(cfg, None) == "from-dataset"


def test_resolve_cache_id_base_missing_raises() -> None:
    """No override and no caching block is an error (can't name the cache)."""
    with pytest.raises(ValueError):
        collect_timing.resolve_cache_id_base({}, None)


def test_parse_args_requires_a_model_source() -> None:
    """Either --models or --estimators-list must be given."""
    with pytest.raises(SystemExit):
        collect_timing.parse_args(["--dataset", "d.yaml", "--cache-root", "c"])


def test_parse_args_defaults() -> None:
    """Basic parse populates timing defaults."""
    ns = collect_timing.parse_args(
        ["--models", "m.yaml", "--dataset", "d.yaml", "--cache-root", "c"]
    )
    assert ns.models == [Path("m.yaml")]
    assert ns.dataset == Path("d.yaml")
    assert ns.batch_size == 1
    assert ns.iters > 0 and ns.repeat > 0
