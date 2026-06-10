"""Tests for the generic cache-collection sweep wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "collect_cache.py"

_spec = importlib.util.spec_from_file_location("collect_cache", _SCRIPT)
assert _spec is not None and _spec.loader is not None
collect_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_cache)


def test_build_eval_command_basic() -> None:
    """The command targets main.py eval with the shared cache root."""
    cmd = collect_cache.build_eval_command(
        ("uv", "run", "python"),
        Path("m.yaml"),
        Path("d.yaml"),
        Path("caches"),
    )
    assert cmd == [
        "uv",
        "run",
        "python",
        "src/main.py",
        "--mode",
        "eval",
        "--estimator",
        "m.yaml",
        "--dataset",
        "d.yaml",
        "--cache-root",
        "caches",
    ]


def test_build_eval_command_with_cache_id() -> None:
    """A base cache id is forwarded to main.py."""
    cmd = collect_cache.build_eval_command(
        (".venv/bin/python",),
        Path("m.yaml"),
        Path("d.yaml"),
        Path("c"),
        cache_id="base",
    )
    assert cmd[0] == ".venv/bin/python"
    assert cmd[cmd.index("--cache-id") + 1] == "base"


def test_main_succeeds_with_noop_runner(tmp_path: Path) -> None:
    """Loop returns 0 when every per-model run exits cleanly."""
    m1 = tmp_path / "m1.yaml"
    m1.write_text("x", encoding="utf-8")
    m2 = tmp_path / "m2.yaml"
    m2.write_text("x", encoding="utf-8")
    d = tmp_path / "d.yaml"
    d.write_text("y", encoding="utf-8")
    rc = collect_cache.main(
        [
            "--models",
            str(m1),
            str(m2),
            "--dataset",
            str(d),
            "--cache-root",
            str(tmp_path / "c"),
            "--runner",
            "true",
        ]
    )
    assert rc == 0


def test_main_reports_failure_with_failing_runner(tmp_path: Path) -> None:
    """A non-zero per-model exit propagates to a non-zero return code."""
    m = tmp_path / "m.yaml"
    m.write_text("x", encoding="utf-8")
    d = tmp_path / "d.yaml"
    d.write_text("y", encoding="utf-8")
    rc = collect_cache.main(
        [
            "--models",
            str(m),
            "--dataset",
            str(d),
            "--cache-root",
            str(tmp_path / "c"),
            "--runner",
            "false",
        ]
    )
    assert rc == 1


def _write_estimators_list(path: Path) -> None:
    """Write a minimal 2-entry estimators_list YAML."""
    path.write_text(
        yaml.dump(
            {
                "estimators": [
                    {
                        "name": "DIS_a",
                        "estimator": "dis_jax",
                        "estimate_type": "flow",
                        "config": {"preset": 1, "patch_size": 7},
                    },
                    {
                        "name": "DIS_b",
                        "estimator": "dis_jax",
                        "estimate_type": "flow",
                        "config": {"preset": 1, "patch_size": 9},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_estimators_list_returns_entries(tmp_path: Path) -> None:
    """The estimators_list loader returns the entry dicts."""
    lst = tmp_path / "dis_models.yaml"
    _write_estimators_list(lst)
    entries = collect_cache.load_estimators_list(lst)
    assert [e["name"] for e in entries] == ["DIS_a", "DIS_b"]


def test_load_estimators_list_rejects_empty(tmp_path: Path) -> None:
    """A file without an `estimators:` list is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="estimators"):
        collect_cache.load_estimators_list(bad)


def test_materialize_drops_name_and_writes_model_yaml(tmp_path: Path) -> None:
    """Each entry becomes a standalone model YAML without the name key."""
    lst = tmp_path / "dis_models.yaml"
    _write_estimators_list(lst)
    entries = collect_cache.load_estimators_list(lst)
    work = tmp_path / "work"
    work.mkdir()
    paths = collect_cache.materialize_model_configs(entries, work)
    assert [p.name for p in paths] == ["000_DIS_a.yaml", "001_DIS_b.yaml"]
    model = yaml.safe_load(paths[0].read_text())
    assert model == {
        "estimator": "dis_jax",
        "estimate_type": "flow",
        "config": {"preset": 1, "patch_size": 7},
    }


def test_materialize_dedupes_duplicate_names(tmp_path: Path) -> None:
    """Entries sharing a name get distinct files (idx prefix), not collision."""
    entries = [
        {"name": "dup", "estimator": "dis_jax", "config": {"patch_size": 7}},
        {"name": "dup", "estimator": "dis_jax", "config": {"patch_size": 9}},
    ]
    work = tmp_path / "work"
    work.mkdir()
    paths = collect_cache.materialize_model_configs(entries, work)
    # Both entries survive: distinct paths, each carrying its own config.
    assert len(set(paths)) == 2
    sizes = {
        yaml.safe_load(p.read_text())["config"]["patch_size"] for p in paths
    }
    assert sizes == {7, 9}


def test_main_fans_out_over_estimators_list(tmp_path: Path) -> None:
    """--estimators-list runs one eval per sub-estimator."""
    lst = tmp_path / "dis_models.yaml"
    _write_estimators_list(lst)
    d = tmp_path / "d.yaml"
    d.write_text("y", encoding="utf-8")
    rc = collect_cache.main(
        [
            "--estimators-list",
            str(lst),
            "--dataset",
            str(d),
            "--cache-root",
            str(tmp_path / "c"),
            "--runner",
            "true",
        ]
    )
    assert rc == 0


def test_requires_a_config_source(tmp_path: Path) -> None:
    """Neither --models nor --estimators-list is an error."""
    d = tmp_path / "d.yaml"
    d.write_text("y", encoding="utf-8")
    with pytest.raises(SystemExit):
        collect_cache.main(
            ["--dataset", str(d), "--cache-root", str(tmp_path / "c")]
        )


def test_limit_truncates_model_list(tmp_path: Path) -> None:
    """--limit processes only the first N models."""
    models = []
    for i in range(3):
        p = tmp_path / f"m{i}.yaml"
        p.write_text("x", encoding="utf-8")
        models.append(str(p))
    d = tmp_path / "d.yaml"
    d.write_text("y", encoding="utf-8")
    rc = collect_cache.main(
        [
            "--models",
            *models,
            "--dataset",
            str(d),
            "--cache-root",
            str(tmp_path / "c"),
            "--runner",
            "true",
            "--limit",
            "1",
        ]
    )
    assert rc == 0
