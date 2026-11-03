"""Tests for CLI overrides of the dataset caching block."""

from __future__ import annotations

import pytest

from flowgym.run_setup import apply_cache_cli_overrides


def test_no_override_returns_config_unchanged() -> None:
    """With no flags the caching config is returned untouched."""
    cfg = {"root_dir": "a", "cache_id": "b", "spec": {}}
    assert apply_cache_cli_overrides(cfg, None, None) is cfg


def test_override_sets_root_and_id() -> None:
    """Both flags populate the caching block, leaving the spec intact."""
    cfg = {"spec": {"epe": ("float32", ())}, "warm_start": "index"}
    out = apply_cache_cli_overrides(cfg, "/tmp/x", "my_cache")
    assert out is not None
    assert out["root_dir"] == "/tmp/x"
    assert out["cache_id"] == "my_cache"
    assert out["spec"] == {"epe": ("float32", ())}


def test_partial_override_only_touches_given_field() -> None:
    """A single flag overrides only that field."""
    cfg = {"root_dir": "keep", "cache_id": "old", "spec": {}}
    apply_cache_cli_overrides(cfg, None, "new")
    assert cfg["root_dir"] == "keep"
    assert cfg["cache_id"] == "new"


def test_override_without_caching_block_raises() -> None:
    """Flags require a caching block (which supplies the spec)."""
    with pytest.raises(ValueError, match="caching"):
        apply_cache_cli_overrides(None, "/tmp/x", None)
