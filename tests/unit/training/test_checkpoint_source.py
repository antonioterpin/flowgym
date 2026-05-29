"""Tests for ``flowgym.checkpoint_source``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from flowgym.checkpoint_source import (
    WANDB_URI_PREFIX,
    is_wandb_uri,
    parse_wandb_uri,
    resolve_checkpoint_source,
)

# ──────────────────────────────────────────────────────────────────────────
# Prefix detection
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source, expected",
    [
        ("wandb://entity/project/name:alias", True),
        ("wandb://name:alias", True),
        ("wandb://", True),  # malformed but matches the prefix
        ("/tmp/some/dir", False),
        ("relative/path", False),
        ("https://example.com/foo", False),
        ("wandb-artifact://entity/project/name:alias", False),
        ("", False),
    ],
)
def test_is_wandb_uri(source: str, expected: bool) -> None:
    """``is_wandb_uri`` accepts only ``wandb://`` prefixes."""
    assert is_wandb_uri(source) is expected


def test_wandb_uri_prefix_constant() -> None:
    """The exported prefix constant is the canonical scheme string."""
    assert WANDB_URI_PREFIX == "wandb://"


# ──────────────────────────────────────────────────────────────────────────
# URI parsing
# ──────────────────────────────────────────────────────────────────────────


def test_parse_full_uri_with_entity_alias_subpath() -> None:
    """Full URI with entity/project/name:alias/subpath is parsed cleanly."""
    ref, subpath = parse_wandb_uri(
        "wandb://artofpiv/RAFT32_finetuning/"
        "raft32_piv_pretrained:pretrained/jax/checkpoints/0"
    )
    assert ref == "artofpiv/RAFT32_finetuning/raft32_piv_pretrained:pretrained"
    assert subpath == "jax/checkpoints/0"


def test_parse_uri_without_subpath_returns_empty_string() -> None:
    """When no subpath is present after the alias, the subpath is ``''``."""
    ref, subpath = parse_wandb_uri(
        "wandb://artofpiv/RAFT32_finetuning/raft32_piv_pretrained:pretrained"
    )
    assert ref == "artofpiv/RAFT32_finetuning/raft32_piv_pretrained:pretrained"
    assert subpath == ""


def test_parse_uri_without_alias_keeps_default_latest() -> None:
    """When no ``:alias`` is given the wandb reference is passed through.

    The downstream wandb API defaults to the ``latest`` alias in that
    case; the parser itself must not invent one.
    """
    ref, subpath = parse_wandb_uri(
        "wandb://artofpiv/RAFT32_finetuning/raft32_piv_pretrained"
    )
    assert ref == "artofpiv/RAFT32_finetuning/raft32_piv_pretrained"
    assert subpath == ""


def test_parse_uri_with_subpath_but_no_alias() -> None:
    """The subpath is everything after the last ``/`` of the artifact ref.

    Without an alias the first slash in ``project/name`` is part of the
    artifact reference, so we can only safely accept subpaths together
    with an explicit alias to keep the boundary unambiguous.
    """
    with pytest.raises(ValueError):
        parse_wandb_uri(
            "wandb://artofpiv/RAFT32_finetuning/"
            "raft32_piv_pretrained/jax/checkpoints/0"
        )


def test_parse_uri_rejects_non_wandb_scheme() -> None:
    """The parser raises on non-``wandb://`` input."""
    with pytest.raises(ValueError):
        parse_wandb_uri("/tmp/some/dir")


@pytest.mark.parametrize(
    "uri",
    [
        "wandb://",  # empty body
        "wandb://:alias",  # empty artifact reference
        "wandb://entity/project/name:",  # empty alias
        "wandb://entity/project/name:/sub",  # empty alias + subpath
        "wandb://:alias/sub",  # empty ref with subpath
    ],
)
def test_parse_uri_rejects_empty_components(uri: str) -> None:
    """Empty body, ref, or alias produce a clear ``ValueError``."""
    with pytest.raises(ValueError):
        parse_wandb_uri(uri)


# ──────────────────────────────────────────────────────────────────────────
# resolve_checkpoint_source
# ──────────────────────────────────────────────────────────────────────────


def test_resolve_plain_path_passes_through(tmp_path: Path) -> None:
    """Non-wandb sources are returned as absolute ``Path`` unchanged."""
    target = tmp_path / "ckpt"
    target.mkdir()
    resolved = resolve_checkpoint_source(str(target))
    assert resolved == target.resolve()


def test_resolve_plain_path_accepts_path_object(tmp_path: Path) -> None:
    """``Path`` inputs are returned resolved."""
    target = tmp_path / "ckpt"
    target.mkdir()
    resolved = resolve_checkpoint_source(target)
    assert resolved == target.resolve()


def _install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    download_root: Path,
    *,
    active_run: bool,
) -> dict[str, Any]:
    """Install a stub ``wandb`` module on ``sys.modules``.

    Records the artifact reference passed to ``use_artifact`` / ``Api``
    and returns the captures dict so the caller can assert on it.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        download_root: Path to return from ``Artifact.download``.
        active_run: When True, ``wandb.run`` returns a run-like mock so
            the resolver uses ``run.use_artifact`` (recording the load
            as a run input). When False, ``wandb.run`` is ``None`` and
            the resolver falls back to ``wandb.Api().artifact``.

    Returns:
        Dict with keys ``ref`` (the reference passed in) and ``route``
        ("run" or "api") so the test can assert which path was taken.
    """
    captures: dict[str, Any] = {"ref": None, "route": None}

    fake_artifact = MagicMock()
    fake_artifact.download.return_value = str(download_root)

    fake_run = MagicMock()

    def _use_artifact(ref: str) -> Any:
        captures["ref"] = ref
        captures["route"] = "run"
        return fake_artifact

    fake_run.use_artifact.side_effect = _use_artifact

    fake_api_instance = MagicMock()

    def _api_artifact(ref: str) -> Any:
        captures["ref"] = ref
        captures["route"] = "api"
        return fake_artifact

    fake_api_instance.artifact.side_effect = _api_artifact

    fake_wandb = MagicMock()
    fake_wandb.run = fake_run if active_run else None
    fake_wandb.Api.return_value = fake_api_instance

    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    return captures


def test_resolve_wandb_uri_uses_run_use_artifact_when_run_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An active wandb run routes the download through ``run.use_artifact``.

    This is the preferred path because it records the artifact as a
    run input so the wandb UI can trace data lineage.
    """
    captures = _install_fake_wandb(monkeypatch, tmp_path, active_run=True)

    resolved = resolve_checkpoint_source("wandb://entity/project/name:alias")

    assert captures["route"] == "run"
    assert captures["ref"] == "entity/project/name:alias"
    assert resolved == tmp_path.resolve()


def test_resolve_wandb_uri_uses_api_when_no_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no wandb run is active, ``Api().artifact`` is used instead."""
    captures = _install_fake_wandb(monkeypatch, tmp_path, active_run=False)

    resolved = resolve_checkpoint_source("wandb://entity/project/name:alias")

    assert captures["route"] == "api"
    assert captures["ref"] == "entity/project/name:alias"
    assert resolved == tmp_path.resolve()


def test_resolve_wandb_uri_appends_subpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-artifact subpath is appended to the download root."""
    inner = tmp_path / "jax" / "checkpoints" / "0"
    inner.mkdir(parents=True)
    _install_fake_wandb(monkeypatch, tmp_path, active_run=True)

    resolved = resolve_checkpoint_source(
        "wandb://entity/project/name:alias/jax/checkpoints/0"
    )
    assert resolved == inner.resolve()


def test_resolve_wandb_uri_raises_when_subpath_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subpath that does not exist in the downloaded tree fails loudly.

    Silent fallback to the download root would mask user typos and
    surface as a confusing "no checkpoints found" error from
    ``load_model`` much further down the stack.
    """
    _install_fake_wandb(monkeypatch, tmp_path, active_run=True)

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_source("wandb://entity/project/name:alias/nope/here")


def test_resolve_wandb_uri_rejects_subpath_escaping_download_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subpath containing ``..`` that escapes the root is rejected.

    Even if such a path happens to exist on disk, returning it would
    let a wandb URI hand back an arbitrary filesystem location, which
    the caller did not authorize.
    """
    download_root = tmp_path / "download"
    download_root.mkdir()
    # Create a sibling that the ``..`` traversal would otherwise reach.
    (tmp_path / "outside").mkdir()

    _install_fake_wandb(monkeypatch, download_root, active_run=True)

    with pytest.raises(ValueError, match="escapes the artifact download root"):
        resolve_checkpoint_source(
            "wandb://entity/project/name:alias/../outside"
        )


def test_resolve_wandb_uri_rejects_absolute_subpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute in-artifact subpath is rejected up front."""
    _install_fake_wandb(monkeypatch, tmp_path, active_run=True)

    # The parser keeps a leading ``/`` as part of the subpath, so feed
    # one directly through ``resolve_checkpoint_source``'s plumbing by
    # constructing a URI where the subpath starts with ``/``. Because
    # ``parse_wandb_uri`` splits on the first ``/`` after the alias,
    # the subpath portion that reaches the safety check is e.g.
    # ``/etc/passwd``.
    with pytest.raises(ValueError, match="absolute"):
        resolve_checkpoint_source(
            "wandb://entity/project/name:alias//etc/passwd"
        )
