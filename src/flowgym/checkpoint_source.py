"""Resolve checkpoint source strings into local filesystem paths.

Three source forms are supported:

1. A plain filesystem path (``str`` or :class:`pathlib.Path`). Returned
   resolved to an absolute path, unchanged.

2. A W&B artifact URI of the form ``wandb://<wandb-reference>[/<subpath>]``
   where ``<wandb-reference>`` is anything ``wandb.use_artifact`` /
   ``wandb.Api().artifact`` accept (e.g.
   ``entity/project/name:alias`` or ``name:alias``). The artifact is
   downloaded and the resolver returns the path to the optional
   in-artifact subpath, or the download root if no subpath is given.

3. A Hugging Face Hub URI of the form
   ``hf://<org>/<repo>[@<revision>][/<subpath>]``. The files under
   ``subpath`` (the whole repo if omitted) are downloaded to the Hub
   cache and the resolver returns the local path. Pin ``revision`` to a
   commit SHA for reproducibility. Reading needs no upload rights; a
   private repo needs ``hf auth login``.

The optional in-artifact subpath must follow an explicit ``:alias``
segment so the boundary between the artifact reference and the subpath
is unambiguous.

Typical use::

    from flowgym.checkpoint_source import resolve_checkpoint_source

    path = resolve_checkpoint_source(load_from)
    trained_state = load_model(path, template_state, mode="params_only")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

WANDB_URI_PREFIX = "wandb://"
HF_URI_PREFIX = "hf://"


def is_wandb_uri(source: str | PathLike[str]) -> bool:
    """Return True iff ``source`` looks like a W&B artifact URI.

    Args:
        source: Candidate source string.

    Returns:
        Whether ``source`` starts with the ``wandb://`` scheme prefix.
    """
    return str(source).startswith(WANDB_URI_PREFIX)


def parse_wandb_uri(uri: str) -> tuple[str, str]:
    """Split a ``wandb://`` URI into (wandb-reference, in-artifact subpath).

    Examples::

        parse_wandb_uri("wandb://entity/project/name:alias/jax/checkpoints/0")
        # -> ("entity/project/name:alias", "jax/checkpoints/0")

        parse_wandb_uri("wandb://entity/project/name:alias")
        # -> ("entity/project/name:alias", "")

        parse_wandb_uri("wandb://entity/project/name")
        # -> ("entity/project/name", "")   # wandb defaults the alias

    Args:
        uri: URI string starting with the ``wandb://`` prefix.

    Returns:
        A ``(wandb_reference, subpath)`` pair. ``subpath`` is ``""``
        when no in-artifact subpath was specified.

    Raises:
        ValueError: If the URI does not have the ``wandb://`` prefix,
            or if a subpath was given without an explicit ``:alias``
            anchor (which would make the boundary between artifact
            reference and subpath ambiguous).
    """
    if not is_wandb_uri(uri):
        raise ValueError(f"Expected a {WANDB_URI_PREFIX!r} URI, got {uri!r}.")
    body = uri[len(WANDB_URI_PREFIX) :]
    if not body:
        raise ValueError(
            f"Empty {WANDB_URI_PREFIX} URI: expected an artifact "
            f"reference after the scheme. Got: {uri!r}."
        )
    if ":" in body:
        ref_head, after_alias = body.split(":", 1)
        # Aliases are letters/digits/._- only, so the first ``/`` after
        # the alias unambiguously starts the in-artifact subpath.
        if "/" in after_alias:
            alias, subpath = after_alias.split("/", 1)
        else:
            alias, subpath = after_alias, ""
        if not ref_head or not alias:
            raise ValueError(
                f"Malformed {WANDB_URI_PREFIX} URI: the artifact "
                "reference and the alias around ':' must both be "
                f"non-empty. Got: {uri!r}."
            )
        wandb_ref = f"{ref_head}:{alias}"
        return wandb_ref, subpath

    # No explicit alias. We cannot disambiguate a trailing subpath from
    # the artifact reference itself (which already contains ``/``
    # separators between entity/project/name), so accept only the bare
    # reference and reject any extra path-like content.
    if body.count("/") <= 2:
        return body, ""
    raise ValueError(
        f"Ambiguous {WANDB_URI_PREFIX} URI: a subpath is only allowed "
        "after an explicit ':alias' segment. Got: "
        f"{uri!r}."
    )


def parse_hf_uri(uri: str) -> tuple[str, str | None, str]:
    """Split an ``hf://`` URI into (repo id, revision, subpath).

    Examples::

        parse_hf_uri("hf://org/repo@abc123/raft32/cylinder")
        # -> ("org/repo", "abc123", "raft32/cylinder")

        parse_hf_uri("hf://org/repo")
        # -> ("org/repo", None, "")

    Args:
        uri: URI string starting with the ``hf://`` prefix.

    Returns:
        A ``(repo_id, revision, subpath)`` triple. ``revision`` is
        ``None`` when unspecified; ``subpath`` is ``""`` for the whole
        repository.

    Raises:
        ValueError: If the prefix is missing, the repo id is not
            ``<org>/<repo>`` or the subpath starts with ``/``.
    """
    if not str(uri).startswith(HF_URI_PREFIX):
        raise ValueError(f"Expected a {HF_URI_PREFIX!r} URI, got {uri!r}.")
    org, sep, rest = str(uri)[len(HF_URI_PREFIX) :].partition("/")
    repo, _, subpath = rest.partition("/")
    repo, _, revision = repo.partition("@")
    if not (org and sep and repo):
        raise ValueError(
            f"Malformed {HF_URI_PREFIX} URI: expected "
            f"'hf://<org>/<repo>[@<revision>][/<subpath>]'. Got: {uri!r}."
        )
    if subpath.startswith("/"):
        raise ValueError(
            f"Malformed {HF_URI_PREFIX} URI: the subpath must not start "
            f"with '/'. Got: {uri!r}."
        )
    return f"{org}/{repo}", revision or None, subpath.rstrip("/")


def _inside(
    download_root: Path, subpath: str, origin: str, *, follow_links: bool = True
) -> Path:
    """Return ``download_root / subpath``, which must exist inside the root.

    Args:
        download_root: Directory the source was materialized in.
        subpath: Relative path inside ``download_root``.
        origin: Name of the source, for error messages.
        follow_links: Resolve symlinks before the containment check, so a
            link pointing outside the root is rejected. Pass ``False`` for
            the Hub cache, which links files into its ``blobs/`` directory.

    Returns:
        The path of ``subpath`` under ``download_root``.

    Raises:
        ValueError: If ``subpath`` is absolute or escapes ``download_root``.
        FileNotFoundError: If ``subpath`` does not exist.
    """
    # Defend against absolute subpaths and ``..`` traversal: the
    # subpath comes from a user-supplied URI and must stay strictly
    # inside the download root.
    if Path(subpath).is_absolute():
        raise ValueError(
            f"Subpath {subpath!r} must be relative to the artifact "
            f"download root (got an absolute path)."
        )
    # The Hub cache links files into ``blobs/``, so there the comparison is
    # lexical; elsewhere a symlink must not lead out of the root.
    if follow_links:
        resolved = (download_root / subpath).resolve()
        root = download_root.resolve()
    else:
        resolved = Path(os.path.normpath(download_root / subpath))
        root = Path(os.path.normpath(download_root))
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"Subpath {subpath!r} escapes the artifact download root "
            f"{download_root}."
        )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Subpath {subpath!r} not found inside artifact downloaded "
            f"from {origin!r} (looked in {download_root})."
        )
    return resolved


def resolve_checkpoint_source(
    source: str | PathLike[str],
    *,
    download_dir: str | PathLike[str] | None = None,
) -> Path:
    """Resolve a checkpoint source string to a local filesystem path.

    Args:
        source: A filesystem path, a ``wandb://`` or an ``hf://`` URI. See the
            module docstring for the URI grammar.
        download_dir: Optional override for where wandb materializes
            the artifact files. When ``None``, wandb uses its default
            cache location (typically ``./artifacts/<name>:<version>``).

    Returns:
        Absolute path to the requested checkpoint. For wandb URIs this
        is ``<download_root>/<subpath>`` when a subpath was specified,
        else ``<download_root>``.

    A missing, absolute or escaping subpath raises
    :class:`FileNotFoundError` / :class:`ValueError` (see ``_inside``).

    Raises:
        ImportError: When a wandb or hf URI is given but the matching
            package (``wandb`` / ``huggingface_hub``) is not installed.
    """
    if str(source).startswith(HF_URI_PREFIX):
        return _resolve_hf(str(source), download_dir)
    if not is_wandb_uri(source):
        return Path(source).resolve()

    wandb_ref, subpath = parse_wandb_uri(str(source))

    # Lazy import: wandb is an optional dependency in some environments,
    # and we only need it for the URI path.
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"Resolving a {WANDB_URI_PREFIX} checkpoint source requires "
            "the wandb package. Install it (e.g. `uv add wandb` or "
            "`pip install wandb`) or pass a local filesystem path "
            "instead."
        ) from exc

    run = getattr(wandb, "run", None)
    if run is not None:
        # Recording the load as a run input populates the wandb UI's
        # data-lineage graph for free.
        artifact = run.use_artifact(wandb_ref)
    else:
        artifact = wandb.Api().artifact(wandb_ref)

    if download_dir is None:
        download_root = Path(artifact.download()).resolve()
    else:
        download_root = Path(
            artifact.download(root=str(download_dir))
        ).resolve()

    if not subpath:
        return download_root
    return _inside(download_root, subpath, wandb_ref)


def _resolve_hf(uri: str, download_dir: str | PathLike[str] | None) -> Path:
    """Download the files an ``hf://`` URI points to and return their path.

    Args:
        uri: The ``hf://`` URI.
        download_dir: Optional ``local_dir`` for the download.

    Returns:
        Local path of the URI's subpath (the repo root without one).

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    repo_id, revision, subpath = parse_hf_uri(uri)
    # Lazy import: only needed for the URI path.
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"Resolving an {HF_URI_PREFIX} checkpoint source requires the "
            "huggingface_hub package (`uv sync --extra hf`)."
        ) from exc

    download_root = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=[f"{subpath}/*", subpath] if subpath else None,
            local_dir=None if download_dir is None else str(download_dir),
        )
    ).resolve()
    if not subpath:
        return download_root
    return _inside(
        download_root,
        subpath,
        f"{repo_id}@{revision or 'main'}",
        follow_links=False,
    )
