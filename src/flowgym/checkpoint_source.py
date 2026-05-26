"""Resolve checkpoint source strings into local filesystem paths.

Two source forms are supported:

1. A plain filesystem path (``str`` or :class:`pathlib.Path`). Returned
   resolved to an absolute path, unchanged.

2. A W&B artifact URI of the form ``wandb://<wandb-reference>[/<subpath>]``
   where ``<wandb-reference>`` is anything ``wandb.use_artifact`` /
   ``wandb.Api().artifact`` accept (e.g.
   ``entity/project/name:alias`` or ``name:alias``). The artifact is
   downloaded and the resolver returns the path to the optional
   in-artifact subpath, or the download root if no subpath is given.

The optional in-artifact subpath must follow an explicit ``:alias``
segment so the boundary between the artifact reference and the subpath
is unambiguous.

Typical use::

    from flowgym.checkpoint_source import resolve_checkpoint_source

    path = resolve_checkpoint_source(load_from)
    trained_state = load_model(path, template_state, mode="params_only")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

WANDB_URI_PREFIX = "wandb://"


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


def resolve_checkpoint_source(
    source: str | PathLike[str],
    *,
    download_dir: str | PathLike[str] | None = None,
) -> Path:
    """Resolve a checkpoint source string to a local filesystem path.

    Args:
        source: Either a filesystem path or a ``wandb://`` URI. See the
            module docstring for the URI grammar.
        download_dir: Optional override for where wandb materializes
            the artifact files. When ``None``, wandb uses its default
            cache location (typically ``./artifacts/<name>:<version>``).

    Returns:
        Absolute path to the requested checkpoint. For wandb URIs this
        is ``<download_root>/<subpath>`` when a subpath was specified,
        else ``<download_root>``.

    Raises:
        FileNotFoundError: When a wandb URI specifies an in-artifact
            subpath that does not exist in the downloaded tree.
        ImportError: When a wandb URI is given but the ``wandb``
            package is not installed.
        ValueError: When the in-artifact subpath is absolute or
            otherwise escapes the artifact's download root (e.g. via
            ``..`` segments).
    """
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

    # Defend against absolute subpaths and ``..`` traversal: the
    # subpath comes from a user-supplied URI and must stay strictly
    # inside the artifact's download root.
    if Path(subpath).is_absolute():
        raise ValueError(
            f"Subpath {subpath!r} must be relative to the artifact "
            f"download root (got an absolute path)."
        )
    resolved = (download_root / subpath).resolve()
    if not resolved.is_relative_to(download_root):
        raise ValueError(
            f"Subpath {subpath!r} escapes the artifact download root "
            f"{download_root}."
        )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Subpath {subpath!r} not found inside artifact downloaded "
            f"from {wandb_ref!r} (looked in {download_root})."
        )
    return resolved
