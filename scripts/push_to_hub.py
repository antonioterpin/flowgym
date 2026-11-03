"""Upload checkpoint folders to a Hugging Face model repo.

Maintainer tool: reproducing results only downloads (``hf://`` in
``load_from``) and never needs this script. Each ``--upload`` copies a local
folder to a path inside the repo, in one commit, which is printed so configs
can pin it (``hf://<repo>@<commit>/<path>``).

Example:

    uv run --extra hf python scripts/push_to_hub.py \\
        --repo activefluidcontrol/art-of-piv-models --private \\
        --upload ~/raft_checkpoints/cylinder:raft32/cylinder \\
        --readme README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main() -> None:
    """Create the repo if needed and upload the given folders in one commit.

    Raises:
        SystemExit: If an ``--upload`` value is not ``LOCAL_DIR:PATH_IN_REPO``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="<org>/<name>")
    parser.add_argument(
        "--upload",
        action="append",
        default=[],
        metavar="LOCAL_DIR:PATH_IN_REPO",
        help="Folder to upload; repeatable.",
    )
    parser.add_argument("--readme", help="File uploaded as the repo README.md.")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (only applies when it is created).",
    )
    parser.add_argument("--message", default="Upload checkpoints")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(
        args.repo, repo_type="model", private=args.private, exist_ok=True
    )

    operations = []
    for item in args.upload:
        local, _, dest = item.rpartition(":")
        folder = Path(local).expanduser()
        if not folder.is_dir() or not dest:
            raise SystemExit(
                f"--upload expects LOCAL_DIR:PATH_IN_REPO, got {item!r}"
            )
        for file in sorted(p for p in folder.rglob("*") if p.is_file()):
            operations.append(
                CommitOperationAdd(
                    f"{dest}/{file.relative_to(folder).as_posix()}", str(file)
                )
            )
    if args.readme:
        operations.append(CommitOperationAdd("README.md", args.readme))

    info = api.create_commit(
        args.repo, operations=operations, commit_message=args.message
    )
    print(f"{len(operations)} files -> {args.repo}@{info.oid}")


if __name__ == "__main__":
    main()
