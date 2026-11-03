"""Helpers for experiment studies and W&B reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

JSONScalar = str | int | float | bool | None
RUN_NAME_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._=-]+")


def validate_scalar_tags(tags: object, context: str) -> dict[str, JSONScalar]:
    """Validate JSON-serializable scalar tags for W&B-friendly metadata.

    Args:
        tags: Object expected to be a mapping of scalar tag values.
        context: Human-readable configuration path used in errors.

    Returns:
        Validated tag mapping.

    Raises:
        ValueError: If the tags object or any tag key/value is invalid.
    """
    if not isinstance(tags, dict):
        raise ValueError(f"{context} must be a dict.")

    validated: dict[str, JSONScalar] = {}
    for key, value in tags.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings.")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(
                f"{context} values must be JSON-serializable "
                "(str/int/float/bool/None)."
            )
        validated[key] = value
    return validated


def validate_model_study(
    model_cfg: Mapping[str, Any], exp_name: str
) -> dict[str, Any]:
    """Validate and normalize a study block from a model config.

    Args:
        model_cfg: Model configuration containing a ``study`` block.
        exp_name: Expected experiment folder name.

    Returns:
        Normalized study metadata.

    Raises:
        ValueError: If the study block is missing or names a different
            experiment.
    """
    study = model_cfg.get("study")
    if not isinstance(study, dict):
        raise ValueError("Model config must contain a dict 'study'.")

    if study.get("name") != exp_name:
        raise ValueError(
            f"study.name must equal exp folder name: {exp_name!r}. "
            f"Got {study.get('name')!r}."
        )

    return {
        "name": exp_name,
        "tags": validate_scalar_tags(study.get("tags"), "study.tags"),
    }


def build_study_run_name(tags: Mapping[str, JSONScalar], seed: int) -> str:
    """Build a deterministic run name from study tags and seed.

    Args:
        tags: Study tags to encode into the run name.
        seed: Dataset seed for the run.

    Returns:
        Filesystem-safe run name.
    """
    tag_items = [
        f"{_sanitize_run_name_component(key)}="
        f"{_sanitize_run_name_component(tags[key])}"
        for key in sorted(tags)
    ]
    tag_part = ".".join(tag_items) if tag_items else "run"
    return f"{tag_part}.s{seed}"


def wandb_run_tags(
    study_tags: Mapping[str, JSONScalar] | None = None,
    dataset_tags: Mapping[str, JSONScalar] | None = None,
) -> list[str]:
    """Flatten study and dataset tag dicts into W&B run-level tag strings.

    W&B run tags are a flat ``list[str]`` (the chips shown in the run UI and
    used for cross-run filtering). Each ``(key, value)`` pair from either
    source becomes a single ``"key=value"`` entry so the originating
    dimension is recoverable when filtering.

    Args:
        study_tags: Tags from the model's ``study`` block, or ``None``.
        dataset_tags: Tags from the dataset config, or ``None``.

    Returns:
        Sorted list of ``"key=value"`` strings; empty when both inputs are
        empty or ``None``.
    """
    items: list[str] = []
    for src in (study_tags, dataset_tags):
        if src:
            items.extend(f"{key}={value}" for key, value in src.items())
    return sorted(items)


@contextmanager
def wandb_git_diff_capture(
    root: str | Path | None = None,
) -> Iterator[str]:
    """Temporarily expose untracked files to W&B's native git diff capture.

    W&B records the current commit and ``diff.patch`` when
    ``wandb.init(save_code=True)`` runs. Git omits untracked files from that
    diff by default, so this context copies the current index to a temporary
    index, marks non-ignored untracked files with ``git add --intent-to-add``,
    and restores the process environment afterward.

    ``wandb.init(save_code=True)`` must be called inside the ``with`` block.
    The temporary ``GIT_INDEX_FILE`` is reverted on exit, so any wandb run
    initialised after the context has closed will record the unmodified
    ``git diff HEAD`` and lose the untracked-file additions.

    Args:
        root: Repository path to inspect, or ``None`` for the current working
            directory.

    Yields:
        str: A state label — ``"nogit"`` outside a git repo, ``"<sha8>"`` for
            a clean HEAD, or ``"<sha8>-<diffhash>"`` when the temporary-index
            view differs from ``HEAD``.
    """
    repo_root = _find_git_root(root)
    if repo_root is None:
        yield "nogit"
        return

    previous_index = os.environ.get("GIT_INDEX_FILE")
    with tempfile.TemporaryDirectory(prefix="flowgym-wandb-index-") as tmp_dir:
        temp_index = Path(tmp_dir) / "index"
        real_index = _current_git_index(repo_root, previous_index)
        if real_index is not None and real_index.exists():
            shutil.copy2(real_index, temp_index)

        os.environ["GIT_INDEX_FILE"] = str(temp_index)
        env = {**os.environ, "GIT_INDEX_FILE": str(temp_index)}
        try:
            raw_paths = subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                cwd=repo_root,
                env=env,
            )
            untracked_paths = sorted(
                Path(rel)
                for rel in raw_paths.decode("utf-8").split("\0")
                if rel
            )
            if untracked_paths:
                subprocess.run(
                    [
                        "git",
                        "add",
                        "--intent-to-add",
                        "--",
                        *(str(p) for p in untracked_paths),
                    ],
                    cwd=repo_root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            yield _build_wandb_git_state(repo_root, env)
        finally:
            if previous_index is None:
                os.environ.pop("GIT_INDEX_FILE", None)
            else:
                os.environ["GIT_INDEX_FILE"] = previous_index


def _find_git_root(root: str | Path | None) -> Path | None:
    cwd = Path.cwd() if root is None else Path(root).resolve()
    output = _git_output(cwd, "rev-parse", "--show-toplevel")
    return Path(output).resolve() if output else None


def _current_git_index(
    repo_root: Path, override_index: str | None
) -> Path | None:
    if override_index:
        return Path(override_index)
    index_path = _git_output(repo_root, "rev-parse", "--git-path", "index")
    if not index_path:
        return None
    path = Path(index_path)
    return path if path.is_absolute() else repo_root / path


def _build_wandb_git_state(repo_root: Path, env: Mapping[str, str]) -> str:
    commit = _git_output(repo_root, "rev-parse", "--verify", "HEAD", env=env)
    if not commit:
        return "nogit"
    porcelain = (
        _git_output(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            env=env,
        )
        or ""
    )
    if not porcelain:
        return commit[:8]
    diff_text = (
        _git_output(repo_root, "diff", "--binary", "HEAD", env=env) or ""
    )
    diff_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()[:12]
    return f"{commit[:8]}-{diff_sha}"


def _git_output(
    root: Path,
    *args: str,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Run ``git`` and return stripped stdout, or ``None`` on non-zero exit.

    Args:
        root: Working directory for the git invocation.
        *args: Arguments to pass to ``git``.
        env: Optional environment overrides (e.g. ``GIT_INDEX_FILE``).

    Returns:
        Stripped stdout when git succeeds, otherwise ``None``.
    """
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _sanitize_run_name_component(value: object) -> str:
    text = RUN_NAME_SAFE_CHARS.sub("-", str(value))
    text = text.strip(".-_")
    if text in {"", ".", ".."}:
        return "x"
    return text


# A README block between ``<!-- BEGIN <file> -->`` and ``<!-- END <file> -->``
# holds a table rendered from the named file in the README's folder.
README_BLOCK = re.compile(
    r"<!-- BEGIN (\S+) -->\n.*?<!-- END \1 -->", re.DOTALL
)


def _mean_std(stat: Mapping[str, Any]) -> str:
    if stat["std"] == 0:
        return f"{stat['mean']:.5f}"
    return f"{stat['mean']:.5f} ± {stat['std']:.5f}"


def render_results(data: Mapping[str, Any]) -> str:
    """Render an experiment's results file as a Markdown table.

    Args:
        data: Either ``run.py``'s ``{model: {metric: {mean, std, seeds}}}``,
            a library-selection sweep with ``sizes`` (appendix B), or a flat
            ``{key: value}`` mapping.

    Returns:
        The table.
    """
    if "sizes" in data:
        lines = [
            f"Pool: {data['pool_size']} configs ({data['dropped_nonfinite']} "
            f"diverging dropped), {data['tune_frames']} tune frames.",
            "",
            "| K | greedy | exact | gap (px) | gap (%) | MILP (s) |",
            "|---|---|---|---|---|---|",
        ]
        for k, size in data["sizes"].items():
            lines.append(
                f"| {k} | {size['greedy_aepe']:.5f} | {size['exact_aepe']:.5f} "
                f"| {size['gap_px']:.5f} | {size['gap_percent']:.2f} "
                f"| {size['milp_seconds']:.1f} |"
            )
        return "\n".join(lines)
    first = next(iter(data.values()))
    if not isinstance(first, Mapping):
        lines = ["| key | value |", "|---|---|"]
        lines += [f"| `{key}` | {value} |" for key, value in data.items()]
        return "\n".join(lines)
    # run.py keeps each model's metrics independently: take every column
    metrics = list(dict.fromkeys(m for stats in data.values() for m in stats))
    lines = [
        "| model | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for model, stats in data.items():
        cells = [_mean_std(stats[m]) if m in stats else "" for m in metrics]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_readme_results(readme: Path) -> str:
    """Return a README's text with every results block re-rendered.

    Args:
        readme: The README; each block names a file in the same folder.

    Returns:
        The updated text.
    """

    def block(match: re.Match[str]) -> str:
        name = match.group(1)
        # An experiment not run yet keeps its placeholder
        if not (readme.parent / name).is_file():
            return match.group(0)
        data = json.loads((readme.parent / name).read_text())
        table = render_results(data)
        return f"<!-- BEGIN {name} -->\n{table}\n<!-- END {name} -->"

    return README_BLOCK.sub(block, readme.read_text())


def update_readme_results(readme: Path) -> bool:
    """Re-render a README's results blocks in place.

    Args:
        readme: The README to update; a missing file is left alone.

    Returns:
        Whether the file changed.
    """
    if not readme.is_file():
        return False
    text = render_readme_results(readme)
    if text == readme.read_text():
        return False
    readme.write_text(text)
    return True
