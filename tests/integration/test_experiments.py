"""Tests for experiment-study helpers."""

from __future__ import annotations

import argparse
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import experiments.run as run_script
from flowgym import run_setup
from flowgym.experiments import (
    build_study_run_name,
    validate_model_study,
    wandb_git_diff_capture,
    wandb_run_tags,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_text(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        text=True,
    )


def _init_repo(path: Path) -> str:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")

    tracked = path / "tracked.py"
    tracked.write_text("print('v1')\n", encoding="utf-8")
    _git(path, "add", "tracked.py")
    _git(path, "commit", "-m", "initial")
    return _git_text(path, "rev-parse", "HEAD").strip()


def test_build_study_run_name_orders_tags_stably():
    run_name = build_study_run_name(
        {"loss": "huber", "method": "regression"},
        seed=3,
    )

    assert run_name == "loss=huber.method=regression.s3"


def test_build_study_run_name_sanitizes_path_like_tags():
    run_name = build_study_run_name(
        {"group/name": "../unsafe value", "method": "regression"},
        seed=1,
    )

    assert "/" not in run_name
    assert ".." not in run_name
    assert run_name == "group-name=unsafe-value.method=regression.s1"


def test_wandb_run_tags_merges_study_and_dataset_sources():
    tags = wandb_run_tags(
        study_tags={"variant": "baseline", "estimator": "dummy"},
        dataset_tags={"dataset": "piv-class1"},
    )

    assert tags == [
        "dataset=piv-class1",
        "estimator=dummy",
        "variant=baseline",
    ]


def test_wandb_run_tags_returns_empty_when_both_sources_empty():
    assert wandb_run_tags(None, None) == []
    assert wandb_run_tags({}, {}) == []


def test_wandb_run_tags_keeps_collisions_distinct():
    tags = wandb_run_tags(
        study_tags={"variant": "a"},
        dataset_tags={"variant": "b"},
    )

    # Same key in both sources surfaces as two distinct entries — searchable
    # per dimension without collapsing one source into the other.
    assert tags == ["variant=a", "variant=b"]


def test_validate_model_study_requires_matching_experiment_name():
    with pytest.raises(
        ValueError,
        match=r"study\.name must equal exp folder name",
    ):
        validate_model_study(
            {"study": {"name": "other_exp", "tags": {"loss": "huber"}}},
            "expected_exp",
        )


def test_wandb_git_diff_capture_exposes_untracked_files_temporarily(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore generated files")
    (repo / "notes.md").write_text("untracked\n", encoding="utf-8")
    (repo / "ignored.txt").write_text("ignore me\n", encoding="utf-8")

    with wandb_git_diff_capture(repo) as state_label:
        diff_inside = _git_text(repo, "diff", "--binary", "HEAD")
        status_inside = _git_text(
            repo,
            "status",
            "--short",
            "--untracked-files=all",
        )

    diff_after = _git_text(repo, "diff", "--binary", "HEAD")
    status_after = _git_text(
        repo,
        "status",
        "--short",
        "--untracked-files=all",
    )

    # Dirty-with-diff label has the "<sha8>-<diffhash>" form.
    assert "-" in state_label
    assert state_label != "nogit"
    assert "notes.md" in diff_inside
    assert "ignored.txt" not in diff_inside
    assert " A notes.md" in status_inside
    assert "notes.md" not in diff_after
    assert "?? notes.md" in status_after


def test_wandb_git_state_labels_distinguish_dirty_cases(tmp_path: Path):
    def _state(root: Path) -> str:
        with wandb_git_diff_capture(root) as label:
            return label

    tracked_repo = tmp_path / "dirty_tracked"
    _init_repo(tracked_repo)
    (tracked_repo / "tracked.py").write_text(
        "print('dirty')\n", encoding="utf-8"
    )
    dirty_tracked = _state(tracked_repo)

    staged_repo = tmp_path / "staged_new"
    _init_repo(staged_repo)
    (staged_repo / "new.py").write_text("print('new')\n", encoding="utf-8")
    _git(staged_repo, "add", "new.py")
    staged_new = _state(staged_repo)

    deleted_repo = tmp_path / "deleted_tracked"
    _init_repo(deleted_repo)
    (deleted_repo / "tracked.py").unlink()
    deleted_tracked = _state(deleted_repo)

    untracked_repo = tmp_path / "untracked"
    _init_repo(untracked_repo)
    (untracked_repo / "notes.py").write_text(
        "print('notes')\n", encoding="utf-8"
    )
    untracked = _state(untracked_repo)

    labels = {dirty_tracked, staged_new, deleted_tracked, untracked}
    assert len(labels) == 4
    for label in labels:
        assert label != "nogit"
        sha8, _, diff_hash = label.partition("-")
        assert len(sha8) == 8
        assert len(diff_hash) == 12


def test_experiment_launcher_preserves_relative_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp_dir = tmp_path / "dup_models"
    (exp_dir / "group_a").mkdir(parents=True)
    (exp_dir / "group_b").mkdir(parents=True)

    (exp_dir / "dataset.yaml").write_text("seed: 0\n", encoding="utf-8")
    (exp_dir / "exp.yaml").write_text(
        "\n".join(
            [
                "project: flowgym",
                "mode: eval",
                "dataset: dataset.yaml",
                "models:",
                "  - group_a/model.yaml",
                "  - group_b/model.yaml",
                "seeds: [0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model_template = (
        "estimator: dummy\n"
        "estimate_type: flow\n"
        "config:\n"
        "  jit: false\n"
        "study:\n"
        "  name: dup_models\n"
        "  tags:\n"
        "    variant: {variant}\n"
    )
    (exp_dir / "group_a" / "model.yaml").write_text(
        model_template.format(variant="a"),
        encoding="utf-8",
    )
    (exp_dir / "group_b" / "model.yaml").write_text(
        model_template.format(variant="b"),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_check_call(cmd: list[str]) -> None:
        calls.append(cmd)

    monkeypatch.setattr(run_script.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiments/run.py",
            "--exp",
            "dup_models",
            "--exp-root",
            str(tmp_path),
        ],
    )

    run_script.main()

    generated_estimators = [
        Path(cmd[cmd.index("--estimator") + 1]) for cmd in calls
    ]
    assert len(generated_estimators) == 2
    assert generated_estimators[0] != generated_estimators[1]
    assert (
        generated_estimators[0]
        .as_posix()
        .endswith("_generated/models/group_a/model.yaml")
    )
    assert (
        generated_estimators[1]
        .as_posix()
        .endswith("_generated/models/group_b/model.yaml")
    )


def test_setup_study_run_collects_git_state_once_around_wandb_init(
    monkeypatch: pytest.MonkeyPatch,
):
    active_capture = False
    events: list[tuple[str, object]] = []
    capture_entries = 0

    @contextmanager
    def fake_git_capture():
        nonlocal active_capture, capture_entries
        capture_entries += 1
        active_capture = True
        events.append(("capture", "enter"))
        try:
            yield "abcdef01-123456789abc"
        finally:
            events.append(("capture", "exit"))
            active_capture = False

    class FakeWandBHandler:
        def __init__(self, **kwargs: object) -> None:
            events.append(("handler", kwargs))

    class FakeLogger:
        def artifact(self, **kwargs: object) -> None:
            assert active_capture is True
            events.append(("artifact", kwargs["name"]))

    def fake_attach(handler: object, scopes: list[str]) -> None:
        assert active_capture is True
        events.append(("attach", scopes))

    monkeypatch.setattr(run_setup, "wandb_git_diff_capture", fake_git_capture)
    monkeypatch.setattr(run_setup.gg, "WandBHandler", FakeWandBHandler)
    monkeypatch.setattr(run_setup.gg, "attach", fake_attach)
    monkeypatch.setattr(run_setup, "logger", FakeLogger())

    wandb_setup = {
        "project": "flowgym",
        "run_name": "run",
        "group": "study",
        "config": {
            "study": {
                "name": "study",
                "tags": {"variant": "a"},
                "seed": 0,
            },
        },
        "tags": ["variant=a"],
    }
    args = argparse.Namespace(
        mode="eval",
        estimator="model.yaml",
        dataset="dataset.yaml",
        debug=False,
    )

    out_dir = run_setup.setup_study_run(
        args,
        wandb_setup,
        {"seed": 0},
        None,
        {"estimator": "dummy"},
        None,
    )

    assert out_dir is None  # eval mode does not create a training out_dir
    assert capture_entries == 1  # regression: collected exactly once
    handler_kwargs = next(value for kind, value in events if kind == "handler")
    assert handler_kwargs["wandb_init_kwargs"] == {
        "save_code": True,
        "settings": {"code_dir": "."},
    }
    assert handler_kwargs["tags"] == ["variant=a"]
    assert handler_kwargs["config"]["study"] == {
        "name": "study",
        "tags": {"variant": "a"},
        "seed": 0,
        "state_label": "abcdef01-123456789abc",
    }
    assert handler_kwargs["config"]["out_dir"] is None
    assert ("artifact", "dataset_config") in events
    assert ("artifact", "estimator_config") in events


def test_prepare_configs_defers_git_state_for_study_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression: prepare_configs must not collect git state itself.

    Git fields are added later in setup_study_run inside
    wandb_git_diff_capture so the state_label embedded in the W&B config
    matches the directory layout (no two-pass collection that could disagree).
    """

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "prepare_configs must not enter wandb_git_diff_capture; "
            "git state is collected once later in setup_study_run."
        )

    monkeypatch.setattr(run_setup, "wandb_git_diff_capture", _explode)

    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        "estimator: dummy\n"
        "estimate_type: flow\n"
        "config:\n"
        "  jit: false\n"
        "study:\n"
        "  name: probe\n"
        "  tags:\n"
        "    variant: a\n",
        encoding="utf-8",
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("seed: 0\n", encoding="utf-8")

    args = argparse.Namespace(
        mode="eval",
        estimator=str(model_yaml),
        dataset=str(dataset_yaml),
        debug=False,
    )

    *_, wandb_setup = run_setup.prepare_configs(args)
    study = wandb_setup["config"]["study"]
    assert set(study) == {"name", "tags", "seed"}


def test_prepare_configs_merges_study_and_dataset_tags_for_wandb(
    tmp_path: Path,
):
    """Both tag sources must reach wandb's run-level ``tags`` list.

    Study tags identify the variant within an experiment; dataset tags
    describe the dataset's properties. Both are useful for cross-run
    filtering in the W&B UI, so they end up unioned in
    ``wandb_setup["tags"]`` (forwarded to ``wandb.init(tags=...)``).
    """
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        "estimator: dummy\n"
        "estimate_type: flow\n"
        "config:\n"
        "  jit: false\n"
        "study:\n"
        "  name: probe\n"
        "  tags:\n"
        "    variant: baseline\n"
        "    estimator: dummy\n",
        encoding="utf-8",
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        "seed: 0\ntags:\n  dataset: piv-class1\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        mode="eval",
        estimator=str(model_yaml),
        dataset=str(dataset_yaml),
        debug=False,
    )

    *_, wandb_setup = run_setup.prepare_configs(args)

    assert wandb_setup["tags"] == [
        "dataset=piv-class1",
        "estimator=dummy",
        "variant=baseline",
    ]


def test_example_experiment_is_well_formed():
    exp_dir = Path("experiments/example_experiment")
    spec = yaml.safe_load((exp_dir / "exp.yaml").read_text(encoding="utf-8"))

    assert spec["mode"] == "eval"
    assert (exp_dir / spec["dataset"]).exists()

    for model_rel_path in spec["models"]:
        model_path = exp_dir / model_rel_path
        assert model_path.exists()
        model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        study = validate_model_study(model_cfg, "example_experiment")
        assert study["name"] == "example_experiment"
