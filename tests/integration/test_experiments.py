"""Integration tests for the experiment-study end-to-end pipeline.

Covers run naming, tag merging, git-state capture, and launcher behaviour.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import experiments.run as run_script
from flowgym import ALL_ESTIMATORS, run_setup
from flowgym.experiments import (
    build_study_run_name,
    validate_model_study,
    wandb_git_diff_capture,
    wandb_run_tags,
)
from flowgym.utils import load_configuration


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
    """build_study_run_name produces a deterministically ordered tag string."""
    run_name = build_study_run_name(
        {"loss": "huber", "method": "regression"},
        seed=3,
    )

    assert run_name == "loss=huber.method=regression.s3"


def test_build_study_run_name_sanitizes_path_like_tags():
    """build_study_run_name strips path separators and unsafe characters."""
    run_name = build_study_run_name(
        {"group/name": "../unsafe value", "method": "regression"},
        seed=1,
    )

    assert "/" not in run_name
    assert ".." not in run_name
    assert run_name == "group-name=unsafe-value.method=regression.s1"


def test_wandb_run_tags_merges_study_and_dataset_sources():
    """wandb_run_tags combines study and dataset tags into a sorted list."""
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
    """wandb_run_tags returns an empty list when both tag sources are absent."""
    assert wandb_run_tags(None, None) == []
    assert wandb_run_tags({}, {}) == []


def test_wandb_run_tags_keeps_collisions_distinct():
    """Duplicate keys across tag sources appear as two separate entries."""
    tags = wandb_run_tags(
        study_tags={"variant": "a"},
        dataset_tags={"variant": "b"},
    )

    # Same key in both sources surfaces as two distinct entries — searchable
    # per dimension without collapsing one source into the other.
    assert tags == ["variant=a", "variant=b"]


def test_validate_model_study_requires_matching_experiment_name():
    """validate_model_study raises ValueError on study/folder name mismatch."""
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
    """wandb_git_diff_capture stages untracked files only within its context."""
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
    """Each distinct dirty-repo scenario yields a unique git state label."""

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
    """Launcher generates unique per-model config paths for each model entry."""
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


def test_experiment_launcher_applies_per_model_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A model's ``mode:`` overrides the experiment mode for its runs."""
    exp_dir = tmp_path / "modes"
    exp_dir.mkdir()
    (exp_dir / "dataset.yaml").write_text("seed: 0\n", encoding="utf-8")
    (exp_dir / "exp.yaml").write_text(
        "mode: train-supervised\ndataset: dataset.yaml\n"
        "models: [trained.yaml, evaluated.yaml]\n",
        encoding="utf-8",
    )
    model = "estimator: dummy\nstudy:\n  name: modes\n  tags: {{v: {v}}}\n"
    (exp_dir / "trained.yaml").write_text(model.format(v="a"), encoding="utf-8")
    (exp_dir / "evaluated.yaml").write_text(
        "mode: eval\n" + model.format(v="b"), encoding="utf-8"
    )

    calls: list[list[str]] = []
    monkeypatch.setattr(run_script.subprocess, "check_call", calls.append)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--exp", "modes", "--exp-root", str(tmp_path)],
    )
    run_script.main()

    modes = {}
    for cmd in calls:
        model = Path(cmd[cmd.index("--estimator") + 1]).name
        modes[model] = cmd[cmd.index("--mode") + 1]
    assert modes == {
        "trained.yaml": "train-supervised",
        "evaluated.yaml": "eval",
    }
    generated = exp_dir / "_generated" / "models" / "evaluated.yaml"
    assert "mode" not in yaml.safe_load(generated.read_text())


def test_setup_study_run_collects_git_state_once_around_wandb_init(
    monkeypatch: pytest.MonkeyPatch,
):
    """setup_study_run captures git state exactly once, wrapping W&B init."""
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
    """prepare_configs does not enter wandb_git_diff_capture itself."""

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
    """prepare_configs places both study and dataset tags in wandb_setup."""
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
    """example_experiment config references valid dataset and model files."""
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


def test_experiment_launcher_aggregates_seed_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Per-seed ``--summary-out`` files are collected into results.json."""
    exp_dir = tmp_path / "agg"
    exp_dir.mkdir()
    (exp_dir / "dataset.yaml").write_text("seed: 0\n", encoding="utf-8")
    (exp_dir / "exp.yaml").write_text(
        "mode: eval\ndataset: dataset.yaml\nmodels: [m.yaml]\nseeds: [0, 1]\n"
        "metrics: [mean_epe, p99_epe]\n",
        encoding="utf-8",
    )
    (exp_dir / "m.yaml").write_text(
        "estimator: dummy\nstudy:\n  name: agg\n  tags: {v: a}\n",
        encoding="utf-8",
    )

    def fake_check_call(cmd: list[str]) -> None:
        dataset = Path(cmd[cmd.index("--dataset") + 1])
        seed = yaml.safe_load(dataset.read_text())["seed"]
        out = Path(cmd[cmd.index("--summary-out") + 1])
        out.write_text(
            json.dumps({"mean_epe": 0.25 + 0.5 * seed, "max_epe": 2.0})
        )

    monkeypatch.setattr(run_script.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--exp", "agg", "--exp-root", str(tmp_path)]
    )
    run_script.main()

    results = json.loads((exp_dir / "results.json").read_text())
    # Only listed metrics are kept; p99_epe is missing from the summaries
    assert results == {
        "m.yaml": {
            "mean_epe": {
                "mean": 0.5,
                "std": 0.25,
                "seeds": {"0": 0.25, "1": 0.75},
            }
        }
    }


def test_experiment_launcher_drops_stale_results_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed launch leaves no results.json from an earlier run."""
    exp_dir = tmp_path / "stale"
    exp_dir.mkdir()
    (exp_dir / "dataset.yaml").write_text("seed: 0\n", encoding="utf-8")
    (exp_dir / "exp.yaml").write_text(
        "mode: eval\ndataset: dataset.yaml\nmodels: [m.yaml]\n",
        encoding="utf-8",
    )
    (exp_dir / "m.yaml").write_text(
        "estimator: dummy\nstudy:\n  name: stale\n  tags: {v: a}\n",
        encoding="utf-8",
    )
    (exp_dir / "results.json").write_text("{}", encoding="utf-8")

    def failing_check_call(cmd: list[str]) -> None:
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(run_script.subprocess, "check_call", failing_check_call)
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--exp", "stale", "--exp-root", str(tmp_path)]
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_script.main()
    assert not (exp_dir / "results.json").exists()


def test_experiment_launcher_skips_models_with_missing_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A model with only some seed summaries is left out of results.json."""
    exp_dir = tmp_path / "partial"
    exp_dir.mkdir()
    (exp_dir / "dataset.yaml").write_text("seed: 0\n", encoding="utf-8")
    (exp_dir / "exp.yaml").write_text(
        "mode: eval\ndataset: dataset.yaml\nmodels: [a.yaml, b.yaml]\n"
        "seeds: [0, 1]\n",
        encoding="utf-8",
    )
    for name in ("a", "b"):
        (exp_dir / f"{name}.yaml").write_text(
            "estimator: dummy\nstudy:\n  name: partial\n"
            f"  tags: {{v: {name}}}\n",
            encoding="utf-8",
        )

    def fake_check_call(cmd: list[str]) -> None:
        out = Path(cmd[cmd.index("--summary-out") + 1])
        if not (out.parent.name == "b.yaml" and out.name == "seed1.json"):
            out.write_text(json.dumps({"mean_epe": 0.5}))

    monkeypatch.setattr(run_script.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--exp", "partial", "--exp-root", str(tmp_path)]
    )
    run_script.main()

    results = json.loads((exp_dir / "results.json").read_text())
    assert list(results) == ["a.yaml"]


def test_experiment_launcher_applies_dataset_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A model's ``dataset_overrides`` are deep-merged into its dataset."""
    exp_dir = tmp_path / "ovr"
    exp_dir.mkdir()
    (exp_dir / "dataset.yaml").write_text(
        "batch_size: 16\n"
        "validation: {interval: 5, dataset: {batch_size: 16}}\n",
        encoding="utf-8",
    )
    (exp_dir / "exp.yaml").write_text(
        "mode: eval\ndataset: dataset.yaml\nmodels: [m.yaml]\n",
        encoding="utf-8",
    )
    (exp_dir / "m.yaml").write_text(
        "estimator: dummy\nstudy:\n  name: ovr\n  tags: {v: a}\n"
        "dataset_overrides:\n  batch_size: 64\n"
        "  validation: {dataset: {batch_size: 64}}\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []
    monkeypatch.setattr(run_script.subprocess, "check_call", calls.append)
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--exp", "ovr", "--exp-root", str(tmp_path)]
    )
    run_script.main()

    [cmd] = calls
    dataset = yaml.safe_load(Path(cmd[cmd.index("--dataset") + 1]).read_text())
    assert dataset["batch_size"] == 64
    assert dataset["validation"] == {
        "interval": 5,
        "dataset": {"batch_size": 64},
    }
    model = yaml.safe_load(Path(cmd[cmd.index("--estimator") + 1]).read_text())
    assert "dataset_overrides" not in model


@pytest.mark.parametrize(
    ("row", "hashes"),
    [
        (
            "dis_20hz",
            "007d9a94 86725532 9f485cd6 ad340132 b68adbad "
            "c072b2f2 c83b3862 cc366684 f37adac3 f581f19f",
        ),
        (
            "dis_1khz",
            "121533ae 37c4290e 555d463e 65957b24 6aec2a81 "
            "9215fc07 dae3d3ef dc377309 dcbc9026 ea3880db",
        ),
    ],
)
def test_art_of_piv_libraries_hit_the_paper_caches(row, hashes):
    """The merged library configs hash to the cache ids the paper used."""
    library = load_configuration(f"experiments/art_of_piv/libraries/{row}.yaml")
    cache_ids = []
    for path in library["estimators"]:
        cfg = load_configuration(path)
        estimator = ALL_ESTIMATORS[cfg["estimator"]].from_config(cfg["config"])
        cache_ids.append(
            library["cache_id_base"] + estimator.get_cache_id_suffix(None)
        )
    assert cache_ids == [f"piv-class1-dis_c{h}" for h in hashes.split()]
