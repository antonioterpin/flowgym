"""Exercise Hydra overrides and isolated multirun outputs end to end."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("hydra")


def test_hydra_multirun_keeps_jobs_and_source_configs_separate(tmp_path):
    """Two seeds and two overrides produce four isolated, reproducible jobs."""
    exp_dir = tmp_path / "probe"
    exp_dir.mkdir()
    (exp_dir / "dataset.yaml").write_text("seed: 99\nvalue: 0\n")
    (exp_dir / "model.yaml").write_text(
        "estimator: dummy\nconfig: {jit: false}\n"
        "study: {name: probe, tags: {variant: baseline}}\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text("""import argparse, json, os
from pathlib import Path
import yaml
p = argparse.ArgumentParser()
for name in ("mode", "estimator", "dataset", "summary-out"):
    p.add_argument("--" + name)
a = p.parse_args()
d = yaml.safe_load(Path(a.dataset).read_text())
m = yaml.safe_load(Path(a.estimator).read_text())
Path(a.summary_out).write_text(json.dumps({"mean_epe": d["seed"] + d["value"]}))
Path(a.summary_out).with_suffix(".context.json").write_text(json.dumps({
    "cwd": os.getcwd(), "socket": os.environ["GOGGLES_SOCKET"],
    "out_dir": m["out_dir"], "jit": m["config"]["jit"],
}))
""")
    (exp_dir / "exp.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": "eval",
                "dataset": str(exp_dir / "dataset.yaml"),
                "models": ["model.yaml"],
                "entrypoint": str(entry),
                "seeds": [99],
            }
        )
    )
    source = {p.name: p.read_text() for p in exp_dir.glob("*.yaml")}
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "sweep"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run_hydra",
            "--multirun",
            "exp=probe",
            f"exp_root={tmp_path}",
            "seed=0,1",
            "+dataset_overrides.value=2,4",
            "+model_overrides.config.jit=true",
            f"hydra.sweep.dir={out}",
            "hydra.job.chdir=true",
        ],
        cwd=root,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = []
    sockets = set()
    for job in range(4):
        job_dir = out / str(job)
        results = json.loads((job_dir / "results.json").read_text())
        values.append(results["model.yaml"]["mean_epe"]["mean"])
        context_path = next(
            job_dir.glob("_generated/summaries/**/*.context.json")
        )
        context = json.loads(context_path.read_text())
        assert context["cwd"] == str(root), (
            "Relative config paths need repo cwd"
        )
        assert Path(context["out_dir"]).is_relative_to(job_dir), (
            "Artifacts must be isolated"
        )
        assert context["jit"] is True, "Nested model overrides must reach child"
        sockets.add(context["socket"])
        assert (job_dir / ".hydra/config.yaml").exists(), (
            "Keep resolved Hydra config"
        )
    assert sorted(values) == [2, 3, 4, 5], "Every sweep combination must run"
    assert len(sockets) == 4, (
        "Concurrent jobs must use separate Goggles sockets"
    )
    assert source == {p.name: p.read_text() for p in exp_dir.glob("*.yaml")}, (
        "Do not rewrite source configs"
    )
    assert not (exp_dir / "results.json").exists(), (
        "Source results must stay untouched"
    )


@pytest.mark.parametrize("custom_root", [False, True])
def test_study_training_honors_job_output_root(
    tmp_path, monkeypatch, custom_root
):
    """Training artifacts use the job root while legacy paths stay stable."""
    import argparse
    from contextlib import nullcontext

    from flowgym import run_setup

    monkeypatch.setattr(
        run_setup, "wandb_git_diff_capture", lambda: nullcontext("abc")
    )
    monkeypatch.setattr(
        run_setup, "_attach_wandb_handler", lambda *a, **k: None
    )
    monkeypatch.setattr(
        run_setup, "log_initial_config_artifacts", lambda *a, **k: None
    )
    created = []
    monkeypatch.setattr(
        run_setup, "_create_run_outputs", lambda *a: created.append(a[1])
    )
    estimator = {"estimator": "dummy"}
    if custom_root:
        estimator["out_dir"] = str(tmp_path / "job" / "runs")
    setup = {"config": {"study": {"name": "probe"}}, "run_name": "run"}
    result = run_setup.setup_study_run(
        argparse.Namespace(mode="train-supervised"),
        setup,
        {"seed": 0},
        None,
        estimator,
        None,
    )
    root = (
        tmp_path / "job" / "runs"
        if custom_root
        else Path("experiments/probe/runs")
    )
    expected = str(root / "state=abc" / "run")
    assert result == expected, "Study artifacts must follow the selected root"
    assert created == [expected], "Create outputs only in that directory"
