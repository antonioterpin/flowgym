"""Tests for the per-run output directory of non-study training runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from flowgym import run_setup


def test_train_out_dir_is_timestamped(tmp_path: Path) -> None:
    """Each run gets its own timestamped dir under ``<estimator>/<seed>``."""
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        f"estimator: dummy\nestimate_type: flow\nout_dir: {tmp_path}\n"
        "config: {}\n",
        encoding="utf-8",
    )
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("seed: 0\n", encoding="utf-8")
    args = argparse.Namespace(
        mode="train",
        estimator=str(model_yaml),
        dataset=str(dataset_yaml),
        debug=False,
    )

    out_dir = Path(run_setup.prepare_configs(args)[3])

    assert out_dir.parent == tmp_path / "dummy" / "0"
    datetime.strptime(out_dir.name, "%Y-%m-%d_%H-%M-%S")
