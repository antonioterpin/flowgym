"""Tests for the label dataset builder script."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.io
import yaml
from PIL import Image

from flowgym.common.preprocess import crop_special, stretch_contrast
from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "build_label_dataset.py"
_EXP = "experiments/real_labels"

_spec = importlib.util.spec_from_file_location("build_label_dataset", _SCRIPT)
assert _spec is not None and _spec.loader is not None
build_label_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_label_dataset)


def test_router_reads_the_example_labels_from_the_built_caches(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(_REPO_ROOT)
    pairs = Path(_EXP, "example", "pairs.csv")
    with pairs.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_label_dataset.py",
            *("--labels", str(pairs), "--library", f"{_EXP}/library.yaml"),
            *("--out", str(out), "--workers", "1"),
        ],
    )

    build_label_dataset.main()

    files = sorted(out.glob("*/*.mat"))
    assert [f.parent.name for f in files] == [r["split"] for r in rows]
    router = ArtOfPIVEstimator(
        router_config={
            "encoder": {"features_list": [4]},
            "head": {"hidden_dims": [4]},
        },
        estimators_list_path=f"{_EXP}/library.yaml",
        image_shape=(1, 64, 64),
    )
    router.preload_caches(str(out / "cache"))
    for f, row in zip(files, rows, strict=True):
        key = int(hashlib.md5(f.name.encode()).hexdigest()[:16], 16)
        costs, mask = router._unified_epe_index[key]
        assert mask.all()
        assert int(np.argmin(costs)) == int(row["label"])


@pytest.mark.parametrize("form", ["estimator", "steps"])
def test_preprocess_applies_the_deployment_steps(tmp_path, monkeypatch, form):
    monkeypatch.chdir(_REPO_ROOT)
    pairs = Path(_EXP, "example", "pairs.csv")
    steps = [
        {
            "name": "crop_special",
            "target_height": 32,
            "target_width": 32,
            "fraction_v": 0.5,
            "fraction_h": 0.5,
        },
        {"name": "stretch_contrast", "scale": 255.0},
    ]
    cfg = (
        {"estimator": "art_of_piv", "config": {"preprocess": steps}}
        if form == "estimator"
        else {"preprocessing_steps": steps}
    )
    cfg_path = tmp_path / "deploy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_label_dataset.py",
            *("--labels", str(pairs), "--library", f"{_EXP}/library.yaml"),
            *("--out", str(out), "--workers", "1"),
            *("--preprocess", str(cfg_path)),
        ],
    )

    build_label_dataset.main()

    mat = scipy.io.loadmat(sorted(out.glob("train/*.mat"))[0])
    frame = np.asarray(
        Image.open(pairs.parent / "frames" / "pair0_1.png"), np.float32
    )
    expected = np.asarray(
        stretch_contrast(
            crop_special(jnp.asarray(frame[None]), 32, 32, 0.5, 0.5), 255.0
        )
    )[0]
    assert mat["I0"].shape == (32, 32)
    np.testing.assert_allclose(mat["I0"], expected, rtol=1e-5, atol=1e-3)
    assert mat["V"].shape == (32, 32, 2)


def test_test_split_rows_go_to_their_own_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(_REPO_ROOT)
    frames = (_REPO_ROOT / _EXP / "example" / "frames").resolve()
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "frame1,frame2,label,split\n"
        + "".join(
            f"{frames}/pair{i}_1.png,{frames}/pair{i}_2.png,{i},{split}\n"
            for i, split in ((0, "train"), (1, "test"))
        )
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_label_dataset.py",
            *("--labels", str(pairs), "--library", f"{_EXP}/library.yaml"),
            *("--out", str(out), "--workers", "1"),
        ],
    )

    build_label_dataset.main()

    assert [f.parent.name for f in sorted(out.glob("*/*.mat"))] == [
        "test",
        "train",
    ]


def test_router_requiring_cached_labels_fails_on_a_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(_REPO_ROOT)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_label_dataset.py",
            *("--labels", f"{_EXP}/example/pairs.csv"),
            *("--library", f"{_EXP}/library.yaml"),
            *("--out", str(out), "--workers", "1"),
        ],
    )
    build_label_dataset.main()
    router = ArtOfPIVEstimator(
        router_config={
            "encoder": {"features_list": [4]},
            "head": {"hidden_dims": [4]},
        },
        estimators_list_path=f"{_EXP}/library.yaml",
        image_shape=(1, 64, 64),
        require_cached_labels=True,
    )
    router.preload_caches(str(out / "cache"))

    def batch(*names: str) -> SimpleNamespace:
        return SimpleNamespace(
            files=list(names),
            params=None,
            flow_fields=np.zeros((len(names), 64, 64, 2)),
        )

    built = sorted(out.glob("train/*.mat"))[0].name
    assert router.enrich(batch(built), np.array([0]))["epe_mask"].all()
    # A padding entry of a partial batch is not among the misses to fill
    padded = router.enrich(batch(built, "padding.mat"), np.array([0]))
    assert padded["epe_mask"].all()
    with pytest.raises(ValueError, match="not in the label caches"):
        router.enrich(batch("unknown.mat"), np.array([0]))


def test_rebuild_removes_the_previous_build(tmp_path, monkeypatch):
    monkeypatch.chdir(_REPO_ROOT)
    frames = (_REPO_ROOT / _EXP / "example" / "frames").resolve()
    pairs, out = tmp_path / "pairs.csv", tmp_path / "out"

    def build(splits: tuple[str, str]) -> None:
        pairs.write_text(
            "frame1,frame2,label,split\n"
            + "".join(
                f"{frames}/pair{i}_1.png,{frames}/pair{i}_2.png,{i},{split}\n"
                for i, split in enumerate(splits)
            )
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_label_dataset.py",
                *("--labels", str(pairs), "--library", f"{_EXP}/library.yaml"),
                *("--out", str(out), "--workers", "1"),
            ],
        )
        build_label_dataset.main()

    build(("train", "val"))
    (out / "cache" / "art_of_piv").mkdir()  # a router's aggregate cache
    build(("train", "train"))

    assert sorted(f.parent.name for f in out.glob("*/*.mat")) == ["train"] * 2
    assert not (out / "cache" / "art_of_piv").exists()

    # A failed rebuild keeps the previous dataset
    with pytest.raises(ValueError, match="unknown split"):
        build(("train", "typo"))
    assert sorted(f.parent.name for f in out.glob("*/*.mat")) == ["train"] * 2
    assert len(list((out / "cache").iterdir())) == 2


@pytest.mark.parametrize("fraction", [-0.1, 1, 2])
def test_invalid_val_fraction_is_rejected_before_writing(fraction):
    rows = [{"frame1": "a", "frame2": "b"}] * 4
    with pytest.raises(ValueError, match="--val-fraction"):
        build_label_dataset.assign_splits(rows, fraction)
