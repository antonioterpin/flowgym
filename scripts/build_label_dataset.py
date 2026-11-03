"""Build a router training set from image pairs labelled with the best member.

Input is a CSV with one row per pair: ``frame1``, ``frame2`` (image paths,
relative to the CSV), ``label`` (index of the best member in the library)
and optionally ``split`` (``train``, ``val`` or ``test``; without it the
last ``--val-fraction`` of the rows go to val). ``test`` is for a separate
recording that no run selects on, e.g. for the augmentation ablation in
``experiments/art_of_piv/appendix/F.3_selection_agent_training``. It writes

- ``<out>/train/``, ``<out>/val/`` (and ``<out>/test/``): one .mat pair
  per row (``V`` is zeros, there is no ground truth);
- ``<out>/cache/``: one cache per library member whose ``epe`` column is
  **not** an endpoint error: 0 for the labelled member and 1 for the
  others. Classification mode then learns argmin = the label, and every
  frame hits the cache, so training and validation run no sub-estimator:
  val ``mean_error`` is the router's error rate on the labels.

    uv run python scripts/build_label_dataset.py \\
        --labels pairs.csv --library libraries/mine.yaml --out data/mine \\
        [--preprocess deployed_estimator.yaml]

``--preprocess`` applies the deployment preprocessing to the frames before
writing them, so the router trains on what it reads when deployed. It takes
the estimator config the deployment loads (its ``config.preprocess``) or a
file with ``preprocessing_steps``, whichever steps those currently list.

Keep ``<out>/cache`` apart from real EPE caches. The build is written next to
``<out>`` and replaces its splits and its whole ``cache`` (including any router
cache trained on them) only once it succeeds: no pair is left in an old split,
and a failed rebuild keeps the previous dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.io
from PIL import Image

from flowgym.common.preprocess import build_preprocessing_steps
from flowgym.flow.art_of_piv import validation as art_of_piv_validation
from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.training.caching import CacheManager
from flowgym.utils import load_configuration


def cache_ids(library: str) -> list[str]:
    """Sub-cache id of each library member, as ArtOfPIV resolves it.

    The members are built by ArtOfPIV itself, so checkpoint-backed members
    load their ``load_from`` weights and get the ids the router looks up.

    Args:
        library: Path of the library YAML.

    Returns:
        One cache id per member, in library order.
    """
    router = ArtOfPIVEstimator(
        router_config={
            "encoder": {"features_list": [1]},
            "head": {"hidden_dims": [1]},
        },
        estimators_list_path=library,
        image_shape=(1, 64, 64),
    )
    base = art_of_piv_validation.validate_and_load_estimators_config(
        library
    ).get("cache_id_base")
    return [
        f"{base or type(model).__name__}{model.get_cache_id_suffix(state)}"
        for model, state in zip(
            router.sub_models, router.sub_model_states, strict=True
        )
    ]


def preprocessing_steps(path: str) -> list[dict]:
    """Preprocessing steps of an estimator config or a steps file.

    Args:
        path: An estimator YAML (steps under ``config.preprocess``, as a list
            or the path of a steps file) or a YAML with ``preprocessing_steps``.

    Returns:
        The steps, in order.

    Raises:
        ValueError: If the file lists no preprocessing steps.
    """
    cfg = load_configuration(path) or {}
    steps = cfg.get("preprocessing_steps")
    if steps is None:
        steps = (cfg.get("config") or {}).get("preprocess")
        if isinstance(steps, str):
            return preprocessing_steps(steps)
    if not steps:
        raise ValueError(f"{path}: no preprocessing steps found")
    return list(steps)


def load_pair(frame1: Path, frame2: Path) -> np.ndarray:
    """Load the two frames of a pair.

    Args:
        frame1: First frame.
        frame2: Second frame.

    Returns:
        The pair, of shape (2, H, W).

    Raises:
        ValueError: If the frames are not 2D or differ in shape.
    """
    i0, i1 = (np.asarray(Image.open(f)) for f in (frame1, frame2))
    if i0.ndim != 2 or i0.shape != i1.shape:
        raise ValueError(
            f"{frame1}, {frame2}: need two grayscale frames of one shape, "
            f"got {i0.shape} and {i1.shape}"
        )
    return np.stack([i0, i1])


def write_pair(path: Path, pair: np.ndarray) -> tuple[int, ...]:
    """Write one pair as a compressed .mat with a zero flow field.

    Args:
        path: Output file.
        pair: The two frames, of shape (2, H, W).

    Returns:
        The frame shape (H, W).
    """
    i0, i1 = pair
    scipy.io.savemat(
        path,
        {"I0": i0, "I1": i1, "V": np.zeros((*i0.shape, 2), np.float32)},
        do_compression=True,
    )
    return i0.shape


def main() -> None:
    """Write the .mat pairs and the label caches.

    Raises:
        ValueError: If a label is outside the library, the val fraction is
            outside [0, 1), a split is unknown or train is empty.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--preprocess")
    args = ap.parse_args()

    with args.labels.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = cache_ids(args.library)
    labels = np.array([int(r["label"]) for r in rows])
    if labels.min() < 0 or labels.max() >= len(ids):
        raise ValueError(f"labels must be in [0, {len(ids)}) for the library")

    splits = assign_splits(rows, args.val_fraction)
    staging = args.out.parent / f".{args.out.name}.build"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        build(args, rows, ids, labels, splits, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def assign_splits(rows: list[dict[str, str]], val_fraction: float) -> list[str]:
    """Give each row its split, before anything is written.

    Args:
        rows: The rows of the labels CSV.
        val_fraction: Share of the last rows that go to val when a row has no
            ``split``.

    Returns:
        The split of each row.

    Raises:
        ValueError: If the fraction is outside [0, 1), a split is unknown or
            the train split ends up empty.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError(
            f"--val-fraction must be in [0, 1), got {val_fraction}"
        )
    n_val = round(len(rows) * val_fraction)
    splits = []
    for i, r in enumerate(rows):
        split = r.get("split") or ("val" if i >= len(rows) - n_val else "train")
        if split not in ("train", "val", "test"):
            raise ValueError(f"row {i}: unknown split {split!r}")
        splits.append(split)
    if "train" not in splits:
        raise ValueError("no row is in the train split")
    return splits


def build(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    ids: list[str],
    labels: np.ndarray,
    splits: list[str],
    staging: Path,
) -> None:
    """Write the dataset into ``staging``, then swap it into ``args.out``.

    Args:
        args: The parsed command line.
        rows: The rows of the labels CSV.
        ids: The library members' cache ids.
        labels: The label of each row.
        splits: The split of each row.
        staging: Empty folder to build in.

    Raises:
        ValueError: If the frames differ in shape.
    """
    # The cache key of a file is the md5 of its basename, so names must be
    # unique across splits: prefix the row index.
    root, files, frames1, frames2 = args.labels.parent, [], [], []
    for i, (r, split) in enumerate(zip(rows, splits, strict=True)):
        path = staging / split / f"{i:06d}_{Path(r['frame1']).stem}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        files.append(path)
        frames1.append(root / r["frame1"])
        frames2.append(root / r["frame2"])
    steps = build_preprocessing_steps(
        preprocessing_steps(args.preprocess) if args.preprocess else []
    )
    # Preprocess here, in chunks, and only compress in the pool: compressing
    # each .mat takes ~100 ms on large frames.
    pool = (
        ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")
        )
        if args.workers > 1
        else None
    )
    shapes: set[tuple[int, ...]] = set()
    chunk = 64
    for i in range(0, len(files), chunk):
        pairs = np.stack(
            [
                load_pair(f1, f2)
                for f1, f2 in zip(
                    frames1[i : i + chunk], frames2[i : i + chunk], strict=True
                )
            ]
        )
        if steps:
            n = pairs.shape[0]
            x = jnp.asarray(pairs.reshape(-1, *pairs.shape[2:]), jnp.float32)
            for step in steps:
                x = step(x)
            pairs = np.asarray(x, np.float32).reshape(n, 2, *x.shape[1:])
        written = (pool.map if pool else map)(
            write_pair, files[i : i + chunk], pairs
        )
        shapes.update(written)
    if pool:
        pool.shutdown()
    if len(shapes) > 1:
        raise ValueError(f"frames differ in shape: {sorted(shapes)}")

    # Same keys as compute_batch_keys gives a file batch: md5 of the basename.
    keys = np.array(
        [int(hashlib.md5(f.name.encode()).hexdigest()[:16], 16) for f in files],
        dtype=np.uint64,
    )
    for member, cache_id in enumerate(ids):
        cm = CacheManager(
            root_dir=str(staging / "cache"),
            cache_id=cache_id,
            spec={"epe": (np.dtype("float32"), ())},
            warm_start="none",
        )
        cm.write(keys, {"epe": (labels != member).astype(np.float32)})
        cm.close()

    # Swap in only now. Whole folders go, so pairs that moved split and any
    # router cache of the old data go with them.
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test", "cache"):
        shutil.rmtree(args.out / name, ignore_errors=True)
        if (staging / name).exists():
            (staging / name).rename(args.out / name)
    for split in ("train", "val", "test"):
        n = sum(f.parent.name == split for f in files)
        if split == "test" and not n:
            continue
        print(f"{split}: {n} pairs in {args.out / split}")
    print(f"label counts: {np.bincount(labels, minlength=len(ids)).tolist()}")
    print(f"image_shape: {list(shapes.pop())}")


if __name__ == "__main__":
    main()
