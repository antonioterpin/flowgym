"""End-to-end demo of the sweep cache / timing / ensemble-selection scripts.

Shows the three estimator-agnostic sweep tools working together on a small,
fully synthetic dataset (no downloads; runs on CPU):

1. ``scripts/collect_cache.py``   -- fill the per-image **error** cache
   (device-independent; reusable across machines).
2. ``scripts/collect_timing.py``  -- write per-config ``timing.json``
   (device-dependent inference time) into the *same* cache dirs.
3. ``scripts/select_ensemble.py`` -- pick a size-K subset minimizing the
   per-image best error, optionally under a latency bound, and export the
   chosen configs as a collect-ready ``estimators_list`` YAML.

The three are inter-compatible by construction: each config's error parquet
and its ``timing.json`` land in the same ``<cache-root>/<cache_id>/`` dir
(``cache_id`` = base + the estimator's ``get_cache_id_suffix``), and
``select_ensemble`` reads both from there.

Run:

    uv run python examples/14_cache_sweep_ensemble.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from example_support import (
    REPO_ROOT,
    make_flow_files,
    make_synthetic_dataset_config,
    write_yaml,
)

# Three dis_jax configs differing only in patch_size -> three distinct
# cache_ids (the config hash differs), so they are separate sweep candidates.
PATCH_SIZES = (7, 9, 11)


def dis_config(patch_size: int) -> dict:
    """Minimal dis_jax model config for a given patch size.

    Args:
        patch_size: DIS interrogation patch size (also used as the stride).

    Returns:
        A model config dict ready to serialize as an estimator YAML.
    """
    return {
        "estimator": "dis_jax",
        "estimate_type": "flow",
        "config": {
            "jit": True,
            "preset": 1,
            "patch_size": patch_size,
            "patch_stride": patch_size,
            "grad_desc_iters": 4,
            "var_refine_iters": 0,
            "use_mean_normalization": False,
            "use_spatial_propagation": False,
            "use_temporal_propagation": False,
            "start_level": 0,
            "levels": 1,
            "level_steps": 1,
            "output_full_res": True,
        },
    }


def run_script(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    """Run one of the sweep scripts with the current interpreter (CPU env).

    Args:
        script: Script filename under ``scripts/``.
        *args: Command-line arguments (stringified).

    Returns:
        The completed subprocess result.

    Raises:
        RuntimeError: If the script exits non-zero.
    """
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script)]
    cmd += [str(a) for a in args]
    print("$ python", f"scripts/{script}", *(str(a) for a in args))
    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_API_KEY": "example",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False, env=env
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{script} failed (exit {result.returncode}).")
    return result


def main() -> int:
    """Run the full collect -> time -> select workflow on synthetic data.

    Returns:
        Process exit code (0 on success).
    """
    with tempfile.TemporaryDirectory(prefix="sweep_demo_") as raw_tmp:
        tmp = Path(raw_tmp)

        # 1. A tiny synthetic dataset (synthpix generates the particle images
        #    from these flow fields) with a caching block declaring the spec.
        flows = make_flow_files(tmp / "flows", num_files=4)
        dataset = make_synthetic_dataset_config(
            file_list=flows, batch_size=2, num_batches=2
        )
        cache_root = tmp / "caches"
        dataset["caching"] = {
            "root_dir": str(cache_root),
            "cache_id": "demo-sweep",
            "spec": {
                "epe": ["float32", []],
                "relative_epe": ["float32", []],
            },
            "warm_start": "index",
        }
        dataset_path = write_yaml(tmp / "dataset.yaml", dataset)

        # 2. The sweep: one estimator YAML per candidate config.
        model_paths = [
            write_yaml(tmp / f"dis_p{ps}.yaml", dis_config(ps))
            for ps in PATCH_SIZES
        ]

        print("\n=== 1) collect_cache: per-image error caches ===")
        run_script(
            "collect_cache.py",
            "--models",
            *model_paths,
            "--dataset",
            dataset_path,
            "--cache-root",
            cache_root,
            "--runner",
            sys.executable,
        )

        print("\n=== 2) collect_timing: per-config timing.json ===")
        run_script(
            "collect_timing.py",
            "--models",
            *model_paths,
            "--dataset",
            dataset_path,
            "--cache-root",
            cache_root,
            "--iters",
            20,
            "--repeat",
            3,
        )

        # Each candidate dir now holds BOTH the error parquet and timing.json.
        print("\n=== cache layout (error + timing co-located per config) ===")
        for cand in sorted(p for p in cache_root.iterdir() if p.is_dir()):
            data_dir = cand / "data"
            has_parquet = data_dir.is_dir() and any(
                data_dir.glob("part-*.parquet")
            )
            has_timing = (cand / "timing.json").is_file()
            print(f"  {cand.name}: error={has_parquet} timing={has_timing}")

        print("\n=== 3) select_ensemble: pick a K=2 subset ===")
        chosen = tmp / "chosen_estimators.yaml"
        result = run_script(
            "select_ensemble.py",
            "--cache-root",
            cache_root,
            "--K",
            2,
            "--metric",
            "epe",
            "--export-models",
            chosen,
            "--export-estimator",
            "dis_jax",
        )
        print(result.stdout.strip())

        print(f"\nExported chosen subset -> {chosen.name}:")
        print(chosen.read_text(encoding="utf-8"))
        print(
            "Feed that file straight back into a full-set run with\n"
            "  collect_cache.py --estimators-list chosen_estimators.yaml ..."
        )

    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
