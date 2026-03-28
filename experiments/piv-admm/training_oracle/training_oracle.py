"""Train the learned oracle threshold estimator on PIV dataset class1.

This launches `src.main` in `train-supervised` mode, which internally calls
`src/train_supervised.py`.

Training split: `.../train`
Validation split: `.../val`

To run:
    uv run python experiments/piv-admm/training_oracle/training_oracle.py
"""

import argparse
import copy
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

# === USER SETTINGS ===
MODEL_PATH = Path("experiments/piv-admm/training_oracle/base_model.yaml")
DATASET_PATH = Path("experiments/piv-admm/training_oracle/dataset.yaml")
# Oracle-threshold axis used in previous PIV-ADMM experiments.
ORACLE_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 1.0]
DIS_ESTIMATORS_LIST_PATH = (
    "src/flowgym/config/estimators/flow/piv_dataset/consensus/dis_list.yaml"
)
FARNEBACK_ESTIMATORS_LIST_PATH = (
    "src/flowgym/config/estimators/flow/piv_dataset/consensus/"
    "farneback_list.yaml"
)
ESTIMATOR_LISTS = {
    "dis": DIS_ESTIMATORS_LIST_PATH,
    "farneback": FARNEBACK_ESTIMATORS_LIST_PATH,
}
# ======================

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(MODEL_PATH),
    "--dataset",
    str(DATASET_PATH),
    "--mode",
    "train-supervised",
]


def set_nested_value(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dotted path."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _tau_tag(tau: float) -> str:
    return str(tau).replace(".", "_")


def _parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "2", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{text}'. Use true/false."
    )


def _parse_taus(raw: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if len(values) == 0:
        raise argparse.ArgumentTypeError(
            "Invalid --taus value: expected at least one numeric tau."
        )
    try:
        taus = [float(v) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --taus value '{raw}'. Use comma-separated floats."
        ) from exc
    if any(tau <= 0.0 for tau in taus):
        raise argparse.ArgumentTypeError(
            f"Invalid --taus value '{raw}'. Taus must be > 0."
        )
    return taus


def _count_training_samples(dataset_cfg: dict[str, Any]) -> int:
    file_list = dataset_cfg.get("file_list", [])
    if not isinstance(file_list, list):
        return 0
    total = 0
    for item in file_list:
        path = Path(str(item))
        if path.is_dir():
            total += sum(1 for _ in path.rglob("*.mat"))
        elif path.is_file() and path.suffix == ".mat":
            total += 1
    return total


def _ensure_min_batches_for_epochs(
    dataset_cfg: dict[str, Any],
    min_epochs: int,
) -> None:
    if min_epochs <= 0:
        return
    batch_size = int(dataset_cfg.get("batch_size", 1))
    batch_size = max(batch_size, 1)
    samples = _count_training_samples(dataset_cfg)
    if samples <= 0:
        return
    batches_per_epoch = math.ceil(samples / batch_size)
    min_batches = batches_per_epoch * min_epochs
    current = int(dataset_cfg.get("num_batches", 0))
    dataset_cfg["num_batches"] = max(current, min_batches)


def run_training_oracle(
    cuda_visible_devices: str,
    goggles_port: str,
    include_image_pair: bool,
    min_epochs: int,
    estimator_list_name: str,
    oracle_thresholds: list[float],
) -> None:
    """Run supervised training for all oracle-threshold values."""
    estimator_list_path = ESTIMATOR_LISTS.get(estimator_list_name)
    if estimator_list_path is None:
        raise ValueError(
            "Unknown estimator list "
            f"'{estimator_list_name}'. Valid values: "
            f"{sorted(ESTIMATOR_LISTS)}."
        )

    with open(MODEL_PATH) as f:
        base_model_cfg = yaml.safe_load(f)
    with open(DATASET_PATH) as f:
        dataset_cfg = yaml.safe_load(f)

    run_id = (
        f"imgpair_{int(include_image_pair)}_gpu_"
        f"{cuda_visible_devices}_pid_{os.getpid()}"
    )
    temp_dataset_no_cache_path = (
        DATASET_PATH.parent / f"temp_dataset_no_cache_{run_id}.yaml"
    )
    temp_val_dataset_no_cache_path = (
        DATASET_PATH.parent / f"temp_val_dataset_no_cache_{run_id}.yaml"
    )

    # Force-disable caching for this experiment.
    dataset_cfg.pop("caching", None)
    _ensure_min_batches_for_epochs(dataset_cfg, min_epochs=min_epochs)
    validation_cfg = dataset_cfg.get("validation", None)
    if isinstance(validation_cfg, dict):
        val_dataset_spec = validation_cfg.get("dataset")
        if isinstance(val_dataset_spec, str):
            with open(val_dataset_spec) as f:
                val_dataset_cfg = yaml.safe_load(f)
            val_dataset_cfg.pop("caching", None)
            with open(temp_val_dataset_no_cache_path, "w") as f:
                yaml.safe_dump(val_dataset_cfg, f)
            validation_cfg["dataset"] = str(temp_val_dataset_no_cache_path)
        elif isinstance(val_dataset_spec, dict):
            val_dataset_spec.pop("caching", None)

    with open(temp_dataset_no_cache_path, "w") as f:
        yaml.safe_dump(dataset_cfg, f)

    try:
        for tau in oracle_thresholds:
            tag = _tau_tag(tau)
            cfg = copy.deepcopy(base_model_cfg)
            set_nested_value(cfg, "config.oracle_epe_threshold", tau)
            set_nested_value(
                cfg, "config.include_image_pair", include_image_pair
            )
            # Keep training aligned with downstream consensus estimator set.
            set_nested_value(
                cfg, "config.estimator_list", estimator_list_path
            )
            if estimator_list_name != "dis":
                set_nested_value(
                    cfg,
                    "out_dir",
                    "results/training_oracle/"
                    f"{estimator_list_name}/imgpair_"
                    f"{int(include_image_pair)}/tau_{tag}",
                )
                set_nested_value(
                    cfg,
                    "run_name",
                    "piv_admm_training_oracle_"
                    f"{estimator_list_name}_imgpair_"
                    f"{int(include_image_pair)}_tau_{tag}",
                )
            else:
                set_nested_value(
                    cfg,
                    "out_dir",
                    f"results/training_oracle/imgpair_{int(include_image_pair)}/tau_{tag}",
                )
                set_nested_value(
                    cfg,
                    "run_name",
                    f"piv_admm_training_oracle_imgpair_{int(include_image_pair)}_tau_{tag}",
                )

            temp_model_path = (
                MODEL_PATH.parent
                / f"temp_model_oracle_threshold_{tag}_{run_id}.yaml"
            )
            with open(temp_model_path, "w") as f:
                yaml.safe_dump(cfg, f)

            cmd = RUN_CMD.copy()
            cmd[cmd.index(str(MODEL_PATH))] = str(temp_model_path)
            cmd[cmd.index(str(DATASET_PATH))] = str(temp_dataset_no_cache_path)

            print(
                "=== Running "
                f"oracle_epe_threshold={tau}, "
                f"estimator_list={estimator_list_name}, "
                f"include_image_pair={include_image_pair}, "
                f"gpu={cuda_visible_devices} ==="
            )
            subprocess.run(
                [
                    "bash",
                    "-c",
                    f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
                    + f"GOGGLES_PORT={goggles_port} "
                    + " ".join(cmd),
                ],
                check=True,
            )

            temp_model_path.unlink(missing_ok=True)
    finally:
        temp_dataset_no_cache_path.unlink(missing_ok=True)
        temp_val_dataset_no_cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--goggles-port", type=str, default="2406")
    parser.add_argument(
        "--include-image-pair",
        type=_parse_bool,
        default=False,
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=10,
        help=(
            "Ensure dataset num_batches corresponds to at least this many "
            "epochs (based on train .mat count and batch_size)."
        ),
    )
    parser.add_argument(
        "--estimator-list",
        type=str,
        default="dis",
        choices=sorted(ESTIMATOR_LISTS.keys()),
        help=(
            "Sub-estimator combination used to generate candidate flows "
            "for training labels."
        ),
    )
    parser.add_argument(
        "--taus",
        type=_parse_taus,
        default=ORACLE_THRESHOLDS.copy(),
        help=(
            "Comma-separated oracle EPE thresholds. "
            "Example: --taus 0.5 or --taus 0.1,0.3,1.0"
        ),
    )
    args = parser.parse_args()
    run_training_oracle(
        cuda_visible_devices=args.gpu,
        goggles_port=args.goggles_port,
        include_image_pair=args.include_image_pair,
        min_epochs=args.min_epochs,
        estimator_list_name=args.estimator_list,
        oracle_thresholds=args.taus,
    )
