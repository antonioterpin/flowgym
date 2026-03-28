#!/usr/bin/env python3
"""Run DIS-only experiment 3.1 profiles: learned vs oracle vs none.

This launcher prepares and runs three downstream-evaluation profiles:
- learned: DIS + learned outlier rejection (flow-only, no image pair)
- oracle:  DIS + true oracle mask
- none:    DIS with no outlier rejection

Each profile/objective pair writes to a dedicated CSV so results are directly
comparable.

Example:
    uv run python experiments/piv-admm/experiment3_1/run_dis_comparison.py
"""

from __future__ import annotations

import argparse
import copy
import subprocess
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path("experiments/piv-admm/experiment3_1/base_model.yaml")
DATASET_PATH = Path("experiments/piv-admm/experiment3_1/dataset.yaml")
BASE_CONFIG_FOLDER = "src/flowgym/config/estimators/flow/piv_dataset/consensus"
DIS_LIST_PATH = Path(f"{BASE_CONFIG_FOLDER}/dis_list.yaml")
LEARNED_ROOT = Path("results/training_oracle/imgpair_0")

DIS_BASELINE_PERFORMANCE = {
    "min_epe": 0.08859,
    "max_epe": 0.24117,
    "mean_epe": 0.15316,
    "min_relative_epe": 0.11474,
    "mean_relative_epe": 12.70115,
    "max_relative_epe": 44.74119,
}

OBJECTIVE_SETTINGS = {
    "l1": {
        "regularizer_weights": {
            "smoothness": 150.0,
            "laplacian": 250.0,
            "divergence": 1100.0,
        },
        "flows_objective_type": "l1",
        "solver_flows": "closed_form_l1",
    },
    "huber": {
        "regularizer_weights": {
            "smoothness": 3.0,
            "laplacian": 30.0,
            "divergence": 1300.0,
        },
        "flows_objective_type": "huber",
        "solver_flows": "closed_form_huber",
    },
}

RUN_CMD = [
    "uv",
    "run",
    "python",
    "-m",
    "src.main",
    "--model",
    str(CONFIG_PATH),
    "--dataset",
    str(DATASET_PATH),
    "--mode",
    "eval",
]


def _set_nested_value(
    d: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _tau_tag(tau: float) -> str:
    return str(tau).replace(".", "_")


def _checkpoint_has_saved_steps(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.exists() and any(
        entry.is_dir() and entry.name.isdigit()
        for entry in checkpoint_dir.iterdir()
    )


def _inject_dis_learned_postprocess(
    estimators_cfg: dict[str, Any],
    checkpoint_dir: Path,
    mask_threshold: float,
    features: list[int],
) -> dict[str, Any]:
    out = copy.deepcopy(estimators_cfg)
    estimators = out.get("estimators", [])
    if not isinstance(estimators, list):
        raise ValueError("DIS estimators config must contain a list.")

    dis_count = sum(
        1 for estimator in estimators if estimator.get("estimator") == "dis_jax"
    )
    dis_idx = 0
    for estimator_cfg in estimators:
        if estimator_cfg.get("estimator") != "dis_jax":
            continue
        est_cfg = estimator_cfg.setdefault("config", {})
        if not isinstance(est_cfg, dict):
            raise ValueError("Estimator config must be a dict.")
        learned_step = {
            "name": "learned_oracle_threshold",
            "threshold_value": mask_threshold,
            "load_from": str(checkpoint_dir),
            "features": features,
            "estimator_index": dis_idx,
            "estimator_count": dis_count,
            "include_image_pair": False,
        }
        existing_postprocess = est_cfg.get("postprocess", [])
        if isinstance(existing_postprocess, list):
            est_cfg["postprocess"] = [learned_step, *existing_postprocess]
        else:
            est_cfg["postprocess"] = [learned_step]
        dis_idx += 1
    return out


def _parse_csv_floats(raw: str) -> list[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return [float(v) for v in values]


def _parse_csv_strings(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=str,
        default="learned,oracle,none",
        help="Comma-separated profiles: learned,oracle,none.",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="huber",
        help="Comma-separated objectives: l1,huber.",
    )
    parser.add_argument(
        "--taus",
        type=str,
        default="0.1,0.3,0.5,0.7,1.0",
        help="Comma-separated epe_limit values.",
    )
    parser.add_argument(
        "--trained-taus",
        type=str,
        default="0.1,0.3,0.5,0.7,1.0",
        help="Learned checkpoint taus expected to exist.",
    )
    parser.add_argument(
        "--learned-root",
        type=Path,
        default=LEARNED_ROOT,
        help="Root containing tau_<x>/.../LearnedOracleThresholdEstimator.",
    )
    parser.add_argument(
        "--dis-list-path",
        type=Path,
        default=DIS_LIST_PATH,
        help="Base DIS estimator list path.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Learned-oracle inlier threshold.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="16,32",
        help="Learned-oracle CNN features as comma-separated ints.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/experiment3_1_dis_compare"),
        help="CSV prefix. Final file: <prefix>_<objective>_<profile>.csv",
    )
    parser.add_argument(
        "--strict-learned",
        action="store_true",
        help="Fail if learned profile has missing tau checkpoints.",
    )
    parser.add_argument(
        "--keep-existing-logs",
        action="store_true",
        help="Append to existing CSVs instead of deleting them first.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="1",
    )
    parser.add_argument(
        "--goggles-port",
        type=str,
        default="2402",
    )
    args = parser.parse_args()

    profiles = _parse_csv_strings(args.profiles)
    objectives = _parse_csv_strings(args.objectives)
    taus = _parse_csv_floats(args.taus)
    trained_taus = set(_parse_csv_floats(args.trained_taus))
    features = [int(v) for v in _parse_csv_strings(args.features)]

    valid_profiles = {"learned", "oracle", "none"}
    if any(profile not in valid_profiles for profile in profiles):
        raise ValueError(f"Invalid profiles: {profiles}.")
    if any(objective not in OBJECTIVE_SETTINGS for objective in objectives):
        raise ValueError(f"Invalid objectives: {objectives}.")
    if not features or any(feature <= 0 for feature in features):
        raise ValueError("`features` must contain positive integers.")

    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    with open(args.dis_list_path) as f:
        base_dis_estimators_cfg = yaml.safe_load(f)

    temp_dir = CONFIG_PATH.parent
    temp_model_paths: list[Path] = []
    temp_estimators_paths: list[Path] = []

    try:
        for profile in profiles:
            for objective in objectives:
                log_path = Path(
                    f"{args.output_prefix}_{objective}_{profile}.csv"
                )
                if not args.keep_existing_logs:
                    log_path.unlink(missing_ok=True)

                obj_cfg = OBJECTIVE_SETTINGS[objective]
                for tau in taus:
                    cfg = copy.deepcopy(base_cfg)
                    _set_nested_value(cfg, "config.oracle", profile == "oracle")
                    # ``epe_limit`` drives oracle masking in consensus.
                    # Keep it only for the true-oracle profile.
                    _set_nested_value(
                        cfg,
                        "config.experiment_params.epe_limit",
                        tau if profile == "oracle" else None,
                    )
                    _set_nested_value(
                        cfg,
                        "config.experiment_params.baseline_performance",
                        DIS_BASELINE_PERFORMANCE,
                    )
                    _set_nested_value(
                        cfg,
                        "config.experiment_params.log_path",
                        str(log_path),
                    )
                    _set_nested_value(
                        cfg,
                        "config.experiment_params.comparison_profile",
                        profile,
                    )
                    _set_nested_value(
                        cfg,
                        "config.experiment_params.comparison_tau",
                        tau,
                    )
                    _set_nested_value(
                        cfg,
                        "config.experiment_params.comparison_checkpoint",
                        None,
                    )
                    _set_nested_value(
                        cfg,
                        "config.consensus_config.regularizer_weights",
                        obj_cfg["regularizer_weights"],
                    )
                    _set_nested_value(
                        cfg,
                        "config.consensus_config.flows_objective_type",
                        obj_cfg["flows_objective_type"],
                    )
                    _set_nested_value(
                        cfg,
                        "config.consensus_config.solver_flows",
                        obj_cfg["solver_flows"],
                    )
                    _set_nested_value(
                        cfg,
                        "config.consensus_config.weights_type",
                        "photograd",
                    )

                    estimators_path = str(args.dis_list_path)
                    if profile == "learned":
                        checkpoint_dir = (
                            args.learned_root
                            / f"tau_{_tau_tag(tau)}"
                            / "learned_oracle_threshold"
                            / "0"
                            / "checkpoints"
                            / "LearnedOracleThresholdEstimator"
                        )
                        has_ckpt = _checkpoint_has_saved_steps(checkpoint_dir)
                        if args.strict_learned and not has_ckpt:
                            raise FileNotFoundError(
                                "Missing learned checkpoint for tau="
                                f"{tau}: {checkpoint_dir}"
                            )
                        if has_ckpt and tau in trained_taus:
                            dis_estimators_cfg = (
                                _inject_dis_learned_postprocess(
                                    base_dis_estimators_cfg,
                                    checkpoint_dir=checkpoint_dir,
                                    mask_threshold=args.mask_threshold,
                                    features=features,
                                )
                            )
                            temp_estimators = (
                                temp_dir
                                / (
                                    "temp_dis_learned_"
                                    f"{profile}_{objective}_{_tau_tag(tau)}.yaml"
                                )
                            )
                            with open(temp_estimators, "w") as f:
                                yaml.safe_dump(dis_estimators_cfg, f)
                            temp_estimators_paths.append(temp_estimators)
                            estimators_path = str(temp_estimators)
                            _set_nested_value(
                                cfg,
                                "config.experiment_params.comparison_checkpoint",
                                str(checkpoint_dir),
                            )
                        else:
                            print(
                                "Skipping learned injection for tau="
                                f"{tau}; no checkpoint or tau not in "
                                f"trained set {sorted(trained_taus)}."
                            )

                    _set_nested_value(
                        cfg,
                        "config.estimators_list_path",
                        estimators_path,
                    )

                    temp_model = (
                        temp_dir
                        / f"temp_dis_compare_{profile}_{objective}_{tau}.yaml"
                    )
                    with open(temp_model, "w") as f:
                        yaml.safe_dump(cfg, f)
                    temp_model_paths.append(temp_model)

                    cmd = RUN_CMD.copy()
                    cmd[cmd.index(str(CONFIG_PATH))] = str(temp_model)
                    print(
                        "Running profile="
                        f"{profile}, objective={objective}, tau={tau} "
                        f"-> {log_path}"
                    )
                    subprocess.run(
                        [
                            "bash",
                            "-c",
                            f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} "
                            f"GOGGLES_PORT={args.goggles_port} "
                            + " ".join(cmd),
                        ],
                        check=True,
                    )
    finally:
        for path in temp_model_paths:
            path.unlink(missing_ok=True)
        for path in temp_estimators_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
