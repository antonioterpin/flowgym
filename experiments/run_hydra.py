"""Optional Hydra entrypoint for composed experiment sweeps.

Run from the repository root with ``uv run --extra experiments python -m
experiments.run_hydra exp=example_experiment --multirun seed=0,1``.
"""

from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from experiments.run import run_experiment


@hydra.main(version_base="1.3", config_path="hydra_conf", config_name="config")
def main(config: DictConfig) -> None:
    """Launch one resolved Hydra job through the existing experiment runner.

    Args:
        config: Composed experiment selection and nested config overrides.
    """
    cfg = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    spec_overrides = dict(cfg["experiment_overrides"])
    if cfg["seed"] is not None:
        spec_overrides["seeds"] = [int(cfg["seed"])]
    run_experiment(
        cfg["exp"],
        cfg["exp_root"],
        selected_models=cfg["models"],
        debug=cfg["debug"],
        spec_overrides=spec_overrides,
        model_overrides=cfg["model_overrides"],
        dataset_overrides=cfg["dataset_overrides"],
        output_dir=Path(HydraConfig.get().runtime.output_dir),
        project_root=Path(get_original_cwd()),
    )


if __name__ == "__main__":
    main()
