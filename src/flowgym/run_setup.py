"""Per-run setup: config parsing, output directories, W&B wiring."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path

import goggles as gg

from flowgym.experiments import (
    build_study_run_name,
    validate_scalar_tags,
    wandb_git_diff_capture,
    wandb_run_tags,
)
from flowgym.utils import dump_yaml, load_configuration

logger = gg.get_logger("flowgym", with_metrics=True)


def prepare_configs(
    args: argparse.Namespace,
) -> tuple[
    dict,
    dict | None,
    dict,
    str | None,
    dict | None,
    dict | None,
    dict,
]:
    """Prepare dataset and estimator configurations from CLI arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple containing:
            - Dataset configuration dictionary.
            - Second dataset configuration dictionary for comparison,
                if in compare mode.
            - Estimator configuration dictionary.
            - Output directory path if in training mode, else None.
            - Validation settings dictionary, if validation is enabled.
            - Caching configuration dictionary, if caching is enabled.
            - W&B setup dictionary.

    Raises:
        ValueError: If validation configuration is invalid or missing.
    """
    # Load the dataset
    dataset_config = load_configuration(args.dataset)
    dataset_config2 = None
    validation_settings = None
    caching_config = None
    dataset_tags = None

    # Load the estimator
    estimator_config = load_configuration(args.estimator)

    # output directory
    out_dir = None

    if args.mode == "compare-samplers":
        dataset_config["randomize"] = False
        dataset_config["include_images"] = False
    if args.mode not in ["train", "train-supervised"]:
        dataset_config["loop"] = False

    validation_spec = dataset_config.pop("validation", None)
    if validation_spec is not None:
        if not isinstance(validation_spec, dict):
            raise ValueError(
                "`validation` entry must be a dictionary if provided."
            )
        val_dataset_spec = validation_spec.get("dataset")
        if val_dataset_spec is None:
            raise ValueError(
                "`validation.dataset` must be specified when using validation."
            )
        if isinstance(val_dataset_spec, str):
            val_dataset_config = load_configuration(val_dataset_spec)
        elif isinstance(val_dataset_spec, dict):
            val_dataset_config = copy.deepcopy(val_dataset_spec)
        else:
            raise ValueError(
                "`validation.dataset` must be either a path to a YAML file "
                "or a configuration dict."
            )
        if val_dataset_config is None:
            raise ValueError(
                "Validation dataset configuration could not be loaded."
            )
        val_dataset_config.setdefault("loop", False)
        val_dataset_config.setdefault("randomize", False)
        val_dataset_config.setdefault("include_images", False)

        interval = validation_spec.get("interval")
        if interval is not None:
            try:
                interval = int(interval)
            except (TypeError, ValueError):
                raise ValueError(
                    "`validation.interval` must be convertible to an integer."
                ) from None
            if interval <= 0:
                raise ValueError(
                    "`validation.interval` must be a positive integer."
                )

        num_batches = validation_spec.get("num_batches", 1)
        try:
            num_batches = int(num_batches)
        except (TypeError, ValueError):
            raise ValueError(
                "`validation.num_batches` must be convertible to an integer."
            ) from None
        if num_batches <= 0:
            raise ValueError(
                "`validation.num_batches` must be a positive integer."
            )

        validation_settings = {
            "dataset_config": val_dataset_config,
            "interval": interval,
            "num_batches": num_batches,
        }
    # Handle the seed for reproducibility
    if "seed" not in dataset_config or not isinstance(
        dataset_config["seed"], int
    ):
        logger.warning(
            "Dataset configuration does not contain a valid integer seed."
            " Defaulting to 0"
        )
        dataset_config["seed"] = 0

    raw_dataset_tags = dataset_config.pop("tags", None)
    if raw_dataset_tags is not None:
        dataset_tags = validate_scalar_tags(
            raw_dataset_tags, "dataset_config['tags']"
        )

    # Parse Caching Configuration
    if "caching" in dataset_config:
        caching_config = dataset_config["caching"]
        if "spec" in caching_config:
            # Parse spec from list/tuple format to (dtype, shape) tuple
            parsed_spec = {}
            for k, v in caching_config["spec"].items():
                dtype_str = v[0]
                shape = tuple(v[1])
                parsed_spec[k] = (dtype_str, shape)
            caching_config["spec"] = parsed_spec

    if args.mode == "compare-samplers":
        # create a second sampler to load real images from files
        dataset_config2 = load_configuration(args.dataset)
        dataset_config2["include_images"] = True
        dataset_config2["loop"] = False
        dataset_config2["randomize"] = False

    log_estimator_config = {**estimator_config}
    log_estimator_config["config"] = {**estimator_config["config"]}
    for k, v in log_estimator_config["config"].items():
        if isinstance(v, str) and v.endswith(".yaml"):
            log_estimator_config["config"][k] = load_configuration(v)

    group: str | None = None
    study_logged: dict[str, object] | None = None
    study_tags: dict | None = None

    study = estimator_config.get("study")
    if study is not None:
        if not isinstance(study, dict):
            raise ValueError("estimator_config['study'] must be a dict.")
        if "run_name" in estimator_config:
            raise ValueError(
                "Do not set run_name in estimator config for study runs."
            )

        study_name = study.get("name")
        if not isinstance(study_name, str) or not study_name:
            raise ValueError(
                "estimator_config['study']['name'] must be a string."
            )

        study_tags = validate_scalar_tags(
            study.get("tags"), "estimator_config['study']['tags']"
        )
        seed = dataset_config["seed"]
        group = study_name
        estimator_config["run_name"] = build_study_run_name(study_tags, seed)

        # Git state is collected once later in setup_study_run, while
        # wandb_git_diff_capture is active, so the state_label in the
        # W&B config matches the one used for the run output directory.
        study_logged = {
            "name": study_name,
            "tags": dict(study_tags),
            "seed": seed,
        }

    if args.mode in {"train", "train-supervised"} and study_logged is None:
        out_dir = estimator_config.get("out_dir", "output")
        out_dir = os.path.join(
            out_dir, estimator_config["estimator"], str(dataset_config["seed"])
        )
        _create_run_outputs(
            args, out_dir, estimator_config, dataset_config, validation_settings
        )
    # For study training runs, out_dir/configs are created later inside
    # setup_study_run, which derives state_label from the captured git state.

    project = estimator_config.get("project", "FlowGym")
    if not isinstance(project, str) or not project:
        raise ValueError(
            "estimator_config['project'] must be a non-empty string."
        )

    if study_logged is None and estimator_config.get("run_name") is None:
        dataset_stem = Path(args.dataset).stem
        estimator_config["run_name"] = (
            f"{args.mode}_{estimator_config['estimator']}_{dataset_stem}"
        )

    run_tags = wandb_run_tags(study_tags, dataset_tags)
    wandb_setup = {
        "project": project,
        "run_name": estimator_config["run_name"],
        "group": group,
        "config": {
            "estimator_config": log_estimator_config,
            "dataset_config": dataset_config,
            "dataset_config2": dataset_config2,
            "validation": validation_settings,
            "mode": args.mode,
            "out_dir": out_dir,
            **({"study": study_logged} if study_logged is not None else {}),
            **(
                {"dataset_tags": dataset_tags}
                if dataset_tags is not None
                else {}
            ),
        },
        **({"tags": run_tags} if run_tags else {}),
    }

    return (
        dataset_config,
        dataset_config2,
        estimator_config,
        out_dir,
        validation_settings,
        caching_config,
        wandb_setup,
    )


def setup_study_run(
    args: argparse.Namespace,
    wandb_setup: dict,
    dataset_config: dict,
    dataset_config_to_compare: dict | None,
    estimator_config: dict,
    validation_settings: dict | None,
) -> str | None:
    """Materialize study run outputs and W&B inside one git capture context.

    Collects git state once via ``wandb_git_diff_capture`` and, while the
    temporary git index is live, builds the study output directory using
    ``state_label``, persists resolved configs, enriches the W&B study config
    with git metadata, attaches the W&B handler with native code capture,
    and logs initial artifacts so ``wandb.init`` records the matching diff.

    Args:
        args: Parsed command line arguments.
        wandb_setup: W&B handler configuration; mutated to inject git fields
            into ``config['study']`` and to record the resolved ``out_dir``.
        dataset_config: Resolved primary dataset config.
        dataset_config_to_compare: Optional secondary dataset config.
        estimator_config: Resolved estimator config.
        validation_settings: Optional validation settings.

    Returns:
        Output directory for training modes, otherwise ``None``.
    """
    with wandb_git_diff_capture() as state_label:
        out_dir: str | None = None
        if args.mode in {"train", "train-supervised"}:
            study = wandb_setup["config"]["study"]
            out_dir = os.path.join(
                "experiments",
                str(study["name"]),
                "runs",
                f"state={state_label}",
                wandb_setup["run_name"],
            )
            _create_run_outputs(
                args,
                out_dir,
                estimator_config,
                dataset_config,
                validation_settings,
            )
        wandb_setup["config"]["study"] = {
            **wandb_setup["config"]["study"],
            "state_label": state_label,
        }
        wandb_setup["config"]["out_dir"] = out_dir
        _attach_wandb_handler(wandb_setup, save_code=True)
        log_initial_config_artifacts(
            dataset_config,
            dataset_config_to_compare,
            estimator_config,
            validation_settings,
        )
    return out_dir


def log_initial_config_artifacts(
    dataset_config: dict,
    dataset_config_to_compare: dict | None,
    estimator_config: dict,
    validation_settings: dict | None,
) -> None:
    """Log initial configuration artifacts after W&B is attached.

    Args:
        dataset_config: Resolved primary dataset config to log.
        dataset_config_to_compare: Optional secondary dataset config.
        estimator_config: Resolved estimator config to log.
        validation_settings: Optional validation settings.
    """
    if hasattr(logger, "artifact"):
        logger.artifact(
            data=dataset_config,
            name="dataset_config",
            format="yaml",
            step=0,
        )
        logger.artifact(
            data=estimator_config,
            name="estimator_config",
            format="yaml",
            step=0,
        )
        if dataset_config_to_compare is not None:
            logger.artifact(
                data=dataset_config_to_compare,
                name="dataset_config_to_compare",
                format="yaml",
                step=0,
            )
        if validation_settings is not None:
            logger.artifact(
                data=validation_settings["dataset_config"],
                name="val_dataset_config",
                format="yaml",
                step=0,
            )
    else:
        logger.warning(
            "Logger does not support artifact logging. "
            "Configuration artifacts will not be logged."
        )


def _create_run_outputs(
    args: argparse.Namespace,
    out_dir: str,
    estimator_config: dict,
    dataset_config: dict,
    validation_settings: dict | None,
) -> None:
    """Create ``out_dir`` and persist resolved configs under ``configs/``.

    Args:
        args: Parsed command line arguments.
        out_dir: Run output directory; created if missing.
        estimator_config: Resolved estimator config to persist.
        dataset_config: Resolved dataset config to persist.
        validation_settings: Optional validation settings to persist.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_dir = os.path.join(out_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    shutil.copy2(args.estimator, os.path.join(cfg_dir, "estimator.yaml"))
    shutil.copy2(args.dataset, os.path.join(cfg_dir, "dataset.yaml"))
    dump_yaml(
        os.path.join(cfg_dir, "estimator_resolved.yaml"), estimator_config
    )
    dump_yaml(os.path.join(cfg_dir, "dataset_resolved.yaml"), dataset_config)
    if validation_settings is not None:
        dump_yaml(
            os.path.join(cfg_dir, "val_dataset_resolved.yaml"),
            validation_settings["dataset_config"],
        )


def _attach_wandb_handler(wandb_setup: dict, *, save_code: bool) -> None:
    handler_kwargs: dict = {
        "project": wandb_setup["project"],
        "run_name": wandb_setup["run_name"],
        "group": wandb_setup["group"],
        "config": wandb_setup["config"],
    }
    if wandb_setup.get("tags"):
        handler_kwargs["tags"] = wandb_setup["tags"]
    if save_code:
        handler_kwargs["wandb_init_kwargs"] = {
            "save_code": True,
            "settings": {"code_dir": "."},
        }
    gg.attach(gg.WandBHandler(**handler_kwargs), scopes=["global"])
