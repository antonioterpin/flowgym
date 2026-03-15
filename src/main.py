"""Training and evaluation script for the different estimators."""

import argparse
import copy
import os
import time

import goggles as gg
import jax
import jax.numpy as jnp
import synthpix
from synthpix.sampler import (
    RealImageSampler,
    SyntheticImageSampler,
)

from compare import comparison
from eval import eval, eval_full_dataset
from flowgym.common.base import NNEstimatorTrainableState

# Training environment
from flowgym.environment.fluid_env import FluidEnv
from flowgym.make import make_estimator, select_gt
from flowgym.training.caching import CacheManager

# Utils
from flowgym.utils import load_configuration, setup_logging
from train import train
from train_supervised import train_supervised

logger = setup_logging(debug=False, use_wandb=False)


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the SyntheticImageSampler pipeline."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="main",
        choices=[
            "main",
            "train",
            "eval",
            "compare-samplers",
            "train-supervised",
        ],
        help="Mode of operation: 'main' for playing around, "
        "'eval' for evaluation only.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Configuration of the model to use.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="src/flowgym/config/piv_dataset_class1_eval.yaml",
        help="Dataset configuration file path.",
    )

    return parser.parse_args()


def prepare_configs(
    args: argparse.Namespace,
) -> tuple[dict, dict | None, dict, str | None, dict | None, dict | None]:
    """Prepare dataset and model configurations based on command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple containing:
            - Dataset configuration dictionary.
            - Second dataset configuration dictionary for comparison,
                if in compare mode.
            - Model configuration dictionary.
            - Output directory path if in training mode, else None.
            - Validation settings dictionary, if validation is enabled.
            - Caching configuration dictionary, if caching is enabled.

    Raises:
        ValueError: If validation configuration is invalid or missing.
    """
    # Load the dataset
    dataset_config = load_configuration(args.dataset)
    dataset_config2 = None
    validation_settings = None
    caching_config = None

    # Load the model
    model_config = load_configuration(args.model)

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

    if args.mode in {"train", "train-supervised"}:
        # Prepare the output directory to save the model
        out_dir = model_config.get("out_dir", "output")
        out_dir = os.path.join(
            out_dir, model_config["estimator"], str(dataset_config["seed"])
        )
        os.makedirs(out_dir, exist_ok=True)
    elif args.mode == "compare-samplers":
        # create a second sampler to load real images from files
        dataset_config2 = load_configuration(args.dataset)
        dataset_config2["include_images"] = True
        dataset_config2["loop"] = False
        dataset_config2["randomize"] = False

    log_model_config = {**model_config}
    log_model_config["config"] = {**model_config["config"]}
    for k, v in log_model_config["config"].items():
        if isinstance(v, str) and v.endswith(".yaml"):
            log_model_config["config"][k] = load_configuration(v)

    if model_config.get("run_name", None) is None:
        model_config["run_name"] = (
            f"{args.mode}_{model_config['estimator']}_{args.dataset.split('/')[-1].split('.')[0]}"
        )

    gg.attach(
        gg.WandBHandler(
            project="Art_of_PIV",
            run_name=model_config["run_name"],
            config={
                "model_config": log_model_config,
                "dataset_config": dataset_config,
                "dataset_config2": dataset_config2,
                "validation": validation_settings,
                "mode": args.mode,
                "out_dir": out_dir,
            },
        )
    )

    return (
        dataset_config,
        dataset_config2,
        model_config,
        out_dir,
        validation_settings,
        caching_config,
    )


if __name__ == "__main__":
    args = parse_args()

    # Load the dataset
    (
        dataset_config,
        dataset_config_to_compare,
        model_config,
        out_dir,
        validation_settings,
        caching_config,
    ) = prepare_configs(args)

    # Log the configurations
    if hasattr(logger, "artifact"):
        logger.artifact(  # pyright: ignore[reportAttributeAccessIssue]
            data=dataset_config,
            name="dataset_config",
            format="yaml",
            step=0,
        )
        logger.artifact(  # pyright: ignore[reportAttributeAccessIssue]
            data=model_config,
            name="model_config",
            format="yaml",
            step=0,
        )
        if dataset_config_to_compare is not None:
            logger.artifact(  # pyright: ignore[reportAttributeAccessIssue]
                data=dataset_config_to_compare,
                name="dataset_config_to_compare",
                format="yaml",
                step=0,
            )
        if validation_settings is not None:
            logger.artifact(  # pyright: ignore[reportAttributeAccessIssue]
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
    if out_dir is not None:
        logger.info(f"Saving models in directory: {out_dir}")

    key = jax.random.PRNGKey(dataset_config["seed"])
    key, subkey = jax.random.split(key)

    if args.mode not in ["train", "train_supervised"]:
        # Load the dataset sampler
        sampler = synthpix.make(
            dataset_config, load_from=dataset_config.get("load_from")
        )

        logger.info("Dataset loaded successfully.")

        try:
            # Extract a batch for initialization
            batch = next(sampler)
        except Exception as e:
            logger.error(f"Error loading initial batch: {e}")
            sampler.shutdown()
            gg.finish()
            raise  # Re-raise the exception after shutdown

        gt = select_gt(
            model_config["estimate_type"],
            batch,
        )
    else:
        # Create the environment for training
        # The state of the environment is a tuple of (sampler, flow_gt)
        env, env_state = FluidEnv.make(dataset_config=dataset_config)

        try:
            # Reset the environment
            # obs is the image pair (prev, curr)
            obs, env_state, done = env.reset(env_state)
            sampler = env_state[0]

            if env_state[1] is None:
                raise ValueError(
                    "Groundtruth flow fields in env_state cannot be None."
                )
            gt = env_state[1]

            logger.info("Environment created successfully.")

            # Extract a batch for initialization
            batch = synthpix.SynthpixBatch(
                images1=obs[0],
                images2=obs[1],
                flow_fields=env_state[1],  # groundtruth flow fields
                done=jnp.array([done]),  # done flag
            )
        except Exception as e:
            logger.error(f"Error loading initial batch: {e}")
            sampler = env_state[0]
            sampler.shutdown()
            gg.finish()

    estimate_shape = model_config.get("estimate_shape", None)
    if estimate_shape is not None:
        estimate_shape = (batch.images1.shape[0], *tuple(estimate_shape))
    else:
        estimate_shape = gt.shape

    # Create the estimator
    (trainable_state, create_state_fn, compute_estimate_fn, model) = (
        make_estimator(
            model_config,
            image_shape=(
                batch.images1.shape[0],
                *dataset_config["image_shape"],
            ),
            estimate_shape=estimate_shape,
            load_from=model_config.get("load_from", None),
            rng=subkey,
        )
    )
    val_sampler = None
    val_interval = None
    val_num_batches = 1
    if (
        args.mode in ["train", "train-supervised"]
        and validation_settings is not None
    ):
        val_dataset_config = validation_settings["dataset_config"]
        val_interval = validation_settings.get("interval")
        val_num_batches = validation_settings.get("num_batches", 1)
        val_sampler = synthpix.make(
            val_dataset_config, load_from=val_dataset_config.get("load_from")
        )
        logger.info("Validation sampler created successfully.")

    elif args.mode == "eval":
        if create_state_fn is None or compute_estimate_fn is None:
            raise ValueError(
                "create_state_fn and compute_estimate_fn must be provided "
                "for evaluation."
            )

        # Caching Setup
        cache_manager = None
        if caching_config:
            logger.info(
                "Initializing CacheManager from dataset config: "
                f"{caching_config}"
            )
            # Append model-specific suffix to cache_id (e.g. hash of weights)
            if "cache_id" in caching_config:
                suffix = model.get_cache_id_suffix(trainable_state)
                if suffix:
                    caching_config["cache_id"] += suffix
                    cache_id = caching_config["cache_id"]
                    logger.info(
                        f"Updated cache_id with model suffix: {cache_id}"
                    )

            cache_manager = CacheManager(**caching_config)

        try:
            eval_full_dataset(
                model=model,
                sampler=sampler,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                trainable_state=trainable_state,
                estimate_type=model_config["estimate_type"],
                key=key,
                print_files=dataset_config.get("print_files", False),
                num_batches=dataset_config.get("num_batches", None),
                cache_manager=cache_manager,
            )
        finally:
            sampler.shutdown()
            if cache_manager is not None:
                cache_manager.close()
            time.sleep(5)  # wait for the sampler to shutdown properly
            gg.finish()
    elif args.mode == "train":
        if out_dir is None:
            raise ValueError("out_dir must be provided for training mode.")
        if trainable_state is None:
            raise ValueError(
                "trainable_state must be provided for training mode."
            )
        if create_state_fn is None or compute_estimate_fn is None:
            raise ValueError(
                "create_state_fn and compute_estimate_fn must be provided "
                "for training."
            )
        if not isinstance(trainable_state, NNEstimatorTrainableState):
            raise ValueError(
                "trainable_state must be an instance of "
                "NNEstimatorTrainableState."
            )

        try:
            train(
                model=model,
                model_config=model_config,
                trainable_state=trainable_state,
                out_dir=out_dir,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                env=env,
                env_state=env_state,
                num_episodes=dataset_config.get("num_episodes", 1000),
                save_every=dataset_config.get("save_every", 100),
                log_every=dataset_config.get("log_every", 100),
                obs=obs,
                key=key,
                replay_buffer_capacity=model_config["config"]
                .get("replay_buffer_config", {})
                .get(
                    "capacity",
                    dataset_config.get("replay_buffer_capacity", 10000),
                ),
                replay_ratio=model_config["config"]
                .get("replay_buffer_config", {})
                .get("replay_ratio", dataset_config.get("replay_ratio", 0.0)),
                prefetch_replay_size=model_config["config"]
                .get("replay_buffer_config", {})
                .get(
                    "prefetch_replay_size",
                    dataset_config.get("prefetch_replay_size", 0),
                ),
            )
        finally:
            sampler = env_state[0]
            sampler.shutdown()
            time.sleep(5)  # wait for the sampler to shutdown properly
            gg.finish()
    elif args.mode == "train-supervised":
        if trainable_state is None:
            raise ValueError(
                "trainable_state must be provided for training mode."
            )
        if out_dir is None:
            raise ValueError("out_dir must be provided for training mode.")
        if create_state_fn is None or compute_estimate_fn is None:
            raise ValueError(
                "create_state_fn and compute_estimate_fn must be provided "
                "for training."
            )
        if sampler is None:
            raise ValueError(
                "sampler must be provided for supervised training mode."
            )
        if not isinstance(trainable_state, NNEstimatorTrainableState):
            raise ValueError(
                "trainable_state must be an instance of "
                "NNEstimatorTrainableState."
            )

        logger.info("Training supervised model...")

        # Caching Setup
        cache_manager = None
        if caching_config:
            logger.info(
                "Initializing CacheManager from dataset config: "
                f"{caching_config}"
            )
            # Append model-specific suffix to cache_id (e.g. hash of weights)
            if "cache_id" in caching_config:
                suffix = model.get_cache_id_suffix(trainable_state)
                if suffix:
                    caching_config["cache_id"] += suffix
                    cache_id = caching_config["cache_id"]
                    logger.info(
                        f"Updated cache_id with model suffix: {cache_id}"
                    )
            cache_manager = CacheManager(**caching_config)

        try:
            train_supervised(
                model=model,
                model_config=model_config,
                trainable_state=trainable_state,
                out_dir=out_dir,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                sampler=sampler,
                val_sampler=val_sampler,
                val_interval=val_interval,
                val_num_batches=val_num_batches,
                num_batches=dataset_config.get("num_batches", 1000),
                estimate_type=model_config["estimate_type"],
                save_every=dataset_config.get("save_every", 100),
                log_every=dataset_config.get("log_every", 1),
                save_only_best=dataset_config.get("save_only_best", False),
                key=key,
                replay_buffer_capacity=model_config["config"]
                .get("replay_buffer_config", {})
                .get(
                    "capacity",
                    dataset_config.get("replay_buffer_capacity", 0),
                ),
                replay_ratio=model_config["config"]
                .get("replay_buffer_config", {})
                .get("replay_ratio", dataset_config.get("replay_ratio", 0.0)),
                prefetch_replay_size=model_config["config"]
                .get("replay_buffer_config", {})
                .get(
                    "prefetch_replay_size",
                    dataset_config.get("prefetch_replay_size", 0),
                ),
                cache_manager=cache_manager,
            )
        finally:
            sampler.shutdown()
            if val_sampler is not None:
                val_sampler.shutdown()
            if cache_manager is not None:
                cache_manager.close()
            time.sleep(5)  # wait for the sampler to shutdown properly
            gg.finish()
    elif args.mode == "main":
        if create_state_fn is None:
            raise ValueError("create_state_fn must be provided for main mode.")
        try:
            for i in range(5):
                if i != 0:
                    batch = next(sampler)
                eval(
                    model=model,
                    trainable_state=trainable_state,
                    create_state_fn=create_state_fn,
                    compute_estimate_fn=compute_estimate_fn,
                    batch=batch,
                    key=key,
                )
        finally:
            sampler.shutdown()
            time.sleep(5)  # wait for the sampler to shutdown properly
            gg.finish()

    elif args.mode == "compare-samplers":
        if create_state_fn is None:
            raise ValueError(
                "create_state_fn must be provided for comparison mode."
            )
        # create a second sampler to load real images from files
        assert dataset_config_to_compare is not None
        sampler2 = synthpix.make(
            dataset_config_to_compare,
            load_from=dataset_config_to_compare.get("load_from"),
        )
        if not isinstance(sampler, SyntheticImageSampler):
            raise TypeError(
                "sampler must be an instance of SyntheticImageSampler."
            )
        if not isinstance(sampler2, RealImageSampler):
            raise TypeError("sampler2 must be an instance of RealImageSampler.")
        try:
            batch2 = next(sampler2)
            assert trainable_state is not None
            comparison(
                model_config=model_config,
                sampler1=sampler,
                sampler2=sampler2,
                model=model,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                trainable_state=trainable_state,
                key=key,
            )
        finally:
            sampler2.shutdown()
            sampler.shutdown()
            time.sleep(5)  # wait for the samplers to shutdown properly
            gg.finish()
