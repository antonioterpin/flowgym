"""Training and evaluation script for the different estimators."""

import argparse
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
from flowgym.checkpointing import CheckpointConfig
from flowgym.common.base import NNEstimatorTrainableState

# Training environment
from flowgym.environment.fluid_env import FluidEnv
from flowgym.make import make_estimator, select_gt
from flowgym.run_setup import prepare_configs, setup_study_run
from flowgym.training.caching import CacheManager

# Utils
from flowgym.utils import setup_logging
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
        "--estimator",
        type=str,
        required=True,
        help="Configuration of the estimator to use.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="src/flowgym/config/piv_dataset_class1_eval.yaml",
        help="Dataset configuration file path.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and validate configs (also writes resolved configs for non-study
    # training runs; study runs defer that until `setup_study_run` so the
    # output directory can include the captured git state label).
    (
        dataset_config,
        dataset_config_to_compare,
        estimator_config,
        out_dir,
        validation_settings,
        caching_config,
        wandb_setup,
    ) = prepare_configs(args)

    # Only study runs attach W&B (with native code capture). Non-study runs
    # keep the existing public behavior of running without wandb.
    if isinstance(wandb_setup["config"].get("study"), dict):
        out_dir = setup_study_run(
            args,
            wandb_setup,
            dataset_config,
            dataset_config_to_compare,
            estimator_config,
            validation_settings,
        )
    if out_dir is not None:
        logger.info(f"Saving estimators in directory: {out_dir}")

    key = jax.random.PRNGKey(dataset_config["seed"])
    key, subkey = jax.random.split(key)

    if args.mode not in ["train", "train-supervised"]:
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
            estimator_config["estimate_type"],
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

    estimate_shape = estimator_config.get("estimate_shape", None)
    if estimate_shape is not None:
        estimate_shape = (batch.images1.shape[0], *tuple(estimate_shape))
    else:
        estimate_shape = gt.shape

    # Create the estimator
    (trainable_state, create_state_fn, compute_estimate_fn, estimator) = (
        make_estimator(
            estimator_config,
            image_shape=(
                batch.images1.shape[0],
                *dataset_config["image_shape"],
            ),
            estimate_shape=estimate_shape,
            load_from=estimator_config.get("load_from", None),
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

    if args.mode == "eval":
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
            # Append estimator-specific suffix to cache_id (e.g. weight hash)
            if "cache_id" in caching_config:
                suffix = estimator.get_cache_id_suffix(trainable_state)
                if suffix:
                    caching_config["cache_id"] += suffix
                    cache_id = caching_config["cache_id"]
                    logger.info(
                        f"Updated cache_id with estimator suffix: {cache_id}"
                    )

            cache_manager = CacheManager(**caching_config)

        try:
            eval_full_dataset(
                estimator=estimator,
                sampler=sampler,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                trainable_state=trainable_state,
                estimate_type=estimator_config["estimate_type"],
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

        ckpt_cfg = CheckpointConfig.from_configs(
            estimator_config, dataset_config
        )
        try:
            train(
                estimator=estimator,
                estimator_config=estimator_config,
                trainable_state=trainable_state,
                out_dir=out_dir,
                checkpoint_config=ckpt_cfg,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                env=env,
                env_state=env_state,
                num_episodes=dataset_config.get("num_episodes", 1000),
                save_every=dataset_config.get("save_every", 100),
                log_every=dataset_config.get("log_every", 100),
                obs=obs,
                key=key,
                replay_buffer_capacity=estimator_config["config"]
                .get("replay_buffer_config", {})
                .get(
                    "capacity",
                    dataset_config.get("replay_buffer_capacity", 10000),
                ),
                replay_ratio=estimator_config["config"]
                .get("replay_buffer_config", {})
                .get("replay_ratio", dataset_config.get("replay_ratio", 0.0)),
                prefetch_replay_size=estimator_config["config"]
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

        logger.info("Training supervised estimator...")

        # Caching Setup
        cache_manager = None
        if caching_config:
            logger.info(
                "Initializing CacheManager from dataset config: "
                f"{caching_config}"
            )
            # Append estimator-specific suffix to cache_id (e.g. weight hash)
            if "cache_id" in caching_config:
                suffix = estimator.get_cache_id_suffix(trainable_state)
                if suffix:
                    caching_config["cache_id"] += suffix
                    cache_id = caching_config["cache_id"]
                    logger.info(
                        f"Updated cache_id with estimator suffix: {cache_id}"
                    )
            cache_manager = CacheManager(**caching_config)

        ckpt_cfg = CheckpointConfig.from_configs(
            estimator_config, dataset_config
        )
        try:
            train_supervised(
                estimator=estimator,
                estimator_config=estimator_config,
                trainable_state=trainable_state,
                out_dir=out_dir,
                checkpoint_config=ckpt_cfg,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                sampler=sampler,
                val_sampler=val_sampler,
                val_interval=val_interval,
                val_num_batches=val_num_batches,
                num_batches=dataset_config.get("num_batches", 1000),
                estimate_type=estimator_config["estimate_type"],
                save_every=dataset_config.get("save_every", 100),
                log_every=dataset_config.get("log_every", 1),
                save_only_best=dataset_config.get("save_only_best", False),
                key=key,
                replay_buffer_capacity=estimator_config["config"]
                .get("replay_buffer_config", {})
                .get(
                    "capacity",
                    dataset_config.get("replay_buffer_capacity", 0),
                ),
                replay_ratio=estimator_config["config"]
                .get("replay_buffer_config", {})
                .get("replay_ratio", dataset_config.get("replay_ratio", 0.0)),
                prefetch_replay_size=estimator_config["config"]
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
                    estimator=estimator,
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
                estimator_config=estimator_config,
                sampler1=sampler,
                sampler2=sampler2,
                estimator=estimator,
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
