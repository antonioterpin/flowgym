"""Training and evaluation script for the different estimators."""

import argparse
import os
import time
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Make flow_estimator available as a module for pickle loading
import flowgym

sys.modules["flow_estimator"] = flowgym

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import synthpix  # noqa: E402
import goggles as gg

# Training environment
from flowgym.environment.fluid_env import FluidEnv  # noqa: E402
from train import train  # noqa: E402
from train_supervised import train_supervised  # noqa: E402
from synthpix.sampler import SyntheticImageSampler, RealImageSampler  # noqa: E402

# Utils
from flowgym.run_setup import prepare_configs, setup_study_run  # noqa: E402
from flowgym.utils import setup_logging  # noqa: E402
from flowgym.make import make_estimator, select_gt  # noqa: E402
from eval import eval_full_dataset, eval  # noqa: E402
from compare import comparison  # noqa: E402

logger = setup_logging(debug=False, use_wandb=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the SyntheticImageSampler pipeline."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="main",
        choices=["main", "train", "eval", "compare_samplers", "train_supervised"],
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


if __name__ == "__main__":
    args = parse_args()

    # Load and validate configs (also writes resolved configs for non-study
    # training runs; study runs defer that until `setup_study_run` so the
    # output directory can include the captured git state label).
    (
        dataset_config,
        dataset_config_to_compare,
        model_config,
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
            model_config,
            validation_settings,
        )
    if out_dir is not None:
        logger.info(f"Saving models in directory: {out_dir}")

    key = jax.random.PRNGKey(dataset_config["seed"])
    key, subkey = jax.random.split(key)

    if args.mode not in ["train", "train_supervised"]:
        # Load the dataset sampler
        sampler = synthpix.make(dataset_config)

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

            if env_state[1] is None:
                raise ValueError("Groundtruth flow fields in env_state cannot be None.")
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
    (trainable_state, create_state_fn, compute_estimate_fn, model) = make_estimator(
        model_config,
        image_shape=(batch.images1.shape[0], *dataset_config["image_shape"]),
        estimate_shape=estimate_shape,
        load_from=model_config.get("load_from", None),
        rng=subkey,
    )

    if args.mode == "eval":
        if create_state_fn is None or compute_estimate_fn is None:
            raise ValueError(
                "create_state_fn and compute_estimate_fn must be provided for evaluation."
            )
        try:
            eval_full_dataset(
                model=model,
                sampler=sampler,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                trainable_state=trainable_state,
                estimate_type=model_config["estimate_type"],
                key=key,
            )
        finally:
            sampler.shutdown()
            time.sleep(5)  # wait for the sampler to shutdown properly
            gg.finish()
    elif args.mode == "train":
        if out_dir is None:
            raise ValueError("out_dir must be provided for training mode.")
        if trainable_state is None:
            raise ValueError("trainable_state must be provided for training mode.")
        if create_state_fn is None or compute_estimate_fn is None:
            raise ValueError(
                "create_state_fn and compute_estimate_fn must be provided for training."
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
            )
        finally:
            sampler = env_state[0]
            sampler.shutdown()
            time.sleep(5)  # wait for the sampler to shutdown properly
            gg.finish()
    elif args.mode == "train_supervised":
        if trainable_state is None:
            raise ValueError("trainable_state must be provided for training mode.")
        if out_dir is None:
            raise ValueError("out_dir must be provided for training mode.")
        try:
            train_supervised(
                model=model,
                model_config=model_config,
                trainable_state=trainable_state,
                out_dir=out_dir,
                create_state_fn=create_state_fn,
                env=env,
                env_state=env_state,
                num_episodes=dataset_config.get("num_episodes", 1000),
                save_every=dataset_config.get("save_every", 100),
                obs=obs,
                key=key,
            )
        finally:
            sampler = env_state[0]
            sampler.shutdown()
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
    elif args.mode == "compare_samplers":
        if create_state_fn is None:
            raise ValueError("create_state_fn must be provided for comparison mode.")
        # create a second sampler to load real images from files
        assert dataset_config_to_compare is not None
        sampler2 = synthpix.make(dataset_config_to_compare)
        if not isinstance(sampler, SyntheticImageSampler):
            raise TypeError("sampler must be an instance of SyntheticImageSampler.")
        if not isinstance(sampler2, RealImageSampler):
            raise TypeError("sampler2 must be an instance of RealImageSampler.")
        try:
            batch2 = next(sampler2)
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
