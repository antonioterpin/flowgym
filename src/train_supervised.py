"""Train flow estimators with supervised learning."""

import time
import jax
import goggles as gg
from flowgym.common.base import Estimator, EstimatorTrainableState
from flowgym.environment.fluid_env import FluidEnv
from flowgym.utils import GracefulShutdown, DEBUG
from flowgym.make import save_model
from flowgym.types import PRNGKey

logger = gg.get_logger(__name__, with_metrics=True)


def train_supervised(
    model: Estimator,
    model_config: dict,
    trainable_state: EstimatorTrainableState,
    out_dir: str,
    create_state_fn,
    env: FluidEnv,
    env_state: tuple,
    num_episodes: int,
    save_every: int,
    obs: tuple,
    key: PRNGKey,
):
    """Train the flow estimator.

    Args:
        model: The flow estimator model.
        model_config: Configuration for the model.
        trainable_state: The initial state of the model.
        out_dir: Directory to save the model.
        create_state_fn: Function to create the initial state of the estimator.
        compute_estimate_fn: Function to compute the flow estimate.
        env: The environment to train in.
        env_state: Initial state of the environment.
        num_episodes: Number of episodes to train for.
        save_every: Frequency to save the model.
        log_every: Frequency to log the training progress.
        obs: Initial observations from the environment.
        key: Random key for JAX operations.
    """
    # Create the training step function
    train_step_fn = model.create_train_step()
    if model_config["config"].get("jit", False) and not DEBUG:
        train_step_fn = jax.jit(train_step_fn)

    logger.info("Model compiled successfully.")

    # Handle randomization key
    key, subkey = jax.random.split(key)

    # Create the estimator state
    # it is initialized with the first image
    # and a first estimate of all zeros
    estimation_state = create_state_fn(obs[0], subkey)

    logger.info("Estimator state created successfully.")
    logger.debug(f"{estimation_state=}")

    with (GracefulShutdown("Stop detected, finishing epoch...") as g,):
        episode_idx = 0
        t_total = time.time()
        while episode_idx < num_episodes and not g.stop:
            episode_idx += 1

            if (
                episode_idx > 1
            ):  # No need to reset the environment on the first iteration
                # Reset the environment to the next episode
                t_reset = time.time()
                obs, env_state, _ = env.reset(env_state)
                t_reset = time.time() - t_reset
                logger.info(f"Environment reset took {t_reset} seconds.")

                # Reset the state of the model but propagate the key
                key, subkey = jax.random.split(key)
                estimation_state = create_state_fn(obs[0], subkey)

            # Store the new ground truth
            gt = env_state[1]

            # Do one training step
            t = time.time()
            loss, trainable_state, _ = train_step_fn(
                estimation_state,
                trainable_state,
                None,
                None,
                obs,  # next state of the environment
                None,  # previous state of the environment
                None,
                gt,  # ground truth
            )
            t = time.time() - t
            logger.info(f"Training step took {t} seconds.")

            logger.scalar("loss", float(loss), step=episode_idx)

            logger.info(f"Batch {episode_idx} finished.\n" f"loss: {loss:.6f}.")

            if episode_idx % save_every == 0:
                save_model(
                    trainable_state,
                    out_dir,
                    f"{model.__class__.__name__}-{episode_idx}",
                )

    t_total = time.time() - t_total
    logger.info(f"Training took {t_total} seconds.")
