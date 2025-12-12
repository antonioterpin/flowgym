"""Train flow estimators with synthetic data using reinforcement learning."""

import time
import numpy as np

import jax
import jax.numpy as jnp
import goggles as gg
from collections.abc import Callable

from flowgym.common.base import Estimator, EstimatorTrainableState
from flowgym.environment.fluid_env import EnvState, FluidEnv, Observation
from flowgym.utils import GracefulShutdown, DEBUG

from flowgym.types import PRNGKey
from goggles.history.types import History
from goggles import Metrics
from flowgym.make import save_model
from flowgym.utils import log_flow_estimate

logger = gg.get_logger(__name__, with_metrics=True)


def train(
    model: Estimator,
    model_config: dict,
    trainable_state: EstimatorTrainableState,
    out_dir: str,
    create_state_fn: Callable[[jnp.ndarray, PRNGKey], History],
    compute_estimate_fn: Callable[
        [jnp.ndarray, History, EstimatorTrainableState], tuple[History, dict]
    ],
    env: FluidEnv,
    env_state: EnvState,
    num_episodes: int,
    save_every: int,
    log_every: int,
    obs: Observation,
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
    # it is initialized with the first image and a zero flow estimate
    estimation_state = create_state_fn(obs[0], subkey)

    logger.info("Estimator state created successfully.")
    logger.debug(f"{estimation_state=}")

    with GracefulShutdown("Stop detected, finishing epoch...") as g:
        episode_idx = 0
        done = jnp.array([False] * obs[0].shape[0], dtype=jnp.bool_)
        t_total = time.time()
        while episode_idx < num_episodes and not g.stop:
            episode_idx += 1

            # Store the same Q function throughout the episode
            old_trainable_state = trainable_state

            # Initialize episode metrics
            episode_metrics: list[Metrics] = []

            if (
                episode_idx > 1
            ):  # No need to reset the environment on the first iteration
                # Reset the environment to the next episode
                t_reset = time.time()
                obs, env_state, done = env.reset(env_state)
                t_reset = time.time() - t_reset
                logger.info(f"Environment reset took {t_reset} seconds.")

                # Reset the state of the model and propagate the key
                key, subkey = jax.random.split(key)
                estimation_state = create_state_fn(obs[0], subkey)

            step_idx = 0
            while not any(done):
                step_idx += 1

                # Store the last estimate
                old_flow_estimate = estimation_state["estimates"][:, -1]

                # Compute the flow estimate
                t = time.time()
                estimation_state, metrics = compute_estimate_fn(
                    obs[1], estimation_state, trainable_state
                )
                t = time.time() - t

                # Post process, log and append the metrics
                metrics = model.process_metrics(metrics)
                logger.push(metrics, step=episode_idx)

                # Extract the action (the last estimate)
                action = estimation_state["estimates"][:, -1]

                # Store the last observation and the last ground truth
                old_obs = obs
                old_gt = env_state[1]

                # Take a step in the environment
                obs, env_state, reward, done = env.step(env_state, action)

                # Do one training step
                loss, trainable_state, _ = train_step_fn(
                    estimation_state,
                    trainable_state,
                    old_trainable_state,
                    reward,
                    obs,  # next state of the environment
                    old_obs,  # previous state of the environment
                    old_flow_estimate,
                )

                # Log flow estimate images
                log_flow_estimate(
                    step_idx,
                    log_every,
                    np.array(estimation_state["estimates"][:, -1]),
                    np.array(old_gt),
                    episode_idx,
                )
                # Log loss and reward scalars
                logger.scalar("loss", float(loss), step=episode_idx)
                logger.scalar("reward", float(jnp.mean(reward)), step=episode_idx)

                # Add loss and reward to episode metrics
                metrics["loss"] = loss
                metrics["reward"] = np.mean(reward).item()
                metrics["step_time"] = t
                episode_metrics.append(metrics)

                # Update the state of the model
                if len(estimation_state["images"][0]) > 1:
                    estimation_state["images"] = (
                        estimation_state["images"]
                        .at[:, :-1, ...]
                        .set(estimation_state["images"][:, 1:, ...])
                    )
                estimation_state["images"] = (
                    estimation_state["images"].at[:, -1, ...].set(obs[0])
                )

            # Log episode metrics averages
            for k in episode_metrics[0].keys():
                avg_value = np.mean([m[k] for m in episode_metrics]).item()
                logger.scalar(f"episode/{k}", avg_value, step=episode_idx)
                logger.info(f"Episode {episode_idx} - {k}: {avg_value}")

            if episode_idx % save_every == 0:
                save_model(
                    trainable_state,
                    out_dir,
                    f"{model.__class__.__name__}-{episode_idx}",
                )

    t_total = time.time() - t_total
    logger.info(f"Training took {t_total} seconds.")
