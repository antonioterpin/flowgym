"""Train flow estimators with synthetic data using reinforcement learning."""

import contextlib
import time

import goggles as gg
import jax
import jax.numpy as jnp
import numpy as np
from goggles import Metrics

from flowgym.checkpointing import CheckpointConfig, Checkpointer
from flowgym.common.base import Estimator, NNEstimatorTrainableState
from flowgym.environment.fluid_env import EnvState, FluidEnv, Observation
from flowgym.training.replay import ReplayBuffer
from flowgym.types import (
    CompiledComputeEstimateFn,
    CompiledCreateStateFn,
    PRNGKey,
    RLExperience,
    RLTrainStep,
)
from flowgym.utils import DEBUG, GracefulShutdown, log_flow_estimate

logger = gg.get_logger(__name__, with_metrics=True)


def train(
    estimator: Estimator,
    estimator_config: dict,
    *,
    trainable_state: NNEstimatorTrainableState,
    out_dir: str,
    create_state_fn: CompiledCreateStateFn,
    compute_estimate_fn: CompiledComputeEstimateFn,
    env: FluidEnv,
    env_state: EnvState,
    num_episodes: int,
    save_every: int,
    log_every: int,
    obs: Observation,
    key: PRNGKey,
    replay_buffer_capacity: int = 10000,
    replay_ratio: float = 0.0,
    prefetch_replay_size: int = 0,
    checkpoint_config: CheckpointConfig | None = None,
) -> None:
    """Train the flow estimator.

    Args:
        estimator: The flow estimator.
        estimator_config: Configuration for the estimator.
        trainable_state: The initial state of the estimator.
        out_dir: Directory to save the estimator.
        create_state_fn: Function to create the initial state of the estimator.
        compute_estimate_fn: Function to compute the flow estimate.
        env: The environment to train in.
        env_state: Initial state of the environment.
        num_episodes: Number of episodes to train for.
        save_every: Frequency to save the estimator.
        log_every: Frequency to log the training progress.
        obs: Initial observations from the environment.
        key: Random key for JAX operations.
        replay_buffer_capacity: Capacity of the replay buffer.
        replay_ratio: Ratio of replay samples to collected samples.
        prefetch_replay_size: Prefetch buffer size for replay. If > 0,
            enables GPU prefetch.
        checkpoint_config: Checkpointer policy. RL training has no
            validation loop, so only ``save_periodic`` fires here and
            ``save_only_best`` / ``wandb_upload="best"`` modes are
            effectively no-ops; ``wandb_upload="every"`` still uploads
            each periodic save (alias ``latest``).
    """
    # Create the training step function
    train_step_fn = estimator.create_train_step()
    assert isinstance(train_step_fn, RLTrainStep), (
        f"Expected RLTrainStep callable, got {type(train_step_fn)}"
    )

    if estimator_config["config"].get("jit", False) and not DEBUG:
        train_step_fn = jax.jit(train_step_fn)

    logger.info("Estimator compiled successfully.")

    # Handle randomization key
    key, subkey = jax.random.split(key)

    # Create the estimator state
    # it is initialized with the first image and a zero flow estimate
    estimation_state = create_state_fn(obs[0], subkey)

    logger.info("Estimator state created successfully.")
    logger.debug(f"{estimation_state=}")

    # Initialize the replay buffer
    replay_buffer = None
    if replay_buffer_capacity > 0:
        key, subkey = jax.random.split(key)
        replay_buffer = ReplayBuffer(
            capacity=replay_buffer_capacity, key=subkey
        )
        logger.info(
            f"Replay buffer initialized with capacity {replay_buffer_capacity}."
        )

    with contextlib.ExitStack() as stack:
        checkpointer = stack.enter_context(
            Checkpointer(
                out_dir=out_dir,
                model=estimator,
                sampler=env_state[0],
                config=checkpoint_config or CheckpointConfig(),
            )
        )
        g = stack.enter_context(
            GracefulShutdown("Stop detected, finishing epoch...")
        )
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

                # Reset the state of the estimator and propagate the key
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
                metrics = dict(estimator.process_metrics(metrics))
                logger.push(metrics, step=episode_idx)

                # Extract the action (the last estimate)
                action = estimation_state["estimates"][:, -1]

                # Store the last observation and the last ground truth
                old_obs = obs
                old_gt = env_state[1]

                # Take a step in the environment
                obs, env_state, reward, done = env.step(env_state, action)

                # Do one training step
                experience = RLExperience(
                    state=estimation_state,
                    reward=reward,
                    obs=obs,
                    old_obs=old_obs,
                    old_flow_estimate=old_flow_estimate,
                )
                loss, trainable_state, _ = train_step_fn(
                    trainable_state=trainable_state,
                    target_state=old_trainable_state,
                    experience=experience,
                )

                # Store experience in the replay buffer
                if replay_buffer is not None:
                    # We store unbatched experiences
                    B = obs[0].shape[0]
                    for i in range(B):
                        exp = RLExperience(
                            state=jax.tree_util.tree_map(
                                lambda x, i=i: x[i], estimation_state
                            ),
                            reward=reward[i],
                            obs=(obs[0][i], obs[1][i]),
                            old_obs=(
                                old_obs[0][i],
                                old_obs[1][i],
                            ),
                            old_flow_estimate=old_flow_estimate[i],
                        )
                        replay_buffer.push(exp)

                    # Replay training steps
                    # We perform replay steps with probability replay_ratio % 1
                    # and int(replay_ratio) steps
                    num_replay_steps = int(replay_ratio)
                    key, subkey = jax.random.split(key)
                    if jax.random.uniform(subkey) < (replay_ratio % 1):
                        num_replay_steps += 1

                    for _ in range(num_replay_steps):
                        if len(replay_buffer) >= B:
                            # Use sample_iter with prefetch for GPU prefetch
                            replay_iter = replay_buffer.sample_iter(
                                batch_size=B,
                                num_batches=1,
                                prefetch=prefetch_replay_size,
                            )
                            for replay_exp in replay_iter:
                                (
                                    replay_loss,
                                    trainable_state,
                                    _,
                                ) = train_step_fn(
                                    trainable_state=trainable_state,
                                    target_state=old_trainable_state,
                                    experience=replay_exp,
                                )
                                logger.scalar(
                                    "loss/replay",
                                    float(replay_loss),
                                    step=episode_idx,
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
                logger.scalar(
                    "reward", float(jnp.mean(reward)), step=episode_idx
                )

                # Add loss and reward to episode metrics
                metrics["loss"] = float(loss)
                metrics["reward"] = np.mean(reward).item()
                metrics["step_time"] = t
                episode_metrics.append(metrics)

                # Update the state of the estimator
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
                avg_value = np.mean(
                    np.array([m[k] for m in episode_metrics])
                ).item()
                logger.scalar(f"episode/{k}", avg_value, step=episode_idx)
                logger.info(f"Episode {episode_idx} - {k}: {avg_value}")

            if episode_idx % save_every == 0:
                saved_path = checkpointer.save_periodic(
                    trainable_state, episode_idx
                )
                if saved_path is not None:
                    logger.info(
                        f"Checkpoint saved at episode {episode_idx} -> "
                        f"{saved_path}"
                    )

    t_total = time.time() - t_total
    logger.info(f"Training took {t_total} seconds.")
