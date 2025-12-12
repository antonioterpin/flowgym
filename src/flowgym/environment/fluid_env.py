"""Environment for training using synthetic images."""

import jax.numpy as jnp
import synthpix
from flowgym.make import select_gt
from flowgym.common.evaluation import loss_supervised_density
from synthpix.sampler import Sampler

EnvState = tuple[Sampler, jnp.ndarray | None]  # (sampler, gt)
Observation = tuple[jnp.ndarray, jnp.ndarray]  # pair of images (prev, curr)


class FluidEnv:
    """Fluid Environment for training using synthetic images."""

    def __init__(self, episode_length: int = 0, gt_type: str = "flow") -> None:
        """Initialize the Fluid Environment.

        Args:
            episode_length: Length of the episode for episodic training.
            gt_type: Type of ground truth to use, either 'flow' or 'density'.
        """
        if gt_type not in ["flow", "density"]:
            raise ValueError(
                f"Invalid ground truth type: {gt_type}. "
                "Must be one of 'flow', 'density'."
            )
        self.gt_type = gt_type
        """Initialize the Fluid Environment."""
        if episode_length < 0:
            raise ValueError("Episode length must be non-negative.")
        if episode_length == 0:
            self.episodic = False
        else:
            self.episodic = True
            self.episode_length = episode_length

    @classmethod
    def make(
        cls,
        dataset_config: dict,
    ) -> tuple["FluidEnv", EnvState]:
        """Create a new instance of the FluidEnv class and a sampler.

        Args:
            dataset_config: Configuration for the dataset.

        Returns:
            An instance of the FluidEnv class and a sampler.
        """
        sampler = synthpix.make(dataset_config)
        episode_length = dataset_config.get("episode_length", 0)
        gt_type = dataset_config.get("gt_type", "flow")
        return cls(episode_length, gt_type), (sampler, None)

    def reset(self, state: EnvState) -> tuple[Observation, EnvState, jnp.ndarray]:
        """Reset the environment to an initial state.

        Args:
            state: The current state of the environment (sampler, gt).

        Returns:
            obs: The initial observation (pair of images).
            state: The next state of the environment.
            done: An array of booleans indicating if each
                episode is done. Shape (batch_size,). If episodic is False,
                this will always be False.
        """
        # Unpack the state
        sampler, _ = state

        # Initialize the sampler and flow ground truth
        sampler.next_episode()
        batch = next(sampler)
        prev = batch.images1
        curr = batch.images2
        gt = select_gt(self.gt_type, batch)
        if self.episodic:
            if batch.done is None:
                raise ValueError("Episodic sampler must provide 'done' flags.")
            done = batch.done
        else:
            done = jnp.zeros(prev.shape[0], dtype=jnp.bool_)

        # Pack the observation and state
        obs = (prev, curr)
        state = (sampler, gt)

        return obs, state, done

    def step(
        self, state: EnvState, action: jnp.ndarray
    ) -> tuple[Observation, EnvState, jnp.ndarray, jnp.ndarray]:
        """Take a step in the environment using the given action.

        Args:
            state: The current state of the environment (sampler, gt).
            action: The estimated flow field.

        Returns:
            obs: next pair of images batch.
            state: The new state of the environment (sampler, gt).
            reward: The reward received after taking the action (-EPE).
            done: A boolean indicating if the episode is done.
        """
        # Unpack the state
        sampler, gt = state
        if gt is None:
            raise ValueError("Ground truth flow field is required in the state.")
        # # Compute the reward based on the flow field estimated
        r = self.calculate_reward(gt, action)

        # Take a step in the environment
        batch = next(sampler)
        prev = batch.images1
        curr = batch.images2
        flow_gt = batch.flow_fields
        if self.episodic:
            if batch.done is None:
                raise ValueError("Episodic sampler must provide 'done' flags.")
            done = batch.done
        else:
            done = jnp.zeros(prev.shape[0], dtype=jnp.bool_)

        # Pack the observation and state
        obs = (prev, curr)
        state = (sampler, flow_gt)

        return obs, state, r, done

    def calculate_reward(self, gt: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Calculate the reward based on the estimated flow field.

        Args:
            gt: The ground truth flow field.
            action: The estimated flow field.

        Returns:
            reward: The reward received after taking the action
            -EPE for flow, negative density loss for density.
        """
        if self.gt_type == "flow":
            # Calculate the EPE (End Point Error)
            epe = jnp.linalg.norm(gt - action, axis=-1)

            # Calculate the reward as the negative EPE
            reward = -jnp.mean(epe, axis=(1, 2))
        elif self.gt_type == "density":
            # Calculate the reward based on the density
            reward = -loss_supervised_density(gt, action)

        return reward

    def close(self, state: EnvState):
        """Shutdown the environment and release resources.

        Args:
            state (EnvState): The current state of the environment (sampler, gt).
        """
        # Unpack the state
        sampler, _ = state

        # Shutdown the sampler
        sampler.shutdown()
