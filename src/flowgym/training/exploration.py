"""Exploration policies for action selection."""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

from flowgym.types import PRNGKey

ExplorationKind = Literal["epsilon_greedy", "softmax"]
ExplorationPolicy = Callable[
    [jnp.ndarray, jnp.ndarray, PRNGKey], jnp.ndarray
]  # (q_values, exploration_rate, key) -> actions
# exploration_rate can be epsilon for epsilon_greedy or temperature for softmax


def epsilon_greedy_explore(
    q_values: jnp.ndarray,
    epsilon: jnp.ndarray,
    key: PRNGKey,
) -> jnp.ndarray:
    """Epsilon-greedy exploration policy.

    Args:
        q_values: Q-values for each action, shape (batch_size, num_actions).
        epsilon: Exploration rate, shape (batch_size,).
        key: PRNG key for sampling.

    Returns:
        Sampled actions, shape (batch_size,).
    """
    epsilon = jnp.broadcast_to(epsilon, q_values.shape[0])
    key_values, key_actions = jax.random.split(key)
    batch_size, num_actions = q_values.shape
    random_values = jax.random.uniform(key_values, shape=(batch_size,))
    random_actions = jax.random.randint(
        key_actions, shape=(batch_size,), minval=0, maxval=num_actions
    )
    greedy_actions = jnp.argmax(q_values, axis=-1)
    explore_mask = random_values < epsilon
    return jnp.where(explore_mask, random_actions, greedy_actions)


def softmax_explore(
    q_values: jnp.ndarray,
    temperature: jnp.ndarray,
    key: PRNGKey,
) -> jnp.ndarray:
    """Softmax exploration policy.

    Args:
        q_values: Q-values for each action, shape (batch_size, num_actions).
        temperature: Temperature parameter, shape (batch_size,).
        key: PRNG key for sampling.

    Returns:
        Sampled actions, shape (batch_size,).
    """
    temperature = jnp.broadcast_to(temperature, q_values.shape[0])
    logits = q_values / jnp.maximum(temperature[:, jnp.newaxis], 1e-8)
    return jax.random.categorical(key, logits, axis=-1)


EXPLORATION_REGISTRY: dict[ExplorationKind, ExplorationPolicy] = {
    "epsilon_greedy": epsilon_greedy_explore,
    "softmax": softmax_explore,
}
