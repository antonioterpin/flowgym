"""Router: Q-value estimation and action selection for ART of PIV."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax

from flowgym.common.base import NNEstimatorTrainableState
from flowgym.flow.art_of_piv.types import RouterOutputs, RouterQEstimateFn
from flowgym.training.exploration import ExplorationPolicy


def q_estimate(
    prev: jnp.ndarray,
    curr: jnp.ndarray,
    trainable_state: NNEstimatorTrainableState,
    *,
    model_apply_fn: Callable[..., Any],
    use_last_estimate: bool,
    flow: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate Q-values for each sub-estimator.

    Args:
        prev: Previous frame, shape (B, H, W).
        curr: Current frame, shape (B, H, W).
        trainable_state: Current trainable state with params.
        model_apply_fn: The Flax model's .apply method.
        use_last_estimate: Whether to concatenate flow to the input.
        flow: Optional last flow estimate, shape (B, H, W, 2).

    Returns:
        Q-values of shape (B, num_estimators).
    """
    # Stack the input channels: prev, curr, flow_x, flow_y
    input = jnp.stack([prev, curr], axis=-1)  # (B, H, W, 2)
    if use_last_estimate and flow is not None:
        input = jnp.concatenate([input, flow], axis=-1)  # (B, H, W, 4)

    # Forward pass
    q_values = model_apply_fn(
        {"params": trainable_state.params},
        input,
    )  # (B, num_estimators)

    return q_values


def run_router(
    prev: jnp.ndarray,
    curr: jnp.ndarray,
    keys: jnp.ndarray,
    steps: jnp.ndarray,
    *,
    q_estimate_fn: RouterQEstimateFn,
    schedule: optax.Schedule,
    explore: ExplorationPolicy,
    flow: jnp.ndarray | None = None,
    train: bool = True,
) -> RouterOutputs:
    """Run the router to select actions from Q-values and exploration.

    Args:
        prev: Previous frame, shape (B, H, W).
        curr: Current frame, shape (B, H, W).
        keys: PRNG keys for random sampling, shape (B, 2).
        steps: Training step count(s), shape (B,) or (B, 1) or scalar.
        q_estimate_fn: Function to estimate Q-values. Called as
            q_estimate_fn(prev, curr, flow=flow) -> (B, num_estimators).
        schedule: Exploration schedule mapping step → exploration value.
        explore: Exploration policy object (e.g. epsilon_greedy).
        flow: Optional last flow estimate.
        train: If False, exploration is zeroed.

    Returns:
        Structured router outputs containing actions, q-values, exploration
        values, and estimate keys.
    """
    B = prev.shape[0]

    # Compute exploration rate from schedule
    steps_arr = jnp.asarray(steps)
    if steps_arr.ndim == 2:
        step_values = steps_arr[:, 0]
    elif steps_arr.ndim == 1:
        step_values = steps_arr
    else:
        step_values = jnp.full((B,), steps_arr.astype(jnp.int32))

    # Broadcast a single shared step to the whole batch.
    if step_values.shape[0] == 1 and B > 1:
        step_values = jnp.full((B,), step_values[0])
    elif step_values.shape[0] != B:
        step_values = jnp.full((B,), step_values[0])

    step_values = step_values.astype(jnp.int32)
    exploration_val = jnp.asarray(jax.vmap(schedule)(step_values))

    # Zero exploration during validation
    exploration_val = jnp.where(train, exploration_val, 0.0)

    # Get Q-values from router
    qs = q_estimate_fn(prev, curr, flow=flow)

    # Split keys for exploration and estimation
    keys_arr = jnp.asarray(keys).reshape(B, -1)[:, -2:]  # (B, 2)
    double_keys = jax.vmap(jax.random.split)(keys_arr)  # (B, 2, 2)
    estimate_keys = double_keys[:, 1]  # (B, 2)

    # Use first key for batch-wide exploration.
    explore_key = jax.random.split(keys_arr[0])[0]
    actions = explore(qs, exploration_val, explore_key)
    actions = jnp.reshape(actions, (-1,))
    return RouterOutputs(
        actions=actions,
        q_values=qs,
        exploration_value=exploration_val,
        estimate_keys=estimate_keys,
    )
