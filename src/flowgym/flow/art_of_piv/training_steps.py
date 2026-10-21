"""Training step factories for ART of PIV."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp
import optax

from flowgym.types import CachePayload

if TYPE_CHECKING:
    from flowgym.common.base import (
        EstimatorTrainableState,
        NNEstimatorTrainableState,
    )
    from flowgym.flow.art_of_piv.types import (
        EstimateFn,
        QEstimateFn,
        RunRouterFn,
    )
    from flowgym.training.losses import LossFn
    from flowgym.training.target_transforms import TargetTransform
    from flowgym.types import (
        RLExperience,
        RLTrainStep,
        SupervisedExperience,
        SupervisedTrainStep,
        TrainStep,
    )


def create_supervised_train_step(
    *,
    # Callables (injected bound methods or plain functions)
    q_estimate_fn: QEstimateFn,
    estimate_fn: EstimateFn,
    run_router_fn: RunRouterFn,
    loss_fn: LossFn,
    target_transform: TargetTransform,
    schedule: optax.Schedule,
    # Config values
    training_mode: Literal["bandits", "regression", "classification"],
    use_temporal_propagation: bool,
    num_estimators: int,
    estimator_configs: list[dict[str, Any]],
    estimator_fns: tuple[EstimateFn, ...],
    estimator_states: tuple[EstimatorTrainableState, ...] | None = None,
    target_metric: str = "epe",
) -> SupervisedTrainStep:
    """Create a supervised training step for multi-armed bandit mode.

    Args:
        q_estimate_fn: Function to compute Q-values. Signature: (prev, curr,
            trainable_state, flow=None) -> (B, num_estimators).
        estimate_fn: Callable that runs the full estimation pipeline. Signature:
            (images, state, trainable_state, cache_payload=None) -> (state,
            metrics).
        run_router_fn: Function to run the router. Signature: (prev, curr,
            state, trainable_state, ...) -> (actions, qs, expl, keys).
        loss_fn: Loss function (e.g. L2, cross-entropy).
        target_transform: Transform applied to raw EPE before negating to get
            reward.
        schedule: Exploration schedule for logging.
        training_mode: One of "bandits", "regression", "classification".
        use_temporal_propagation: Whether to pass previous flow to Q-network.
        num_estimators: Number of sub-models.
        estimator_configs: List of sub-model config dicts (for metric names).
        estimator_fns: Tuple of sub-model callables (for regression mode
            fallback).
        estimator_states: Optional tuple of sub-model trainable states aligned
            with `estimator_fns`. When not provided, `trainable_state` is used
            as a fallback for backwards compatibility.
        target_metric: Score the cached ``epe_all`` holds; anything but EPE
            can only come from the cache, since the uncached path computes
            EPE against ground truth.

    Returns:
        The supervised training step closure.
    """

    def train_step(
        trainable_state: NNEstimatorTrainableState,
        experience: SupervisedExperience,
    ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
        """Train step for supervised/MAB mode.

        Args:
            trainable_state: Current trainable state of the model.
            experience: Supervised experience with state, obs, ground_truth.

        Returns:
            The computed loss.
            The updated trainable state.
            A metrics dictionary.

        Raises:
            ValueError: If ``target_metric`` is not EPE and the batch has no
                cached labels.
        """
        # Backward-compat: coerce legacy dict-based cache_payload.
        if isinstance(experience.cache_payload, dict):
            experience = replace(
                experience,
                cache_payload=CachePayload.from_enrich_result(
                    experience.cache_payload
                ),
            )

        state = experience.state
        images1, images2 = experience.obs
        ground_truth = experience.ground_truth  # (B, H, W, 2)

        # Check if we have cached EPEs for all sub-models.
        has_cached_epe = (
            experience.cache_payload is not None
            and experience.cache_payload.epe_all is not None
        )
        if not has_cached_epe and target_metric != "epe":
            raise ValueError(
                f"target_metric={target_metric!r} needs cached labels; "
                "configure caching for the training dataset."
            )

        # Use router method
        def run_router(s):
            return run_router_fn(images1, images2, s, trainable_state)[0]

        def fetch_reward_from_cache(
            actions: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Fetch rewards for given actions from cache payload.

            Args:
                actions: Array of actions to fetch rewards for.

            Returns:
                Tuple of reward and EPE raw values.
            """
            assert experience.cache_payload is not None
            assert experience.cache_payload.epe_all is not None
            epe_all_shortcut = experience.cache_payload.epe_all
            B = epe_all_shortcut.shape[0]
            actions_idx = jnp.reshape(actions, (B, 1))
            epe_raw = jnp.take_along_axis(
                epe_all_shortcut, actions_idx, axis=-1
            ).reshape(B)
            reward = -target_transform(epe_raw)
            return reward, epe_raw

        def run_inference_full(state):
            # Full path: run selecting estimators and then compute reward
            state, _ = estimate_fn(
                images2,
                state,
                trainable_state,
                cache_payload=experience.cache_payload,
            )
            actions = jnp.ravel(state["choices"])
            estimates = state["estimates"][:, -1]

            epe_computed = jnp.sqrt(
                jnp.sum((estimates - ground_truth) ** 2, axis=-1)
            )
            epe_raw = jnp.mean(epe_computed, axis=(1, 2))
            reward = -target_transform(epe_raw)
            return actions, reward, epe_raw

        # Conditional path selection
        if training_mode in ("regression", "classification"):
            if has_cached_epe:
                # Reward for all actions from cache
                assert experience.cache_payload is not None
                epe_all = jnp.asarray(
                    experience.cache_payload.epe_all
                )  # (B, num_estimators)
                epe_mask = experience.cache_payload.epe_mask
                if epe_mask is not None:
                    epe_mask = jnp.asarray(epe_mask, dtype=bool)
                    epe_all = jnp.asarray(jnp.where(epe_mask, epe_all, jnp.inf))
                reward = -target_transform(epe_all)
                # For logging/metrics, we still need selected action
                actions = run_router(state)
            else:
                # Reward for all actions by running all sub-estimators
                # Run all sub-estimators
                # We use a helper function to avoid repeating switch logic
                def run_all(curr_img, s):
                    def get_epe(i):
                        new_s = jax.tree.map(lambda x: x[jnp.newaxis, ...], s)
                        new_curr = curr_img[jnp.newaxis, ...]
                        sub_estimator_state = (
                            estimator_states[i]
                            if estimator_states is not None
                            else trainable_state
                        )
                        try:
                            _, metrics = estimator_fns[i](
                                new_curr,
                                new_s,
                                sub_estimator_state,
                                None,
                            )
                        except TypeError:
                            # Backward-compat for legacy tuple-call mocks.
                            legacy_fn = cast(Any, estimator_fns[i])
                            _, metrics = legacy_fn((new_curr, new_s))
                        # Most estimators return epe in metrics
                        return metrics.get("epe", metrics.get("epe_mean", 0.0))

                    return jnp.array(
                        [get_epe(i) for i in range(num_estimators)]
                    )

                # vmap over batch
                epe_all = jax.vmap(run_all)(images2, state)
                reward = -target_transform(epe_all)

                # For logging/metrics, we still need selected action
                actions = run_router(state)

            # epe_all guaranteed in regression/classification
            assert epe_all is not None
            chosen_epe = jnp.take_along_axis(
                epe_all, actions[:, None], axis=1
            ).squeeze(-1)  # (B,)
            best_epe = jnp.min(epe_all, axis=1)  # (B,)
            regret = chosen_epe - best_epe  # (B,)

            regret_mean = jnp.mean(regret)
            regret_p90 = jnp.quantile(regret, 0.9)
            chosen_epe_mean = jnp.mean(chosen_epe)
            best_epe_mean = jnp.mean(best_epe)
        else:
            # Default behavior: bandits mode (one action)
            if has_cached_epe:
                actions = run_router(state)
                reward, chosen_epe = fetch_reward_from_cache(actions)
            else:
                actions, reward, chosen_epe = run_inference_full(state)
            chosen_epe_mean = jnp.mean(chosen_epe)

        # Define variables needed for loss and metrics
        prev, curr = images1, images2

        # Transition to parameters training
        reward = cast(jax.Array, reward)

        # Get flow from previous timestep if needed
        prev_flow = (
            state["estimates"][:, -1, ...] if use_temporal_propagation else None
        )

        params = trainable_state.params

        def loss_fn_inner(params: jnp.ndarray) -> jnp.ndarray:
            """Compute loss.

            Args:
                params: Model parameters.

            Returns:
                Computed loss value.
            """
            tmp_ts = trainable_state.replace(params=params)

            if use_temporal_propagation and prev_flow is not None:
                q_values = q_estimate_fn(prev, curr, tmp_ts, flow=prev_flow)
            else:
                q_values = q_estimate_fn(prev, curr, tmp_ts)

            if training_mode == "regression":
                # Regression on all actions
                # q_values: (B, num_estimators)
                # target (reward): (B, num_estimators)
                loss = loss_fn(q_values, reward)
            elif training_mode == "classification":
                # Classification: predict the best arm index
                # q_values shape: (B, num_estimators) - raw logits
                # best_actions shape: (B,) - integer class labels
                # (0 to num_estimators-1)
                best_actions = jnp.argmin(epe_all, axis=-1)
                # Loss function (e.g.,
                # softmax_cross_entropy_with_integer_labels)
                # will internally apply softmax to q_values and
                # compute cross-entropy
                loss = loss_fn(q_values, best_actions)
            else:
                # Bandits: regression on selected action only
                one_hot_actions = jax.nn.one_hot(
                    actions, num_classes=q_values.shape[-1]
                )  # (B, num_estimators)
                q_sa = jnp.sum(one_hot_actions * q_values, axis=-1)
                target = cast(jax.Array, reward)
                loss = loss_fn(q_sa, target)
            return loss

        loss, grads = jax.value_and_grad(loss_fn_inner)(params)
        trainable_state = trainable_state.apply_gradients(grads=grads)
        trainable_state = trainable_state.replace(
            extras=jax.tree.map(lambda x: x + 1, trainable_state.extras)
        )

        # For classification mode, recompute actions with updated model (greedy)
        # This ensures accuracy metrics reflect the current model's performance
        if training_mode == "classification":
            # Get updated Q-values with new parameters
            if use_temporal_propagation and prev_flow is not None:
                updated_q_values = q_estimate_fn(
                    prev, curr, trainable_state, flow=prev_flow
                )
            else:
                updated_q_values = q_estimate_fn(prev, curr, trainable_state)
            # Use greedy selection (argmax) for metrics, not exploration
            actions = jnp.argmax(updated_q_values, axis=-1)

            # Recompute EPE metrics with updated actions
            chosen_epe = jnp.take_along_axis(
                epe_all, actions[:, None], axis=1
            ).squeeze(-1)  # (B,)
            best_epe = jnp.min(epe_all, axis=1)  # (B,)
            regret = chosen_epe - best_epe  # (B,)
            regret_mean = jnp.mean(regret)
            regret_p90 = jnp.quantile(regret, 0.9)
            chosen_epe_mean = jnp.mean(chosen_epe)
            best_epe_mean = jnp.mean(best_epe)

        # Compute model choice distribution
        choice_counts = jnp.bincount(actions, length=num_estimators)

        # Compute current exploration value (temperature/epsilon)
        # Use the global step from the first batch item
        train_extras = cast(dict, trainable_state.extras)
        steps_arr = jnp.asarray(train_extras["steps"])
        current_step = steps_arr[0, 0].astype(jnp.int32)
        exploration_val = schedule(current_step)

        # Extract learning rate from optimizer hyperparams if available
        learning_rate = 0.0
        hp = getattr(trainable_state.opt_state, "hyperparams", None)
        if hp is not None and isinstance(hp, dict) and "learning_rate" in hp:
            lr_val = hp["learning_rate"]
            # If it's a schedule, call it with current step
            if callable(lr_val):
                learning_rate = lr_val(current_step)
            else:
                learning_rate = lr_val

        # q_values shape is (B, num_estimators)
        if use_temporal_propagation and prev_flow is not None:
            q_vals_batch = q_estimate_fn(
                prev, curr, trainable_state, flow=prev_flow
            )
        else:
            q_vals_batch = q_estimate_fn(prev, curr, trainable_state)

        if training_mode in ("regression", "classification"):
            chosen_reward = jnp.take_along_axis(
                reward, actions[:, None], axis=1
            ).squeeze(-1)
            reward_mean = jnp.mean(chosen_reward)
        else:
            reward_mean = jnp.mean(reward)

        metrics = {
            "reward_mean": reward_mean,
            "epe/raw_mean": chosen_epe_mean,
            "target/mean": -jnp.mean(reward),
            "exploration_val": exploration_val,
            "learning_rate": learning_rate,
            "loss": loss,
        }

        # Choice Entropy: how uniform/skewed is the model selection?
        probs = choice_counts / jnp.sum(choice_counts)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
        metrics["choices/entropy"] = entropy

        # Optimal Selection Accuracy (only if epe_all is available)
        if has_cached_epe and experience.cache_payload is not None:
            epe_all_data = experience.cache_payload.epe_all
            assert epe_all_data is not None
            epe_mask_data = experience.cache_payload.epe_mask
            if epe_mask_data is not None:
                epe_mask_data = jnp.asarray(epe_mask_data, dtype=bool)
                epe_all_data_masked = jnp.asarray(
                    jnp.where(epe_mask_data, epe_all_data, jnp.inf)
                )
                row_has_any = jnp.any(epe_mask_data, axis=-1)
                best_actions = jnp.argmin(epe_all_data_masked, axis=-1)
                best_actions = jnp.where(row_has_any, best_actions, 0)
                valid_f = row_has_any.astype(jnp.float32)
                correct = (actions == best_actions).astype(
                    jnp.float32
                ) * valid_f
                accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(valid_f), 1.0)
            else:
                best_actions = jnp.argmin(epe_all_data, axis=-1)
                accuracy = jnp.mean(actions == best_actions)
            metrics["choices/accuracy"] = accuracy
        elif (
            training_mode in ("regression", "classification")
            and not has_cached_epe
        ):
            # If we computed epe_all during training, use it for accuracy
            assert epe_all is not None
            best_actions = jnp.argmin(epe_all, axis=-1)
            accuracy = jnp.mean(actions == best_actions)
            metrics["choices/accuracy"] = accuracy

        # Add counts and Q-value stats for each sub-model
        for i, cfg in enumerate(estimator_configs):
            name = cfg.get("name", f"model_{i}")
            metrics[f"choices/{name}"] = choice_counts[i]
            metrics[f"q_values/mean_{name}"] = jnp.mean(q_vals_batch[:, i])
            metrics[f"q_values/max_{name}"] = jnp.max(q_vals_batch[:, i])

        if training_mode in ("regression", "classification"):
            metrics["regret/raw_mean"] = regret_mean
            metrics["regret/raw_p90"] = regret_p90
            metrics["epe/raw_best_mean"] = best_epe_mean

        return loss, trainable_state, metrics

    return train_step


def create_rl_train_step(
    *,
    q_estimate_fn: QEstimateFn,
    loss_fn: LossFn,
    use_temporal_propagation: bool,
    gamma: float,
) -> RLTrainStep:
    """Create a reinforcement learning training step.

    Args:
        q_estimate_fn: Q-value estimation function.
        loss_fn: Loss function.
        use_temporal_propagation: Whether to pass previous flow to Q-network.
        gamma: Discount factor (must be 0; >0 raises NotImplementedError).

    Returns:
        The RL training step closure.
    """

    def train_step(
        trainable_state: NNEstimatorTrainableState,
        target_state: NNEstimatorTrainableState,
        experience: RLExperience,
    ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
        """Train step for RL mode.

        Args:
            trainable_state: Current trainable state of the model.
            target_state: Target network state for stable training.
            experience: RL experience with state, reward, obs, old_obs,
                old_flow.

        Returns:
            The computed loss.
            The updated trainable state.
            A metrics dictionary.
        """
        state = experience.state
        reward = experience.reward
        prev_obs = experience.old_obs
        prev_flow = experience.old_flow_estimate

        prev, curr = prev_obs
        actions = jnp.squeeze(state["choices"], axis=-1)  # (B,)

        params = trainable_state.params

        def loss_fn_inner(params: jnp.ndarray) -> jnp.ndarray:
            """Compute DQN loss.

            Args:
                params: Model parameters.

            Returns:
                Computed loss value.

            Raises:
                NotImplementedError: When gamma > 0.
            """
            tmp_ts = trainable_state.replace(params=params)

            if use_temporal_propagation:
                q_values = q_estimate_fn(prev, curr, tmp_ts, flow=prev_flow)
            else:
                q_values = q_estimate_fn(prev, curr, tmp_ts)

            one_hot_actions = jax.nn.one_hot(
                actions, num_classes=q_values.shape[-1]
            )  # (B, num_estimators)
            q_sa = jnp.sum(one_hot_actions * q_values, axis=-1)

            # For gamma > 0, would compute target with target_state
            if gamma > 0:
                raise NotImplementedError(
                    "Temporal propagation with gamma > 0 not implemented."
                )

            target = reward
            loss = loss_fn(q_sa, target)
            return loss

        loss, grads = jax.value_and_grad(loss_fn_inner)(params)
        trainable_state = trainable_state.apply_gradients(grads=grads)
        trainable_state = trainable_state.replace(
            extras=jax.tree.map(lambda x: x + 1, trainable_state.extras)
        )

        metrics = {"reward_mean": jnp.mean(reward)}

        return loss, trainable_state, metrics

    return train_step


def create_train_step(
    *,
    train_type: Literal["supervised", "rl"],
    create_supervised_fn: Callable[[], SupervisedTrainStep],
    create_rl_fn: Callable[[], RLTrainStep],
) -> TrainStep:
    """Route to supervised or RL training step.

    Args:
        train_type: Either "supervised" or "rl".
        create_supervised_fn: Factory for the supervised step.
        create_rl_fn: Factory for the RL step.

    Returns:
        The appropriate training step.

    Raises:
        ValueError: When train_type is neither "supervised" nor "rl".
    """
    if train_type == "supervised":
        return create_supervised_fn()
    elif train_type == "rl":
        return create_rl_fn()
    else:
        raise ValueError(
            f"Unknown train_type: {train_type}. Use 'supervised' or 'rl'."
        )


def prepare_experience_for_replay(
    experience: SupervisedExperience,
    trainable_state: EstimatorTrainableState,
    estimate_fn: EstimateFn,
) -> SupervisedExperience:
    """Enrich experience with computed estimates before storing in replay.

    Args:
        experience: The experience with state, obs, ground_truth.
        trainable_state: Current trainable state.
        estimate_fn: Callable that runs estimation.
            Signature: (images, state, trainable_state) -> (state, metrics).

    Returns:
        Experience with enriched state.
    """

    # Run inference to compute estimates and choices
    _, images2 = experience.obs
    state = experience.state

    # We call estimate_fn directly. Since it's JITted (via jit: true in config),
    # this will be the same computation as in the train step.
    updated_state, _ = estimate_fn(images2, state, trainable_state)

    return replace(experience, state=updated_state)
