"""Typed callable protocols used across ART of PIV internals."""

from typing import Any, NamedTuple, Protocol

import jax.numpy as jnp

from flowgym.common.base import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.types import CachePayload, History


class RouterOutputs(NamedTuple):
    """Structured outputs returned by ART of PIV router execution.

    Attributes:
        actions: Selected estimator indices for each batch element, shape (B,).
        q_values: Router Q-values over estimators, shape (B, num_estimators).
        exploration_value: Per-sample exploration value, shape (B,).
        estimate_keys: RNG keys for sub-estimator sampling, shape (B, 2).
    """

    actions: jnp.ndarray
    q_values: jnp.ndarray
    exploration_value: jnp.ndarray
    estimate_keys: jnp.ndarray


class QEstimateFn(Protocol):
    """Estimate router Q-values from frame pairs and explicit model state."""

    def __call__(
        self,
        prev: jnp.ndarray,
        curr: jnp.ndarray,
        trainable_state: NNEstimatorTrainableState,
        *,
        flow: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Estimate router Q-values from frame pairs.

        Args:
            prev: Previous frame batch, shape ``(B, H, W)``.
            curr: Current frame batch, shape ``(B, H, W)``.
            trainable_state: Router trainable state holding parameters.
            flow: Optional previous flow estimate, shape ``(B, H, W, 2)``.

        Returns:
            Router Q-values, shape ``(B, num_estimators)``.
        """
        ...


class EstimateFn(Protocol):
    """Estimate flow and extras from images/state for ART of PIV internals."""

    def __call__(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        cache_payload: CachePayload | None = None,
    ) -> tuple[History, dict[str, Any]]:
        """Run one estimator call and return updated state plus metrics.

        Args:
            images: Input image batch consumed by the estimator.
            state: Estimator history/state for the batch.
            trainable_state: Estimator trainable state.
            cache_payload: Optional precomputed cache payload for this batch.

        Returns:
            Tuple of ``(new_state, metrics_dict)``.
        """
        ...


class RouterQEstimateFn(Protocol):
    """Estimate router Q-values without exposing trainable state."""

    def __call__(
        self,
        prev: jnp.ndarray,
        curr: jnp.ndarray,
        *,
        flow: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Estimate router Q-values without explicit trainable state input.

        Args:
            prev: Previous frame batch, shape ``(B, H, W)``.
            curr: Current frame batch, shape ``(B, H, W)``.
            flow: Optional previous flow estimate, shape ``(B, H, W, 2)``.

        Returns:
            Router Q-values, shape ``(B, num_estimators)``.
        """
        ...


class RunRouterFn(Protocol):
    """Run the router to produce actions, Q-values, and RNG keys."""

    def __call__(
        self,
        prev: jnp.ndarray,
        curr: jnp.ndarray,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        flow: jnp.ndarray | None = None,
        train: bool = True,
    ) -> RouterOutputs:
        """Run router inference and action selection for a batch.

        Args:
            prev: Previous frame batch, shape ``(B, H, W)``.
            curr: Current frame batch, shape ``(B, H, W)``.
            state: Estimator history/state for the batch.
            trainable_state: Router trainable state.
            flow: Optional previous flow estimate, shape ``(B, H, W, 2)``.
            train: Whether exploration should be active.

        Returns:
            Structured router outputs with actions, Q-values, exploration
            values, and per-sample estimate keys.
        """
        ...


class RunSubEstimatorsFn(Protocol):
    """Protocol for sub-estimator dispatch in ART of PIV inference.

    Note: trainable_state is the main ART of PIV model's state
    (NNEstimatorTrainableState), while sub-estimator states
    are passed separately and can be EstimatorTrainableState.
    """

    def __call__(
        self,
        state: History,
        trainable_state: NNEstimatorTrainableState,
        curr: jnp.ndarray,
        choices: jnp.ndarray,
        extras: dict[str, Any],
        qs: jnp.ndarray,
        exploration_rate: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, Any], dict[str, Any]]:
        """Run selected sub-estimators and aggregate outputs.

        Args:
            state: Estimator history/state for the batch.
            trainable_state: Main ART of PIV trainable state.
            curr: Current frame batch, shape ``(B, H, W)``.
            choices: Selected estimator indices, shape ``(B,)``.
            extras: Mutable extras dictionary propagated through inference.
            qs: Router Q-values, shape ``(B, num_estimators)``.
            exploration_rate: Per-sample exploration values, shape ``(B,)``.

        Returns:
            Tuple ``(flow_batch, updated_extras, metrics_dict)``.
        """
        ...
