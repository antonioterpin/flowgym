"""Trainable states for flow field estimators."""

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import optax
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from goggles import get_logger
from jax.tree_util import register_pytree_node
from typing_extensions import Self

from flowgym.training.optimizer import build_optimizer_from_config

logger = get_logger(__name__)


class EstimatorTrainableState:
    """Base class for trainable states.

    - Can be instantiated and carried around as an "empty" state.
    - Safe to use for non-trainable estimators (e.g. eval-only models).
    - If you actually try to *train* with this base instance
      (apply_gradients / replace), it will fail loudly.

    Attributes:
        step: Current training step.
        apply_fn: Callable used to compute model outputs.
        params: Trainable model parameters.
        extras: Additional estimator state.
    """

    step: int
    apply_fn: Callable[..., Any] | None
    params: FrozenDict[str, jnp.ndarray]
    extras: FrozenDict[str, jnp.ndarray]

    def __init__(
        self,
        step: int = 0,
        apply_fn: Callable[..., Any] | None = None,
        params: FrozenDict[str, jnp.ndarray] | None = None,
        extras: FrozenDict[str, jnp.ndarray] | None = None,
    ):
        """Initialize the empty trainable state.

        Args:
            step: The current training step.
            apply_fn: The function used for computing estimates.
            params: Parameters of the estimator.
            extras: Additional state that needs to be carried around.
        """
        self.step = step
        self.apply_fn = apply_fn
        self.params = params if params is not None else FrozenDict()
        self.extras = extras if extras is not None else FrozenDict()

    @classmethod
    def create(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Factory for concrete trainable states.

        Args:
            *args: Positional arguments to pass to the constructor.
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            An instance of EstimatorTrainableState.

        Raises:
            RuntimeError: If the method is not implemented by a subclass.
        """
        raise RuntimeError(
            f"{cls.__name__}.create() is not implemented. "
            "You are likely trying to construct a trainable state using the "
            "base EstimatorTrainableState instead of a concrete subclass."
        )

    def apply_gradients(
        self,
        *,
        grads: Mapping[str, jnp.ndarray] | Mapping[str, Any],
        **kwargs: Any,
    ) -> Self:
        """Apply gradients and return a new trainable state.

        Args:
            grads: Gradients to apply.
            **kwargs: Keyword arguments to pass to the constructor.

        Returns:
            An instance of EstimatorTrainableState.

        Raises:
            RuntimeError: If the method is not implemented by a subclass.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support apply_gradients(). "
            "You are likely trying to train a non-trainable estimator or "
            "using the base EstimatorTrainableState instead of a concrete "
            "implementation (e.g. NNEstimatorTrainableState)."
        )


def _estimator_state_flatten(state: EstimatorTrainableState):
    children = (state.params, state.extras, state.step)
    aux_data = (state.apply_fn,)
    return children, aux_data


def _estimator_state_unflatten(aux_data, children):
    params, extras, step = children
    (apply_fn,) = aux_data
    return EstimatorTrainableState(
        step=step,
        apply_fn=apply_fn,
        params=params,
        extras=extras,
    )


# We register EstimatorTrainableState manually as a PyTree node instead of
# using @struct.dataclass. This avoids multiple inheritance conflicts in
# subclasses like NNEstimatorTrainableState which already inherit from
# flax.training.train_state.TrainState (itself a frozen @struct.dataclass).
#
# Manual registration is preferred over decorators here to keep the base class
# as a standard Python class, avoiding complex interaction between multiple
# dataclass decorators and field resolution in the Method Resolution
# Order (MRO).
register_pytree_node(
    EstimatorTrainableState,
    _estimator_state_flatten,
    _estimator_state_unflatten,
)


class NNEstimatorTrainableState(TrainState, EstimatorTrainableState):
    """Trainable state holding params and optimizer state.

    This class extends the TrainState class from Flax and implements the
    EstimatorTrainableState interface.

    Attributes:
        extras: Additional estimator state carried alongside TrainState fields.
    """

    extras: FrozenDict[str, jnp.ndarray] = struct.field(
        default_factory=FrozenDict
    )

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable[..., Any],
        params: FrozenDict[str, jnp.ndarray],
        tx: optax.GradientTransformation,
        extras: Mapping[str, jnp.ndarray] | None = None,
    ) -> Self:
        """Create a new trainable state with initialized optimizer state.

        Args:
            apply_fn: Function to apply the parameters to the model.
            params: Parameters of the model.
            tx: Optimizer transformation.
            extras: Optional extras to include in the state.

        Returns:
            A new instance of EstimatorTrainableState.
        """
        if extras is None:
            extras_fd: FrozenDict[str, jnp.ndarray] = FrozenDict()
        else:
            extras_fd = FrozenDict(extras)

        base = super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            extras=extras_fd,
        )
        return base

    @classmethod
    def from_config(
        cls,
        *,
        apply_fn: Callable[..., Any],
        params: FrozenDict[str, jnp.ndarray],
        optimizer_config: dict[str, Any],
        extras: Mapping[str, jnp.ndarray] | None = None,
    ) -> Self:
        """Create a new trainable state from an optimizer configuration.

        Args:
            apply_fn: Function to apply the parameters to the model.
            params: Model parameters.
            optimizer_config: Configuration dictionary for the optimizer.
            extras: Optional extras to include in the state.

        Returns:
            An instance of EstimatorTrainableState with initialized optimizer
            state.
        """
        tx = build_optimizer_from_config(optimizer_config)
        return cls.create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            extras=extras,
        )
