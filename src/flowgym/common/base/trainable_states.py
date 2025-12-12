"""Trainable states for flow field estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jax import tree_util, numpy as jnp
from flax.core.frozen_dict import FrozenDict
import optax

from goggles import get_logger

from flowgym.training.optimizer import build_optimizer_from_config

logger = get_logger(__name__)


@tree_util.register_pytree_node_class
@dataclass
class EstimatorTrainableState:
    """Trainable state of the flow field estimator."""

    def tree_flatten(self):
        """Flatten the state for JAX tree utilities."""
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the state from JAX tree utilities."""
        return cls(*children)
    
    
@tree_util.register_pytree_node_class
@dataclass
class NNEstimatorTrainableState(EstimatorTrainableState):
    """Trainable state holding params and optimizer state, with static tx."""

    params: FrozenDict[str, jnp.ndarray]
    opt_state: optax.OptState
    tx: optax.GradientTransformation
    extras: FrozenDict[str, jnp.ndarray]

    @classmethod
    def create(
        cls, 
        params: FrozenDict[str, jnp.ndarray],
        tx: optax.GradientTransformation, 
        extras: Mapping[str, jnp.ndarray] | None = None
    ) -> NNEstimatorTrainableState:
        """Create a new trainable state with initialized optimizer state.

        Args:
            params: Parameters of the model.
            tx: Optimizer transformation.

        Returns:
            A new instance of NNEstimatorTrainableState.
        """
        if extras is None:
            extras_fd: FrozenDict[str, jnp.ndarray] = FrozenDict()
        else:
            extras_fd: FrozenDict[str, jnp.ndarray] = FrozenDict(extras)
        opt_state = tx.init(params)
        return cls(params=params, opt_state=opt_state, tx=tx, extras=extras_fd)

    def replace(self, **kwargs: Any) -> NNEstimatorTrainableState:
        """Return a new state with specified fields replaced."""
        return NNEstimatorTrainableState(
            params=kwargs.get("params", self.params),
            opt_state=kwargs.get("opt_state", self.opt_state),
            tx=kwargs.get("tx", self.tx),
            extras=kwargs.get("extras", self.extras),
        )

    def tree_flatten(
        self
        ) -> tuple[
            tuple[
                FrozenDict[str, jnp.ndarray],
                optax.OptState,
                FrozenDict[str, jnp.ndarray]
                ], optax.GradientTransformation]:
        """Flatten the state for JAX tree utilities.

        Returns:
            A tuple containing the flattened state and auxiliary data.
        """
        # Only params, opt_state, and extras are leaves; tx is static
        children = (self.params, self.opt_state, self.extras)
        aux_data = self.tx
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: optax.GradientTransformation,
        children: tuple[
            FrozenDict[str, jnp.ndarray], optax.OptState, FrozenDict[str, jnp.ndarray]]
        ) -> NNEstimatorTrainableState:
        """Unflatten the state from JAX tree utilities.

        Args:
            aux_data: Auxiliary data (static tx).
            children: Flattened state (params, opt_state, extras).
        """
        params, opt_state, extras = children
        tx = aux_data
        return cls(params=params, opt_state=opt_state, tx=tx, extras=extras)

    def apply_gradients(self, grads):
        """Apply gradients to the parameters and update the optimizer state.

        Args:
            grads: Gradients to apply.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        # Use replace to keep tx static and update params + opt_state
        return self.replace(params=new_params, opt_state=new_opt_state)
    
    @classmethod
    def from_config(
        cls,
        params: FrozenDict,
        optimizer_config: dict[str, Any],
        extras: Mapping[str, jnp.ndarray] | None = None,
    ) -> "NNEstimatorTrainableState":
        """Create a new trainable state from an optimizer configuration.
        
        Args:
            params: Model parameters.
            optimizer_config: Configuration dictionary for the optimizer.
            extras: Optional extras to include in the state.
            
        Returns:
            An instance of NNEstimatorTrainableState with initialized optimizer state.
        """
        tx = build_optimizer_from_config(optimizer_config)
        return cls.create(params=params, tx=tx, extras=extras)