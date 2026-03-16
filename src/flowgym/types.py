"""Types used in the flowgym package."""

from dataclasses import dataclass, field, fields
from typing import Any, Protocol, TypeAlias, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from goggles.history.types import History
from typing_extensions import Self

from flowgym.common.base.trainable_state import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)

PRNGKey: TypeAlias = jnp.ndarray
ExperimentParams: TypeAlias = dict[
    str, jnp.ndarray | float | int | bool | str | dict
]
Observation: TypeAlias = tuple[jnp.ndarray, jnp.ndarray]

_ARRAY_FIELD_NAMES = ("epe", "relative_epe", "epe_all", "epe_mask", "estimates")

# Keys in enrich() result dicts that map to named CachePayload fields.
_KNOWN_ARRAY_KEYS = {
    "epe",
    "errors",
    "relative_epe",
    "epe_all",
    "epe_mask",
    "estimates",
}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CachePayload:
    """Typed container for cached data passed to estimator training/eval steps.

    This is a pure data container holding precomputed batch-level information.
    Individual estimators decide how to use this data via estimate_extras.

    Attributes:
        has_precomputed_errors: Whether per-sample EPE data is available.
        epe: Per-sample EPE (B,) for regular estimators.
        relative_epe: Relative EPE (B,) for regular estimators.
        epe_all: Per-sub-model EPE (B, K) for meta-estimators (ArtOfPIV).
        epe_mask: Validity mask (B, K) aligned with epe_all.
        estimates: Pre-computed flow estimates (B, ...).
        extras: Opaque estimator-specific data (e.g., candidate flows for
            the critic).
    """

    has_precomputed_errors: bool = False
    epe: np.ndarray | None = None
    relative_epe: np.ndarray | None = None
    epe_all: np.ndarray | None = None
    epe_mask: np.ndarray | None = None
    estimates: jnp.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_enrich_result(
        cls,
        result: dict[str, Any] | None,
    ) -> Self:
        """Construct from an enrichment result dictionary.

        Args:
            result: The dict returned by `training.caching.enrich_batch()` or
                `Estimator.enrich(...)`, or None.

        Returns:
            A CachePayload with the recognised fields populated.
        """
        if result is None:
            return cls()
        epe = result.get("epe")
        if epe is None:
            epe = result.get("errors")
        return cls(
            has_precomputed_errors=epe is not None,
            epe=epe,
            relative_epe=result.get("relative_epe"),
            epe_all=result.get("epe_all"),
            epe_mask=result.get("epe_mask"),
            estimates=result.get("estimates"),
            extras={
                k: v for k, v in result.items() if k not in _KNOWN_ARRAY_KEYS
            },
        )

    # ------------------------------------------------------------------
    # JAX pytree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self) -> tuple[list[Any], tuple]:
        """Flatten for JAX pytree registration.

        Returns:
            Tuple of (children list, auxiliary data).
        """
        present = tuple(
            name
            for name in _ARRAY_FIELD_NAMES
            if getattr(self, name) is not None
        )
        children: list[Any] = [getattr(self, name) for name in present]
        children.append(self.extras)
        aux = (self.has_precomputed_errors, present)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: list[Any]) -> Self:
        """Reconstruct from flattened pytree.

        Args:
            aux: Auxiliary data.
            children: List of fields to reconstruct the dataclass.

        Returns:
            An instance of CachePayload reconstructed from the fields.
        """
        has_precomputed_errors, present = aux
        kwargs: dict[str, Any] = {name: None for name in _ARRAY_FIELD_NAMES}
        for name, val in zip(present, children[:-1], strict=True):
            kwargs[name] = val
        kwargs["extras"] = children[-1]
        return cls(
            has_precomputed_errors=has_precomputed_errors,
            **kwargs,
        )


@jax.tree_util.register_pytree_node_class
@dataclass
class RLExperience:
    """Batch of experiences for RL training.

    Stores transitions from environment interaction for learning.

    Attributes:
        state: Estimator internal state (History) at experience time.
        reward: Reward signal from the environment.
        obs: Current observation (image tuple) that led to reward.
        old_obs: Previous observation for contextual continuity.
        old_flow_estimate: Previous flow estimate for consistency.
    """

    state: History
    reward: jnp.ndarray
    obs: Observation
    old_obs: Observation
    old_flow_estimate: jnp.ndarray

    def tree_flatten(self) -> tuple[list[Any], None]:
        """Flatten for JAX PyTree compatibility.

        Returns:
            Tuple of (children list, None).
        """
        children = [getattr(self, f.name) for f in fields(self)]
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: None, children: list[Any]) -> Self:
        """Reconstruct RLExperience from flattened fields.

        Args:
            aux: Auxiliary data (not used).
            children: Dataclass fields.

        Returns:
            RLExperience instance.
        """
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class SupervisedExperience:
    """Batch of experiences for supervised training.

    Useful for contextual bandits or training on static datasets with slow
    estimators. Allows replaying past inferences for efficiency.

    Attributes:
        state: Estimator internal state (History) for this sample.
        obs: Observation (image tuple) for this sample.
        ground_truth: Target estimate (flow or density).
        cache_payload: Optional training data (e.g., candidate flows).
    """

    state: History
    obs: Observation
    ground_truth: jnp.ndarray
    cache_payload: CachePayload | None = None

    def tree_flatten(self) -> tuple[list[Any], None]:
        """Flattens the SupervisedExperience dataclass.

        Returns:
            A tuple containing a list of the dataclass fields and None.
        """
        children = [getattr(self, f.name) for f in fields(self)]
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: None, children: list[Any]) -> Self:
        """Reconstruct SupervisedExperience from flattened fields.

        Args:
            aux: Auxiliary data (not used).
            children: Dataclass fields.

        Returns:
            SupervisedExperience instance.
        """
        return cls(*children)


@runtime_checkable
class SupervisedTrainStep(Protocol):
    """Protocol for a supervised training step."""

    def __call__(
        self,
        trainable_state: NNEstimatorTrainableState,
        experience: SupervisedExperience,
    ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
        """Perform a single supervised training step.

        Args:
            trainable_state: Current trainable state.
            experience: Training experience.

        Returns:
            Loss, updated trainable state, and metrics dict.
        """
        ...


@runtime_checkable
class RLTrainStep(Protocol):
    """Protocol for a reinforcement learning training step."""

    def __call__(
        self,
        trainable_state: NNEstimatorTrainableState,
        target_state: NNEstimatorTrainableState,
        experience: RLExperience,
    ) -> tuple[jnp.ndarray, NNEstimatorTrainableState, dict]:
        """Perform a single RL training step.

        Args:
            trainable_state: Current trainable state.
            target_state: Target network state.
            experience: RL training experience.

        Returns:
            Loss, updated trainable state, and metrics dict.
        """
        ...


# Union type for flexible train step return type
TrainStep: TypeAlias = SupervisedTrainStep | RLTrainStep


class CompiledCreateStateFn(Protocol):
    """Protocol for the compiled state creation function.

    NOTE: This protocol is for the create-state callable returned by
    `make_estimator`/`compile_model`, not `Estimator.create_state`.
    `compile_model` binds `estimates` and history sizes ahead of time.
    """

    def __call__(
        self,
        images: jnp.ndarray,
        rng: PRNGKey,
    ) -> History:
        """Create the state.

        Args:
            images: Input images.
            rng: Random number generator key.

        Returns:
            Estimator state.
        """
        ...


class CompiledComputeEstimateFn(Protocol):
    """Protocol for the flow estimate computation function."""

    def __call__(
        self,
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        cache_payload: CachePayload | None = None,
    ) -> tuple[History, dict]:
        """Compute the flow estimate.

        Args:
            images: Input images.
            state: Current estimator state.
            trainable_state: Trainable model state.
            cache_payload: Optional precomputed data.

        Returns:
            Updated state and metrics dict.
        """
        ...
