"""Dummy flow estimator for testing purposes."""

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

from flowgym.flow.base import FlowFieldEstimator
from flowgym.types import (
    RLExperience,
    RLTrainStep,
    SupervisedExperience,
    SupervisedTrainStep,
    TrainStep,
)

if TYPE_CHECKING:
    from flowgym.common.base.trainable_state import NNEstimatorTrainableState


class DummyEstimator(FlowFieldEstimator):
    """Dummy flow field estimator for testing purposes.

    This estimator returns zero flow fields and provides minimal train steps
    for both supervised and RL training modes. Useful for integration testing
    of the training loops without the overhead of real model computation.
    """

    def __init__(
        self,
        train_type: Literal["supervised", "rl"] = "supervised",
        enrichment_marker: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Dummy estimator.

        Args:
            train_type: Training mode, "supervised" or "rl".
            enrichment_marker: If True, enrichment adds a marker to experiences.
            **kwargs: Additional arguments passed to FlowFieldEstimator.
        """
        super().__init__(**kwargs)
        self.train_type = train_type
        self.enrichment_marker = enrichment_marker

    def _estimate(
        self, image: jnp.ndarray, _, __, ___
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Dummy estimation function that returns zeros.

        Args:
            image: The input image.

        Returns:
            Tuple of zero flow field, additional outputs, and metrics.
        """
        return jnp.zeros((*image.shape, 2)), {}, {}

    def create_train_step(self) -> TrainStep:
        """Create a training step function based on train_type.

        Returns:
            SupervisedTrainStep if train_type is "supervised",
            RLTrainStep if train_type is "rl".
        """
        if self.train_type == "rl":
            return self._create_rl_train_step()
        return self._create_supervised_train_step()

    def _create_supervised_train_step(self) -> SupervisedTrainStep:
        """Create a supervised training step that does nothing.

        Returns:
            A training step function that returns zero loss and dummy metrics.
        """

        def train_step(
            trainable_state: "NNEstimatorTrainableState",
            experience: SupervisedExperience,
        ) -> tuple[jnp.ndarray, "NNEstimatorTrainableState", dict]:
            return (
                jnp.array(0.0),
                trainable_state,
                {"dummy_metric": jnp.array(1.0)},
            )

        return train_step

    def _create_rl_train_step(self) -> RLTrainStep:
        """Create an RL training step that does nothing.

        Returns:
            A training step function that returns zero loss and dummy metrics.
        """

        def train_step(
            trainable_state: "NNEstimatorTrainableState",
            target_state: "NNEstimatorTrainableState",
            experience: RLExperience,
        ) -> tuple[jnp.ndarray, "NNEstimatorTrainableState", dict]:
            return (
                jnp.array(0.0),
                trainable_state,
                {"dummy_metric": jnp.array(1.0)},
            )

        return train_step

    def prepare_experience_for_replay(
        self,
        experience: SupervisedExperience,
        trainable_state: "NNEstimatorTrainableState",
    ) -> SupervisedExperience:
        """Prepare an experience for storage in the replay buffer.

        If enrichment_marker is True, adds a marker field to the state dict
        to verify that enrichment was called during testing.

        Args:
            experience: The experience to prepare.
            trainable_state: Current trainable state of the model.

        Returns:
            The prepared experience (may have marker added to state).
        """
        if not self.enrichment_marker:
            return experience

        # Add a marker to the state to verify enrichment was called
        # Use a batched array to match the batch dimension of other state fields
        B = experience.obs[0].shape[0]
        enriched_state = dict(experience.state)
        enriched_state["_enriched"] = jnp.ones((B,), dtype=jnp.bool_)

        return SupervisedExperience(
            state=enriched_state,
            obs=experience.obs,
            ground_truth=experience.ground_truth,
        )
