"""Module to estimate the flow field from a sequence of images."""

from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from collections.abc import Callable

from flowgym.common.base import (
    Estimator,
    EstimatorTrainableState,
)
from flowgym.types import PRNGKey

from flowgym.utils import DEBUG
from flowgym.flow.postprocess import validate_params, apply_postprocessing
from goggles import get_logger
from goggles.history.types import History

logger = get_logger(__name__)


class FlowFieldEstimator(Estimator):
    """Base class for flow field estimators."""

    velocity_filters: list = ["Manual", "Adaptive", "None"]
    output_types: list = ["uint8", "float32"]

    def __init__(
        self,
        postprocessing_steps: list | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the flow field estimator with a maximum speed.

        Args:
            postprocessing_steps: List of postprocessing steps to apply.
            kwargs: Additional keyword arguments for the base Estimator class.
        """
        if postprocessing_steps is None:
            postprocessing_steps = []

        # Validate postprocessing steps
        self.postprocessing_steps = []
        for step in postprocessing_steps:
            if not isinstance(step, dict):
                raise ValueError(f"Postprocessing step {step} must be a dictionary.")
            if "name" not in step:
                raise ValueError(f"Postprocessing step {step} must have a 'name' key.")
            validate_params(
                step["name"], **{k: v for k, v in step.items() if k != "name"}
            )
            self.postprocessing_steps.append(partial(apply_postprocessing, **step))

        # Add postprocessing to the _estimate method
        self._estimate = self._add_postprocessing(self._estimate)

        super().__init__(**kwargs)

    def _add_postprocessing(
        self,
        fn: Callable,
    ) -> Callable:
        """Add postprocessing steps to the flow field estimator.

        Args:
            fn: The original function to call for estimating the flow field.

        Returns:
            fn with postprocessing applied.
        """

        def call(
            image: jnp.ndarray,
            state: History,
            trainable_state: EstimatorTrainableState,
            key: PRNGKey | None = jax.random.PRNGKey(0),
        ) -> tuple[jnp.ndarray, dict, dict]:
            """Call the flow field estimator."""
            flow_field, extras, metrics = fn(image, state, trainable_state, key)

            if len(self.postprocessing_steps) == 0:
                # If no postprocessing steps are defined, return the flow as is.
                return flow_field, extras, metrics
            valid = jnp.ones_like(flow_field[..., 0], dtype=jnp.bool_)
            for step in self.postprocessing_steps:
                flow_field, valid, state = step(
                    flow=flow_field, valid=valid, state=state
                )
                if DEBUG:
                    logger.debug(
                        f"Flow field shape after filtering: {flow_field.shape}"
                    )
                    n_outliers = jnp.mean(jnp.sum(valid, axis=(1, 2)))
                    logger.debug(f"Average number of outliers per field: {n_outliers}")
            return flow_field, extras, metrics

        return call
