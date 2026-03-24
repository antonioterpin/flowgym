"""Module to estimate the flow field from a sequence of images."""

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import jax.numpy as jnp
from goggles import get_logger
from goggles.history.types import History

from flowgym.common.base import (
    Estimator,
    EstimatorTrainableState,
)
from flowgym.flow.postprocess import apply_postprocessing, validate_params
from flowgym.utils import DEBUG

logger = get_logger(__name__)
OUTLIER_REJECTION_STEPS = {
    "constant_threshold_filter",
    "adaptive_global_filter",
    "adaptive_local_filter",
    "universal_median_test",
}


class FlowFieldEstimator(Estimator):
    """Base class for flow field estimators.

    Attributes:
        velocity_filters: List of available velocity filter types.
        output_types: List of supported output data types.
    """

    velocity_filters: ClassVar[list] = ["Manual", "Adaptive", "None"]
    output_types: ClassVar[list] = ["uint8", "float32"]

    def __init__(
        self,
        postprocessing_steps: list | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the flow field estimator.

        Args:
            postprocessing_steps: List of postprocessing steps to apply.
            **kwargs: Additional keyword arguments for the base Estimator class.

        Raises:
            ValueError: If postprocessing steps are invalid.
        """
        if postprocessing_steps is None:
            postprocessing_steps = []

        # Validate postprocessing steps
        self.postprocessing_steps = []
        for step in postprocessing_steps:
            if not isinstance(step, dict):
                raise ValueError(
                    f"Postprocessing step {step} must be a dictionary."
                )
            if "name" not in step:
                raise ValueError(
                    f"Postprocessing step {step} must have a 'name' key."
                )
            validate_params(
                step["name"], **{k: v for k, v in step.items() if k != "name"}
            )
            self.postprocessing_steps.append(
                partial(apply_postprocessing, **step)
            )

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
            extras: dict | None = None,
        ) -> tuple[jnp.ndarray, dict, dict]:
            """Call the flow field estimator with postprocessing.

            Args:
                image: The input image.
                state: The state object containing historical images.
                trainable_state: Current trainable state of the model.
                extras: Additional extras dictionary.

            Returns:
                Tuple of flow field, extras, and metrics.
            """
            if extras is None:
                extras = {}
            flow_field, out_extras, metrics = fn(
                image, state, trainable_state, extras
            )

            if len(self.postprocessing_steps) == 0:
                # If no postprocessing steps are defined, return the flow as is.
                return flow_field, out_extras, metrics
            valid = jnp.ones_like(flow_field[..., 0], dtype=jnp.bool_)
            combined_rejected_mask: jnp.ndarray | None = None
            for idx, step in enumerate(self.postprocessing_steps):
                step_name = step.keywords.get("name", f"step_{idx}")
                flow_field, valid, state = step(
                    flow=flow_field, valid=valid, state=state, trainable_state=trainable_state
                )
                if valid is not None:
                    # NOTE: validation steps return a boolean mask where
                    # True marks valid/inlier pixels, False marks rejected/outlier.
                    outlier_frac = 1.0 - jnp.mean(
                        valid.astype(jnp.float32), axis=(1, 2)
                    )
                    metric_prefix = (
                        f"postprocess_{step_name}_{idx}_outlier_percentage"
                    )
                    metrics[metric_prefix] = outlier_frac * 100.0

                    if step_name in OUTLIER_REJECTION_STEPS:
                        metrics[
                            f"postprocess_{step_name}_{idx}_rejected_percentage"
                        ] = (outlier_frac * 100.0)
                        if combined_rejected_mask is None:
                            combined_rejected_mask = jnp.logical_not(valid.astype(jnp.bool_))
                        else:
                            combined_rejected_mask = jnp.logical_or(
                                combined_rejected_mask, jnp.logical_not(valid.astype(jnp.bool_))
                            )

                if DEBUG:
                    logger.debug(
                        f"Flow field shape after filtering: {flow_field.shape}"
                    )
                    n_outliers = jnp.mean(jnp.sum(valid, axis=(1, 2)))
                    outlier_pct = jnp.mean(
                        jnp.mean(valid.astype(jnp.float32), axis=(1, 2)) * 100.0
                    )
                    logger.debug(
                        f"Average number of outliers per field: {n_outliers}"
                    )
                    logger.debug(
                        f"Average outlier percentage after "
                        f"{step_name}: {outlier_pct:.4f}%"
                    )
            if combined_rejected_mask is not None:
                metrics["postprocess_combined_rejected_mask"] = (
                    combined_rejected_mask
                )
            return flow_field, out_extras, metrics

        return call
