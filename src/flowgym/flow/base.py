"""Module to estimate the flow field from a sequence of images."""

import copy
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from goggles import get_logger
from goggles.history.types import History
from synthpix.types import SynthpixBatch

from flowgym.common.base import (
    Estimator,
    EstimatorTrainableState,
)
from flowgym.flow.postprocess import (
    apply_postprocessing,
    is_jittable_step,
    is_outlier_rejection_step,
    validate_params,
)
from flowgym.flow.quality import correlation_peak_quality
from flowgym.utils import DEBUG

logger = get_logger(__name__)


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

        self._postprocessing_config = copy.deepcopy(postprocessing_steps)

        # Add postprocessing to the _estimate method
        self._estimate = self._add_postprocessing(self._estimate)

        super().__init__(**kwargs)
        # A class-level JIT with static self roots every estimator it sees.
        self._compute_cache_payload_jit = jax.jit(self._compute_cache_payload)

    def _processing_config(self) -> dict[str, Any]:
        """Non-empty pre/post-processing steps that change the estimate.

        Returns:
            Step lists keyed by name, to hash into the cache id.
        """
        config = super()._processing_config()
        steps = getattr(self, "_postprocessing_config", None)
        if steps:
            config["postprocessing_steps"] = steps
        return config

    def enrich(
        self,
        batch: SynthpixBatch,
        miss_idxs: np.ndarray,
        trainable_state: EstimatorTrainableState | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Compute the per-sample cache payload for missing keys.

        Runs the estimator on the missing samples and scores the estimate
        with the ground-truth-free ``corr16_peak`` and, when the batch
        carries ground truth, its mean endpoint error and mean relative
        error. A cache over real images (no ground truth) can therefore only
        store ``corr16_peak``.

        Args:
            batch: The batch of data as returned by the `SynthpixBatch` sampler.
            miss_idxs: Indices of samples in the batch that are missing from
                cache.
            trainable_state: State used for inference; when None a fresh
                trainable state is created.
            **kwargs: Unused cache context (e.g. ``cache_manager``).

        Returns:
            Dict with per-sample ``"corr16_peak"`` for ``miss_idxs``, plus
            ``"epe"`` and ``"relative_epe"`` if the batch has ground truth.
        """
        images1 = batch.images1[miss_idxs]
        images2 = batch.images2[miss_idxs]
        gt_flow = (
            None if batch.flow_fields is None else batch.flow_fields[miss_idxs]
        )

        del kwargs
        # Fixed key: cached estimators must be deterministic.
        # TODO(#11): derive per-sample keys from the cache keys.
        key = jax.random.PRNGKey(0)
        if trainable_state is None:
            trainable_state = self.create_trainable_state(images1, key)

        compute = (
            self._compute_cache_payload_jit
            if self.supports_jit()
            else self._compute_cache_payload
        )
        return compute(images1, images2, gt_flow, trainable_state, key)

    def _compute_cache_payload(
        self,
        images1: jnp.ndarray,
        images2: jnp.ndarray,
        gt_flow: jnp.ndarray | None,
        trainable_state: EstimatorTrainableState,
        rng: jax.Array,
    ) -> dict[str, jnp.ndarray]:
        """Per-sample scores of one estimate.

        Args:
            images1: First frame images (B, H, W).
            images2: Second frame images (B, H, W).
            gt_flow: Ground truth flow fields (B, H, W, 2), or None.
            trainable_state: Trainable state used for inference.
            rng: Random key for state creation.

        Returns:
            ``corr16_peak`` and, with ground truth, ``epe`` and
            ``relative_epe``; each of shape (B,).
        """
        B, H, W = images1.shape
        init_flow = jnp.zeros((B, H, W, 2), dtype=jnp.float32)
        state = self.create_state(
            images1, init_flow, image_history_size=2, rng=rng
        )
        state, _ = self(images2, state, trainable_state)
        est_flow = state["estimates"][:, -1]

        payload = {
            "corr16_peak": correlation_peak_quality(images1, images2, est_flow)
        }
        if gt_flow is None:
            return payload

        epe = jnp.linalg.norm(est_flow - gt_flow, axis=-1)
        gt_norm = jnp.linalg.norm(gt_flow, axis=-1)
        rel = (epe**2) / (jnp.maximum(gt_norm, 0.01) ** 2)
        payload["epe"] = jnp.mean(epe, axis=(1, 2))
        payload["relative_epe"] = jnp.mean(rel, axis=(1, 2))
        return payload

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
                trainable_state: Current trainable state of the estimator.
                extras: Additional extras dictionary.

            Returns:
                Tuple of flow field, extras, and metrics.
            """
            if extras is None:
                extras = {}
            flow_field, out_extras, metrics = fn(
                image, state, trainable_state, extras
            )

            flow_field, postprocess_metrics = self.postprocess(
                flow_field, image, state, trainable_state
            )
            return flow_field, out_extras, {**metrics, **postprocess_metrics}

        return call

    def supports_postprocess_jit(self) -> bool:
        """Return whether the postprocessing pipeline can be JIT-compiled.

        Returns:
            True if every configured postprocessing step is jittable.
        """
        return all(
            is_jittable_step(step.keywords["name"])
            for step in self.postprocessing_steps
        )

    def postprocess(
        self,
        flow_field: jnp.ndarray,
        image: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
    ) -> tuple[jnp.ndarray, dict]:
        """Apply configured flow filters and collect their diagnostic metrics.

        Args:
            flow_field: Raw batched flow, before postprocessing.
            image: Current image after preprocessing.
            state: History with preprocessed previous images and estimates.
            trainable_state: Parameters used by learned postprocessing steps.

        Returns:
            Processed flow and the postprocessing metrics.
        """
        metrics = {}
        if len(self.postprocessing_steps) == 0:
            # If no postprocessing steps are defined, return the flow as is.
            return flow_field, metrics
        valid = jnp.ones_like(flow_field[..., 0], dtype=jnp.bool_)
        combined_rejected_mask: jnp.ndarray | None = None
        for idx, step in enumerate(self.postprocessing_steps):
            step_name = step.keywords.get("name", f"step_{idx}")
            flow_field, valid, state = step(
                flow=flow_field,
                valid=valid,
                state=state,
                trainable_state=trainable_state,
                previous_image=state["images"][:, -1, ...],
                current_image=image,
            )
            if valid is not None:
                # NOTE: validation steps return a boolean mask where
                # True marks valid/inlier pixels, False marks
                # rejected/outlier.
                outlier_frac = 1.0 - jnp.mean(
                    valid.astype(jnp.float32), axis=(1, 2)
                )
                metric_prefix = (
                    f"postprocess_{step_name}_{idx}_outlier_percentage"
                )
                metrics[metric_prefix] = outlier_frac * 100.0

                if is_outlier_rejection_step(step_name):
                    metrics[
                        f"postprocess_{step_name}_{idx}_rejected_percentage"
                    ] = outlier_frac * 100.0
                    if combined_rejected_mask is None:
                        combined_rejected_mask = jnp.logical_not(
                            valid.astype(jnp.bool_)
                        )
                    else:
                        combined_rejected_mask = jnp.logical_or(
                            combined_rejected_mask,
                            jnp.logical_not(valid.astype(jnp.bool_)),
                        )

            if DEBUG:
                logger.debug(
                    f"Flow field shape after filtering: {flow_field.shape}"
                )
                if valid is not None:
                    # ``valid`` is True for inliers, so outliers are its
                    # complement.
                    n_outliers = jnp.mean(jnp.sum(~valid, axis=(1, 2)))
                    outlier_pct = jnp.mean(
                        (1.0 - jnp.mean(valid.astype(jnp.float32), axis=(1, 2)))
                        * 100.0
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
        return flow_field, metrics
