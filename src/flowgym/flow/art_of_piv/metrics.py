"""Metrics processing for the Art of PIV estimator.

Handles conversion of raw JAX metric dictionaries produced during inference
and training into the formatted ``Metrics`` objects consumed by the logging
/ visualization pipeline.  Stateful components (running-mean and sliding-window
choice-rate trackers) are encapsulated in :class:`ChoiceTracker`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from goggles import Metrics
from goggles.filters import AverageFilter

from flowgym.types import CachePayload


class ChoiceTracker:
    """Track estimator choice statistics with running and windowed means.

    Maintains two views of per-estimator selection rates:

    - Running mean: proportion of each estimator selected across all
      samples seen since creation.
    - Sliding-window mean: proportion over the last `window_size` batches.
    """

    def __init__(self, num_estimators: int, window_size: int = 100) -> None:
        """Initialize the tracker.

        Args:
            num_estimators: Number of sub-estimators.
            window_size: Number of recent batches to keep in the sliding window.
        """
        self.num_estimators = num_estimators
        self.window_size = window_size

        # Running count of chosen estimator IDs across all seen samples.
        # Array of shape (num_estimators,) where each element is the count of
        # how many times that estimator was selected.
        self._running_counts = np.zeros(num_estimators, dtype=np.int64)

        # Sliding-window average over batch counts and batch sizes.
        # Keeping two averages preserves sample-weighted rates:
        #   mean(counts) / mean(batch_size) == sum(counts) / sum(batch_size)
        self._window_counts_filter = AverageFilter(window_size=window_size)
        self._window_batch_size_filter = AverageFilter(window_size=window_size)

    def update(
        self,
        choices: np.ndarray | jnp.ndarray,
    ) -> dict[str, np.ndarray]:
        """Record a batch of choices and return per-estimator rate metrics.

        Args:
            choices: 1-D integer array of shape ``(batch_size,)`` where each
                element is the index of the chosen estimator.

        Returns:
            Dictionary with keys of the form
            ``choices/{batch,running,window}_rate_estimator_{i}``.
        """
        choices = np.asarray(choices)
        batch_size = int(choices.shape[0])
        metrics: dict[str, np.ndarray] = {}

        # Per-batch counts and rates.
        batch_counts = np.bincount(
            choices.astype(np.int64), minlength=self.num_estimators
        ).astype(np.int64)
        batch_rates = batch_counts / float(batch_size)
        for i in range(self.num_estimators):
            metrics[f"choices/rate_estimator_{i}"] = np.asarray(batch_rates[i])

        # Running mean over all previously seen choices.
        self._running_counts += batch_counts
        running_total = int(self._running_counts.sum())
        running_rates = self._running_counts / float(running_total)
        for i in range(self.num_estimators):
            metrics[f"choices/running_rate_estimator_{i}"] = np.asarray(
                running_rates[i]
            )

        # Sliding-window sample-weighted mean over recent batches.
        window_counts_avg = np.asarray(self._window_counts_filter(batch_counts))
        window_batch_size_avg = float(
            np.asarray(
                self._window_batch_size_filter(
                    np.asarray(batch_size, dtype=np.float64)
                )
            )
        )
        window_rates = window_counts_avg / window_batch_size_avg
        for i in range(self.num_estimators):
            metrics[f"choices/window_rate_estimator_{i}"] = np.asarray(
                window_rates[i]
            )

        return metrics


def build_inference_metrics(
    *,
    qs: jnp.ndarray,
    choices: jnp.ndarray,
    exploration_rate: jnp.ndarray,
    steps: jnp.ndarray,
    estimated_epe: jnp.ndarray | float,
    cache_payload: CachePayload | None = None,
    errors: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Build inference metrics from router outputs and optional cache payload.

    This helper centralizes the metric assembly used by:

    - cache short-circuit path in ``inference.estimate``,
    - sub-estimator dispatch path in ``inference.run_subestimators``.

    Args:
        qs: Router Q-values, shape ``(B, num_estimators)``.
        choices: Selected estimator indices, shape ``(B,)``.
        exploration_rate: Per-sample exploration rates, shape ``(B,)``.
            Kept per-sample (instead of scalar) to support heterogeneous
            batches where schedule values may differ across samples.
        steps: Per-sample training-step values, shape ``(B,)``.
            Kept per-sample for the same reason as ``exploration_rate``.
        estimated_epe: Per-sample estimate for logging.
        cache_payload: Optional cache payload with ``epe_all`` and ``epe_mask``.
        errors: Optional per-sample error values to emit as ``errors``.

    Returns:
        Dictionary of inference metrics.
    """
    batch_size = qs.shape[0]
    choices_idx = jnp.reshape(choices, (batch_size, 1))
    choices_flat = choices_idx.reshape(batch_size)

    metrics: dict[str, Any] = {
        "q_values": qs,
        "choices": choices_flat,
        "exploration_rate": exploration_rate,
        "steps": steps,
        "estimated_epe": estimated_epe,
    }
    if errors is not None:
        metrics["errors"] = errors

    probs = jax.nn.softmax(qs, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
    metrics["choices/entropy"] = entropy

    if cache_payload is None or cache_payload.epe_all is None:
        return metrics

    epe_all = jnp.asarray(cache_payload.epe_all)
    epe_mask_raw = cache_payload.epe_mask
    if epe_mask_raw is not None:
        epe_mask = jnp.asarray(epe_mask_raw, dtype=bool)
    else:
        epe_mask = jnp.ones_like(epe_all, dtype=bool)

    epe_all_masked = jnp.asarray(jnp.where(epe_mask, epe_all, jnp.inf))
    row_has_any = jnp.any(epe_mask, axis=-1)
    best_model_idx = jnp.argmin(epe_all_masked, axis=-1)
    accuracy = jnp.where(
        row_has_any,
        (choices_flat == best_model_idx).astype(jnp.float32),
        0.0,
    )
    metrics["choices/accuracy"] = accuracy

    min_epe = jnp.min(epe_all_masked, axis=-1, keepdims=True)
    oracle_epe = jnp.where(
        row_has_any, jnp.asarray(min_epe).squeeze(-1), jnp.nan
    )
    metrics["oracle_epe"] = oracle_epe

    chosen_epe = jnp.take_along_axis(epe_all, choices_idx, axis=-1).reshape(
        batch_size
    )
    chosen_valid = jnp.take_along_axis(epe_mask, choices_idx, axis=-1).reshape(
        batch_size
    )
    regret = jnp.asarray(chosen_epe) - jnp.asarray(oracle_epe)
    metrics["regret"] = jnp.where(chosen_valid & row_has_any, regret, jnp.nan)

    for eps in [0.005, 0.001]:
        is_within_eps = (epe_all_masked <= (min_epe + eps)).astype(jnp.float32)
        is_within_eps = jnp.where(row_has_any[:, None], is_within_eps, 0.0)
        chosen_within_eps = jnp.take_along_axis(
            is_within_eps, choices_idx, axis=-1
        ).reshape(batch_size)
        metrics[f"choices/accuracy_eps_{eps}"] = chosen_within_eps

    return metrics


def process_art_of_piv_metrics(
    metrics: dict,
    choice_tracker: ChoiceTracker,
) -> Metrics:
    """Process raw Art of PIV metrics into formatted ``Metrics`` for logging.

    This is a stateless transformation except for the *choice_tracker* which
    accumulates running and windowed selection rates.

    Args:
        metrics: Raw metrics dictionary produced by ``_estimate`` or the
            training step.
        choice_tracker: Mutable tracker for per-estimator selection statistics.

    Returns:
        Formatted Metrics object.
    """
    processed: dict = {}

    # Q-value summary stats
    if "q_values" in metrics:
        q_values = metrics["q_values"]
        processed["q_values/mean"] = np.asarray(jnp.mean(q_values))
        processed["q_values/max"] = np.asarray(jnp.max(q_values))
        processed["q_values/min"] = np.asarray(jnp.min(q_values))

    # Choice statistics via tracker
    if "choices" in metrics:
        choice_metrics = choice_tracker.update(metrics["choices"])
        processed.update(choice_metrics)

    # Scalar passthrough keys
    if "exploration_rate" in metrics:
        # `exploration_rate` is tracked as a per-sample vector in raw metrics;
        # log the batch-mean scalar for dashboards.
        processed["exploration/rate"] = np.asarray(
            jnp.mean(jnp.asarray(metrics["exploration_rate"]))
        )

    if "estimated_epe" in metrics:
        processed["estimated_epe"] = np.asarray(metrics["estimated_epe"])

    if "choices/entropy" in metrics:
        processed["choices/entropy"] = np.asarray(metrics["choices/entropy"])

    if "choices/accuracy" in metrics:
        processed["choices/accuracy"] = np.asarray(metrics["choices/accuracy"])

    for eps in [0.005, 0.001]:
        key = f"choices/accuracy_eps_{eps}"
        if key in metrics:
            processed[key] = np.asarray(metrics[key])

    if "oracle_epe" in metrics:
        processed["oracle_epe"] = np.asarray(metrics["oracle_epe"])

    if "regret" in metrics:
        processed["regret"] = np.asarray(metrics["regret"])

    # Target-transform and training-step metrics
    for key in [
        "epe/raw_mean",
        "target/mean",
        "regret/raw_mean",
        "regret/raw_p90",
        "epe/raw_best_mean",
    ]:
        if key in metrics:
            processed[key] = np.asarray(metrics[key])

    for key in ["errors", "relative_errors"]:
        if key in metrics:
            processed[key] = np.asarray(metrics[key])

    # Derived improvement metrics (populated after multi-batch aggregation)
    if "mean_error" in metrics and "oracle_epe/mean" in metrics:
        mean_error = float(metrics["mean_error"])
        oracle_mean = float(metrics["oracle_epe/mean"])
        improvement_margin = mean_error - oracle_mean
        processed["improvement_margin"] = improvement_margin
        if mean_error > 0:
            processed["improvement_pct"] = improvement_margin / mean_error * 100

    return processed
