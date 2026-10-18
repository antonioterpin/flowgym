"""Evaluation script for the different estimators."""

import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# Utils
from goggles import Metrics, get_logger
from goggles.shutdown import GracefulShutdown
from synthpix import SynthpixBatch
from synthpix.sampler import Sampler

# Estimators
from flowgym.common.base import Estimator, EstimatorTrainableState
from flowgym.common.evaluation import loss_supervised_density
from flowgym.training.caching import CacheManager, enrich_batch
from flowgym.types import (
    CachePayload,
    CompiledComputeEstimateFn,
    CompiledCreateStateFn,
    PRNGKey,
)
from flowgym.utils import block_until_ready_dict

logger = get_logger(__name__, with_metrics=True)


def _disable_exploration(
    cache_payload: CachePayload | None,
) -> CachePayload:
    """Return a payload with ``is_training=False`` for evaluation paths.

    Hoisted above the ``has_precomputed_errors`` short-circuit so the invariant
    "any payload leaving an eval entry point has ``is_training=False``" holds
    unconditionally -- if a future change ever makes the short-circuit fall
    through to the estimator, we will not silently regress to training-time
    exploration.

    Args:
        cache_payload: Optional incoming payload from the caller.

    Returns:
        A ``CachePayload`` with ``is_training=False``; either rebuilt from
        ``cache_payload`` (preserving its other fields) or a fresh one when
        the caller did not supply a payload.
    """
    if cache_payload is None:
        return CachePayload(is_training=False)
    return replace(cache_payload, is_training=False)


def eval_flow(
    estimator: Estimator,
    *,
    trained_state: EstimatorTrainableState,
    create_state_fn: CompiledCreateStateFn,
    compute_flow_fn: CompiledComputeEstimateFn,
    batch: SynthpixBatch,
    key: PRNGKey | None = None,
    cache_payload: CachePayload | None = None,
) -> Metrics:
    """Evaluate the estimator using the provided sampler.

    Args:
        estimator: The estimator to evaluate.
        trained_state: The trained state of the estimator.
        create_state_fn: Function to create the state.
        compute_flow_fn: Function to compute the flow.
        batch: A batch of images and flow fields.
        key: Random key for JAX operations.
        cache_payload: Optional cached data for the batch.

    Returns:
        Metrics containing evaluation results.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    # Extract images and ground truth flow fields from the batch
    img1 = batch.images1
    img2 = batch.images2
    flow_field_gt = batch.flow_fields

    # Normalize so any payload leaving eval has is_training=False; estimators
    # (e.g. the ArtOfPIV router) branch on this to disable exploration noise.
    cache_payload = _disable_exploration(cache_payload)

    # Create the estimation state
    estimation_state = create_state_fn(img1, key)

    # Compute the flow field estimate
    t = time.time()

    if estimator.is_oracle():
        # If the estimator is an oracle, provide the ground truth flow field
        estimation_state["estimates"] = (
            estimation_state["estimates"].at[:, -1].set(flow_field_gt)
        )

    estimation_state, metrics = compute_flow_fn(
        img2, estimation_state, trained_state, cache_payload=cache_payload
    )
    block_until_ready_dict(estimation_state)
    block_until_ready_dict(metrics)

    t = time.time() - t

    # Extract the flow field from the estimation state
    flow_field = estimation_state["estimates"][:, -1]
    per_pixel_epe = jnp.linalg.norm(flow_field - flow_field_gt, axis=-1)

    # If metrics already has errors (from cache), use them.
    if "errors" not in metrics:
        metrics["errors"] = np.array(jnp.mean(per_pixel_epe, axis=(1, 2)))
        if "relative_errors" not in metrics:
            relative_errors = (
                per_pixel_epe**2
                / jnp.maximum(jnp.linalg.norm(flow_field_gt, axis=-1), 0.01)
                ** 2
            )
            metrics["relative_errors"] = np.array(
                jnp.mean(relative_errors, axis=(1, 2))
            )

    metrics = dict(
        estimator.process_metrics(
            metrics,
            flow_field=flow_field,
            flow_field_gt=flow_field_gt,
        )
    )

    metrics["time"] = t
    return metrics


def eval_density(
    estimator: Estimator,
    *,
    trained_state: EstimatorTrainableState | None,
    create_state_fn: Callable,
    compute_estimate_fn: Callable,
    batch: SynthpixBatch,
    key: PRNGKey | None = None,
    cache_payload: CachePayload | None = None,
) -> Metrics:
    """Evaluate the estimator using the provided sampler.

    Args:
        estimator: The estimator to evaluate.
        trained_state: The trained state of the estimator.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        batch: A batch of images and flow fields.
        key: Random key for JAX operations.
        cache_payload: Optional cached data for the batch.

    Returns:
        Mean and standard deviation of the loss.

    Raises:
        ValueError: If batch params are None.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    img1 = batch.images1
    img2 = batch.images2
    if batch.params is None:
        raise ValueError("Batch params cannot be None for density evaluation.")
    density_gt = batch.params.seeding_densities

    # Normalize so any payload leaving eval has is_training=False; estimators
    # (e.g. the ArtOfPIV router) branch on this to disable exploration noise.
    cache_payload = _disable_exploration(cache_payload)

    # Create the estimation state
    estimation_state = create_state_fn(img1, key)

    # Compute the density estimate
    t = time.time()
    estimation_state, metrics = compute_estimate_fn(
        img2, estimation_state, trained_state, cache_payload=cache_payload
    )
    block_until_ready_dict(estimation_state)
    block_until_ready_dict(metrics)
    t = time.time() - t

    # Post process the metrics
    metrics = dict(estimator.process_metrics(metrics))

    # Compute the supervised loss if not already provided (e.g., from cache)
    if "errors" not in metrics:
        # Extract the density from the estimation state
        density = estimation_state["estimates"][:, -1]

        # Compute the supervised loss
        e = loss_supervised_density(density, density_gt)
        metrics["errors"] = np.array(e)

    metrics["time"] = t
    return metrics


def eval(
    estimator: Estimator,
    *,
    trainable_state: EstimatorTrainableState,
    create_state_fn: CompiledCreateStateFn,
    compute_estimate_fn: CompiledComputeEstimateFn,
    batch: SynthpixBatch,
    estimate_type: str = "flow",
    key: PRNGKey | None = None,
    cache_payload: CachePayload | None = None,
    time_sample: float | None = None,
    time_enriching: float | None = None,
) -> Metrics:
    """Evaluate the estimator on a SynthpixBatch.

    Args:
        estimator: The estimator to evaluate.
        trainable_state: The trained state of the estimator.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        batch: A batch of images and flow fields.
        estimate_type: Type of the estimator ("flow" or "density").
        key: Random key for JAX operations.
        cache_payload: Optional cached data for the batch.
        time_sample: Optional sampling time for this batch.
        time_enriching: Optional cache-enrichment time for this batch.

    Returns:
        Metrics containing evaluation results.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    if estimate_type == "flow":
        metrics = eval_flow(
            estimator=estimator,
            trained_state=trainable_state,
            create_state_fn=create_state_fn,
            compute_flow_fn=compute_estimate_fn,
            batch=batch,
            key=key,
            cache_payload=cache_payload,
        )
    else:
        metrics = eval_density(
            estimator=estimator,
            trained_state=trainable_state,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            batch=batch,
            key=key,
            cache_payload=cache_payload,
        )

    log = "Evaluation Results - "
    if estimate_type == "flow":
        log += f"Average EPE: {np.asarray(metrics['errors']).mean():.5f} - "
    else:
        avg_error = np.asarray(metrics["errors"]).mean()
        log += f"Average Density Error: {avg_error:.5f} - "
    time_evaluating = float(metrics["time"])
    if time_sample is not None and time_enriching is not None:
        log += (
            f"time_sample: {time_sample:.5f}s - "
            f"time_enriching: {time_enriching:.5f}s - "
            f"time_evaluating: {time_evaluating:.5f}s"
        )
    else:
        log += f"Computation Time: {time_evaluating:.5f} seconds"
    logger.info(log)

    return metrics


def evaluate_batches(
    estimator: Estimator,
    *,
    sampler: Sampler,
    create_state_fn: CompiledCreateStateFn,
    compute_estimate_fn: CompiledComputeEstimateFn,
    trainable_state: EstimatorTrainableState,
    estimate_type: str = "flow",
    key: PRNGKey | None = None,
    num_batches: int = 1,
    reset_sampler: bool = True,
    cache_manager: CacheManager | None = None,
) -> Metrics:
    """Evaluate a subset of batches from a sampler and return aggregated
    metrics.

    Args:
        estimator: Estimator to evaluate.
        sampler: Data sampler producing SynthpixBatch items.
        create_state_fn: Function that initializes estimator state.
        compute_estimate_fn: Function that computes estimates.
        trainable_state: Trainable parameters/state of the estimator.
        estimate_type: Either "flow" or "density".
        key: PRNGKey used for estimator initialization.
        num_batches: Number of batches to evaluate. Must be positive.
        reset_sampler: Whether to reset the sampler before evaluation.
        cache_manager: Optional CacheManager for read-through caching.

    Returns:
        Metrics object containing aggregated validation statistics.

    Raises:
        ValueError: If num_batches is not positive.
        RuntimeError: If sampler yields no batches.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    if num_batches <= 0:
        raise ValueError("num_batches must be a positive integer.")

    if reset_sampler and hasattr(sampler, "reset"):
        sampler.reset()

    errors_list: list[float] = []
    files_list: list[str] = []
    aggregated_metrics: dict[str, list[float]] = {}
    batches_processed = 0
    eval_key = key

    for batch in sampler:
        assert eval_key is not None
        eval_key, batch_key = jax.random.split(eval_key)

        cache_payload = enrich_batch(
            batch,
            estimator,
            cache_manager=cache_manager,
            trainable_state=trainable_state,
        )

        metrics = eval(
            estimator=estimator,
            trainable_state=trainable_state,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            batch=batch,
            estimate_type=estimate_type,
            key=batch_key,
            cache_payload=cache_payload,
        )
        errors = metrics["errors"]
        errors = np.asarray(errors)

        if batch.mask is not None and errors.ndim > 0:
            mask = np.asarray(batch.mask, dtype=bool)
            if errors.shape[0] == mask.shape[0]:
                errors = errors[mask]
        if errors.ndim == 0:
            errors = errors[None]
        errors_list.extend(errors.tolist())
        files_list.extend(_frame_files(batch))

        for metric_name, value in metrics.items():
            if metric_name == "errors":
                continue
            arr = np.asarray(value)
            metric_values: list[float] = []
            if arr.ndim == 0:
                metric_values = [float(arr)]
            elif arr.ndim == 1:
                data = arr
                if batch.mask is not None and data.shape[0] == len(batch.mask):
                    mask = np.asarray(batch.mask, dtype=bool)
                    data = data[mask]
                metric_values = data.astype(float).tolist()
            else:
                continue
            if not metric_values:
                continue
            aggregated_metrics.setdefault(metric_name, []).extend(metric_values)

        batches_processed += 1
        if batches_processed >= num_batches:
            break

    if batches_processed == 0:
        raise RuntimeError("Validation sampler did not yield any batches.")

    errors_array = np.asarray(errors_list, dtype=np.float32)
    summary = {}
    summary["errors"] = errors_array
    summary["mean_error"] = errors_array.mean()
    summary["max_error"] = errors_array.max()
    summary["min_error"] = errors_array.min()
    summary["num_batches"] = batches_processed
    summary["num_samples"] = len(errors_list)
    for metric_name, values in aggregated_metrics.items():
        arr = np.asarray(values, dtype=np.float32)

        # Skip metrics that have no valid data
        if arr.size == 0:
            logger.debug(
                f"Skipping {metric_name}: no data collected across "
                f"{batches_processed} batches"
            )
            continue

        # Check for all-NaN case and skip with informative message
        if np.all(np.isnan(arr)):
            logger.debug(
                f"Skipping {metric_name}: all values are NaN "
                f"(likely no applicable cases across "
                f"{batches_processed} batches)"
            )
            continue

        # Use NaN-aware operations for partial NaN arrays
        mean_value = jnp.nanmean(arr)
        summary[f"{metric_name}"] = mean_value
        summary[f"{metric_name}/mean"] = mean_value
        summary[f"{metric_name}/max"] = jnp.nanmax(arr)
        summary[f"{metric_name}/min"] = jnp.nanmin(arr)

    # Allow estimator to add derived metrics to the summary
    processed = estimator.process_metrics(dict(summary))
    summary.update(processed)
    # After the hook, which may turn every value into an array
    summary["files"] = files_list
    return dict(summary)


def metrics_summary(values: npt.ArrayLike, name: str) -> dict[str, float]:
    """Mean, spread, tail quantiles and extremes of a per-frame metric.

    Args:
        values: The metric of each frame.
        name: Metric name used in the keys, e.g. ``"epe"``.

    Returns:
        ``mean_<name>``, ``std_<name>``, ``median_<name>``, ``p90_<name>``,
        ``p95_<name>``, ``p99_<name>``, ``max_<name>`` and ``min_<name>``.
    """
    values = np.asarray(values, dtype=np.float32)
    return {
        f"mean_{name}": float(values.mean()),
        f"std_{name}": float(values.std()),
        f"median_{name}": float(np.median(values)),
        f"p90_{name}": float(np.percentile(values, 90)),
        f"p95_{name}": float(np.percentile(values, 95)),
        f"p99_{name}": float(np.percentile(values, 99)),
        f"max_{name}": float(values.max()),
        f"min_{name}": float(values.min()),
    }


def write_summary(
    path: str | Path,
    summary: dict[str, float],
    *,
    files: list[str],
    errors: npt.ArrayLike,
) -> None:
    """Write a summary as JSON, with each frame's EPE under ``frames``.

    Args:
        path: Output JSON path.
        summary: Scalar metrics; NaN/inf (not valid JSON) are written as null.
        files: File name of each frame; ``frames`` is omitted when these do
            not match ``errors`` (e.g. synthetic data has no files).
        errors: EPE of each frame.
    """
    out: dict = {k: v if math.isfinite(v) else None for k, v in summary.items()}
    errors = np.asarray(errors, dtype=np.float32).reshape(-1)
    if files and len(files) == errors.size:
        out["frames"] = {
            "file": files,
            "epe": [e if math.isfinite(e) else None for e in errors.tolist()],
        }
    Path(path).write_text(json.dumps(out, indent=2))


def _frame_files(batch: SynthpixBatch) -> list[str]:
    """File names of a batch's real (unpadded) frames.

    Args:
        batch: The evaluated batch.

    Returns:
        The base names, or ``[]`` if the batch has no files.
    """
    files = getattr(batch, "files", None)
    if files is None:
        return []
    if batch.mask is not None:
        files = [
            f for f, m in zip(files, np.asarray(batch.mask), strict=True) if m
        ]
    return [Path(f).name for f in files]


def scalar_metrics(metrics: Metrics) -> dict[str, float]:
    """Scalar entries of evaluation metrics (per-sample ``errors`` dropped).

    Args:
        metrics: Metrics returned by ``evaluate_batches``.

    Returns:
        Metric name to float.
    """
    return {
        name: float(value)
        for name, value in metrics.items()
        if name != "errors"
        and isinstance(value, (int, float, np.generic, np.ndarray, jax.Array))
        and np.ndim(value) == 0
    }


def log_eval_metrics(
    metrics: Metrics, *, prefix: str, step: int, title: str
) -> None:
    """Log scalar metrics under ``<prefix>/`` and as one info line.

    Args:
        metrics: Metrics returned by ``evaluate_batches``.
        prefix: Metric namespace (e.g. ``val`` or ``test``).
        step: Step to log the metrics at.
        title: Start of the info line (e.g. ``Validation after batch 10``).
    """
    scalars = scalar_metrics(metrics)
    for name, value in scalars.items():
        logger.scalar(f"{prefix}/{name}", value, step=step)
    shown = ", ".join(
        f"{name}={value:.5f}"
        for name, value in scalars.items()
        if name not in ("num_batches", "num_samples")
    )
    logger.info(f"{title}: {shown}")


def eval_full_dataset(
    estimator: Estimator,
    *,
    sampler: Sampler,
    create_state_fn: CompiledCreateStateFn,
    compute_estimate_fn: CompiledComputeEstimateFn,
    trainable_state: EstimatorTrainableState,
    estimate_type: str = "flow",
    key: PRNGKey | None = None,
    print_files: bool = False,
    num_batches: int | None = None,
    cache_manager: CacheManager | None = None,
    summary_out: str | Path | None = None,
) -> None:
    """Run the full evaluation of the estimator.

    Args:
        estimator: The estimator to evaluate.
        sampler: The image sampler for evaluation.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        trainable_state: The trained state of the estimator.
        estimate_type: Type of the estimator ("flow" or "density").
        key: Random key for JAX operations.
            Defaults to jax.random.PRNGKey(0).
        print_files: Whether to print the files being evaluated.
        num_batches: Number of batches to evaluate. Must be positive.
        cache_manager: Optional CacheManager for read-through caching.
        summary_out: Optional JSON path for the final flow summary
            (mean/max/min EPE and relative error).

    Raises:
        RuntimeError: If evaluation sampler yields no batches.
    """
    with contextlib.ExitStack() as stack:
        g = stack.enter_context(
            GracefulShutdown("Stop detected, finishing batch...")
        )
        if cache_manager is not None:
            # We use a contextlib.ExitStack instead of a simple with-statement
            # to ensure the cache manager gets added as a context manager only
            # if it's not None, since it may be optional.
            stack.enter_context(cache_manager)
        errors_list = []
        files_list: list[str] = []
        relative_errors_list = []
        if key is None:
            key = jax.random.PRNGKey(0)

        logger.info("Starting evaluation...")

        # Reset the sampler before starting the evaluation
        sampler.reset()
        logger.info("Sampler reset.")
        t_start = time.time()

        total_time_sample = 0.0
        total_time_enriching = 0.0
        total_time_evaluating = 0.0
        batches_processed = 0
        sampler_iter = iter(sampler)

        while True:
            t_sample_start = time.time()
            try:
                batch = next(sampler_iter)
            except StopIteration:
                break
            time_sample = time.time() - t_sample_start
            total_time_sample += time_sample

            batch_idx = batches_processed + 1
            logger.info(f"Evaluating batch {batch_idx}...")
            assert key is not None
            key, batch_key = jax.random.split(key)

            t_enrich_start = time.time()
            cache_payload = enrich_batch(
                batch,
                estimator,
                cache_manager=cache_manager,
                trainable_state=trainable_state,
            )
            time_enriching = time.time() - t_enrich_start
            total_time_enriching += time_enriching

            t_evaluate_start = time.time()
            metrics = dict(
                eval(
                    estimator=estimator,
                    trainable_state=trainable_state,
                    create_state_fn=create_state_fn,
                    compute_estimate_fn=compute_estimate_fn,
                    batch=batch,
                    estimate_type=estimate_type,
                    key=batch_key,
                    cache_payload=cache_payload,
                    time_sample=time_sample,
                    time_enriching=time_enriching,
                )
            )
            time_evaluating = time.time() - t_evaluate_start
            total_time_evaluating += time_evaluating
            metrics["time_sample"] = time_sample
            metrics["time_enriching"] = time_enriching
            metrics["time_evaluating"] = time_evaluating

            # If the batch was padded, we need to ignore the padded part
            for k, v in metrics.items():
                if (
                    isinstance(v, np.ndarray)
                    and v.ndim > 0
                    and v.shape[0] == batch.images1.shape[0]
                ):
                    masked_v = v[batch.mask]
                    metrics[k] = masked_v
                if (
                    k not in {"errors", "relative_errors"}
                    and np.isscalar(v)
                    and k
                    not in {"time_sample", "time_enriching", "time_evaluating"}
                ):
                    logger.info(f"Batch {batch_idx} - {k}: {v}")
                elif isinstance(v, np.ndarray) and k not in {
                    "errors",
                    "relative_errors",
                }:
                    metrics[k] = v.mean()

            logger.push(metrics=metrics, step=batches_processed)

            # Unzip the errors from a batch of errors per flow fields
            # to a list of errors per flow field
            if "errors" in metrics:
                errors = jnp.asarray(metrics["errors"])
                logger.info(f"errors: {errors}")
                if errors.ndim == 0:
                    errors = jnp.expand_dims(errors, axis=0)
                errors_list.extend(list(errors))
                files_list.extend(_frame_files(batch))

            if "relative_errors" in metrics:
                relative_errors = jnp.asarray(metrics["relative_errors"])
                if relative_errors.ndim == 0:
                    relative_errors = jnp.expand_dims(relative_errors, axis=0)
                relative_errors_list.extend(list(relative_errors))

            batches_processed += 1

            if g.stop:
                logger.info("Stop requested, finishing evaluation.")
                break

            if print_files and batch.files is not None:
                files = batch.files
                if batch.mask is not None:
                    files = tuple(
                        f for j, f in enumerate(files) if batch.mask[j].item()
                    )

                errors = np.asarray(metrics["errors"])
                if errors.ndim == 0:
                    errors = np.expand_dims(errors, axis=0)

                for f, e in zip(files, errors, strict=True):
                    logger.info(f"File: {f}, EPE: {e:.5f}")

            if num_batches is not None and batches_processed >= num_batches:
                logger.info(f"Reached limit of {num_batches} batches.")
                break

        if batches_processed == 0:
            raise RuntimeError("Evaluation sampler did not yield any batches.")

        # Calculate the mean error over all batches
        errors_array = jnp.asarray(errors_list)
        num_flows = errors_array.shape[0]
        logger.info(f"Evaluated {num_flows} image pairs in total.")
        mean_errors = errors_array.mean()
        max_errors = errors_array.max()
        min_errors = errors_array.min()

        relative_errors_array = jnp.asarray(relative_errors_list)
        if relative_errors_array.size > 0:
            mean_relative_errors = relative_errors_array.mean()
            max_relative_errors = relative_errors_array.max()
            min_relative_errors = relative_errors_array.min()
        else:
            mean_relative_errors = 0.0
            max_relative_errors = 0.0
            min_relative_errors = 0.0

        if estimate_type == "flow":
            logger.info(
                f"Mean EPE over {batches_processed} batches: {mean_errors:.5f}"
            )
            logger.info(
                f"Max EPE over {batches_processed} batches: {max_errors:.5f}"
            )
            logger.info(
                f"Min EPE over {batches_processed} batches: {min_errors:.5f}"
            )
            logger.info(
                f"Mean Relative EPE over {batches_processed} batches: "
                f"{mean_relative_errors:.5f}"
            )
            logger.info(
                f"Max Relative EPE over {batches_processed} batches: "
                f"{max_relative_errors:.5f}"
            )
            logger.info(
                f"Min Relative EPE over {batches_processed} batches: "
                f"{min_relative_errors:.5f}"
            )
            summary = metrics_summary(errors_array, "epe")
            # Cached evals may report errors without relative errors
            if relative_errors_array.size > 0:
                summary["mean_relative_error"] = float(mean_relative_errors)
                summary["max_relative_error"] = float(max_relative_errors)
                summary["min_relative_error"] = float(min_relative_errors)
            estimator.record_eval_summary(summary)
            if summary_out is not None:
                write_summary(
                    summary_out, summary, files=files_list, errors=errors_array
                )
        else:
            logger.info(
                f"Mean Density Error over {batches_processed} batches: "
                f"{mean_errors:.5f}"
            )
            logger.info(
                f"Max Density Error over {batches_processed} batches: "
                f"{max_errors:.5f}"
            )
            logger.info(
                f"Min Density Error over {batches_processed} batches: "
                f"{min_errors:.5f}"
            )

        # Push finalized metrics to logger
        logger.push(estimator.finalize_metrics(), step=batches_processed)

        wall_time = time.time() - t_start
        logger.info(
            f"Final timing - time_sample: {total_time_sample:.2f}s - "
            f"time_enriching: {total_time_enriching:.2f}s - "
            f"time_evaluating: {total_time_evaluating:.2f}s"
        )
        time_per_sample = total_time_sample / batches_processed
        time_per_enrich = total_time_enriching / batches_processed
        time_per_eval = total_time_evaluating / batches_processed
        logger.info(
            f"Timing per batch - sample: {time_per_sample:.5f}s, "
            f"enrich: {time_per_enrich:.5f}s, "
            f"evaluate: {time_per_eval:.5f}s"
        )

        logger.info("Evaluation completed successfully.")
        logger.info(f"Total evaluation time: {wall_time:.2f} seconds.")
