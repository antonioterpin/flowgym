"""Evaluation script for the different estimators."""

from collections.abc import Callable
import time
import numpy as np

import sys

from synthpix import SynthpixBatch
import flowgym

import jax.numpy as jnp
import jax
from synthpix.sampler import Sampler
from flowgym.common.evaluation import loss_supervised_density
from flowgym.utils import block_until_ready_dict

# Models
from flowgym.common.base import EstimatorTrainableState, Estimator

# Utils
from goggles import get_logger
from goggles import Metrics
from goggles.shutdown import GracefulShutdown
from flowgym.types import PRNGKey

sys.modules["estimator"] = flowgym

logger = get_logger(__name__, with_metrics=True)


def eval_flow(
    model: Estimator,
    trained_state: EstimatorTrainableState,
    create_state_fn: Callable,
    compute_flow_fn: Callable,
    batch: SynthpixBatch,
    key: PRNGKey = jax.random.PRNGKey(0),
) -> Metrics:
    """Evaluate the model using the provided sampler.

    Args:
        model: The model to evaluate.
        trained_state: The trained state of the model.
        create_state_fn: Function to create the state.
        compute_flow_fn: Function to compute the flow.
        batch: A batch of images and flow fields.
        key: Random key for JAX operations.

    Returns:
        Metrics containing evaluation results.
    """
    # Extract images and ground truth flow fields from the batch
    img1 = batch.images1
    img2 = batch.images2
    flow_field_gt = batch.flow_fields

    # Create the estimation state
    estimation_state = create_state_fn(img1, key)

    # Compute the flow field estimate
    t = time.time()

    if model.is_oracle():
        # If the model is an oracle, provide the ground truth flow field
        estimation_state["estimates"] = (
            estimation_state["estimates"].at[:, -1].set(flow_field_gt)
        )

    estimation_state, metrics = compute_flow_fn(img2, estimation_state, trained_state)
    block_until_ready_dict(estimation_state)
    block_until_ready_dict(metrics)

    t = time.time() - t

    # Post process the metrics
    metrics = model.process_metrics(metrics)

    # Extract the flow field from the estimation state
    flow_field = estimation_state["estimates"][:, -1]

    # Log flow estimate and ground truth
    errors = jnp.linalg.norm(flow_field - flow_field_gt, axis=-1)

    metrics["errors"] = np.array(jnp.mean(errors, axis=(1, 2)))
    metrics["time"] = t
    return metrics


def eval_density(
    model: Estimator,
    trained_state: EstimatorTrainableState,
    create_state_fn: Callable,
    compute_estimate_fn: Callable,
    batch: SynthpixBatch,
    key: PRNGKey = jax.random.PRNGKey(0),
) -> Metrics:
    """Evaluate the model using the provided sampler.

    Args:
        model: The model to evaluate.
        trained_state: The trained state of the model.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        batch: A batch of images and flow fields.
        key: Random key for JAX operations.

    Returns:
        Mean and standard deviation of the loss.
    """
    img1 = batch.images1
    img2 = batch.images2
    if batch.params is None:
        raise ValueError("Batch params cannot be None for density evaluation.")
    density_gt = batch.params.seeding_densities

    # Create the estimation state
    estimation_state = create_state_fn(img1, key)

    # Compute the density estimate
    t = time.time()
    estimation_state, metrics = compute_estimate_fn(
        img2, estimation_state, trained_state
    )
    block_until_ready_dict(estimation_state)
    block_until_ready_dict(metrics)
    t = time.time() - t

    # Post process the metrics
    metrics = model.process_metrics(metrics)

    # Extract the density from the estimation state
    density = estimation_state["estimates"][:, -1]

    # Compute the supervised loss
    e = loss_supervised_density(density, density_gt)

    # Log density estimate and ground truth
    metrics["errors"] = np.array(e)
    metrics["time"] = t

    return metrics


def eval(
    model: Estimator,
    trainable_state: EstimatorTrainableState | None,
    create_state_fn: Callable,
    compute_estimate_fn: Callable,
    batch: SynthpixBatch,
    estimate_type: str = "flow",
    key: PRNGKey = jax.random.PRNGKey(0),
) -> Metrics:
    """Evaluate the model on a SynthpixBatch.

    Args:
        model: The model to evaluate.
        trainable_state: The trained state of the model.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        batch: A batch of images and flow fields.
        estimate_type: Type of the estimator ("flow" or "density").
        key: Random key for JAX operations.

    Returns:
        Metrics containing evaluation results.
    """
    if estimate_type == "flow":
        metrics = eval_flow(
            model=model,
            trained_state=trainable_state,
            create_state_fn=create_state_fn,
            compute_flow_fn=compute_estimate_fn,
            batch=batch,
            key=key,
        )
    else:
        metrics = eval_density(
            model=model,
            trained_state=trainable_state,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            batch=batch,
            key=key,
        )

    log = "Evaluation Results - "
    if estimate_type == "flow":
        log += f"Average EPE: {metrics['errors'].mean():.5f} - "
    else:
        log += f"Average Density Error: {metrics['errors'].mean():.5f} - "
    log += f"Computation Time: {metrics.pop('time'):.5f} seconds"
    logger.info(log)

    return metrics


def eval_full_dataset(
    model: Estimator,
    sampler: Sampler,
    create_state_fn: Callable,
    compute_estimate_fn: Callable,
    trainable_state: EstimatorTrainableState | None,
    estimate_type: str = "flow",
    key: PRNGKey = jax.random.PRNGKey(0),
):
    """Run the full evaluation of the model.

    Args:
        model: The model to evaluate.
        sampler: The image sampler for evaluation.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        trainable_state: The trained state of the model.
        estimate_type: Type of the estimator ("flow" or "density").
        key: Random key for JAX operations.
            Defaults to jax.random.PRNGKey(0).
    """
    with GracefulShutdown("Stop detected, finishing batch...") as g:
        errors_list = []

        logger.info("Starting evaluation...")

        # Reset the sampler before starting the evaluation
        sampler.reset()
        logger.info("Sampler reset.")

        for i, batch in enumerate(sampler):
            logger.info(f"Evaluating batch {i + 1}...")

            metrics = eval(
                model=model,
                trainable_state=trainable_state,
                create_state_fn=create_state_fn,
                compute_estimate_fn=compute_estimate_fn,
                batch=batch,
                estimate_type=estimate_type,
                key=key,
            )

            # If the batch was padded, we need to ignore the padded part
            if isinstance(metrics["errors"], np.ndarray):
                errors = metrics["errors"][batch.mask]
            for k, v in metrics.items():
                if (
                    isinstance(v, np.ndarray)
                    and v.ndim > 0
                    and v.shape[0] == batch.images1.shape[0]
                ):
                    v = v[batch.mask]
                if k != "errors":
                    logger.info(f"Batch {i + 1} - {k}: {v}")

            logger.push(metrics=metrics, step=i)

            # Unzip the epe_per_flow from a batch of epe per flow fields
            # to a list of epe per flow field
            if errors.ndim == 0:
                errors = jnp.expand_dims(errors, axis=0)
            errors_list.extend(list(errors))

            if g.stop:
                logger.info("Stop requested, finishing evaluation.")
                break

        # Calculate the mean EPE over all batches
        num_flows = len(errors_list)
        logger.info(f"Evaluated {num_flows} images in total.")
        errors_array = jnp.asarray(errors_list)
        mean_errors = errors_array.mean()
        max_errors = errors_array.max()
        min_errors = errors_array.min()

        if estimate_type == "flow":
            logger.info(f"Mean EPE over {i + 1} batches: {mean_errors:.5f}")
            logger.info(f"Max EPE over {i + 1} batches: {max_errors:.5f}")
            logger.info(f"Min EPE over {i + 1} batches: {min_errors:.5f}")
        else:
            logger.info(f"Mean Density Error over {i + 1} batches: {mean_errors:.5f}")
            logger.info(f"Max Density Error over {i + 1} batches: {max_errors:.5f}")
            logger.info(f"Min Density Error over {i + 1} batches: {min_errors:.5f}")

        # Push finalized metrics to logger
        logger.push(model.finalize_metrics(), step=i)
        logger.info("Evaluation completed successfully.")
