"""Comparison script for the different samplers."""

from collections.abc import Callable
import time
import sys
from synthpix import SynthpixBatch


import numpy as np
import jax.numpy as jnp
import jax
import flowgym
from goggles import get_logger
from goggles.types import Metrics
from flowgym.utils import (
    write_dicts_to_csv,
)
from synthpix.sampler import SyntheticImageSampler, RealImageSampler

# Models
from flowgym.common.base import EstimatorTrainableState
from flowgym.common.base import Estimator
from flowgym.types import PRNGKey

# Utils
from flowgym.common.evaluation import compute_stats
from flowgym.utils import (
    GracefulShutdown,
)

sys.modules["estimator"] = flowgym

logger = get_logger(__name__, with_metrics=True)


def compare_performances_on_batches(
    model: Estimator,
    trained_state: EstimatorTrainableState,
    create_state_fn: Callable,
    compute_flow_fn: Callable,
    batch1: SynthpixBatch,
    batch2: SynthpixBatch,
    key: PRNGKey = jax.random.PRNGKey(0),
) -> Metrics:
    """Compare the estimator on two batches using the provided estimator.

    Args:
        model: The model to evaluate.
        trained_state: The trained state of the model.
        create_state_fn: Function to create the state.
        compute_flow_fn: Function to compute the flow.
        batch1: The first batch of images.
        batch2: The second batch of images.
        key: Random key for JAX operations.

    Returns:
        Metrics: The metrics containing the comparison results.
    """
    img1_1 = batch1.images1
    img2_1 = batch1.images2
    flow_field_gt_1 = batch1.flow_fields

    img1_2 = batch2.images1
    img2_2 = batch2.images2
    flow_field_gt_2 = batch2.flow_fields

    # Check if the gt are the same
    if not jnp.isclose(flow_field_gt_1, flow_field_gt_2, atol=1e-4).all():
        raise ValueError(
            "The ground truth flow fields from the two batches are not equal. "
            "This is not supported in this comparison script."
            f"Max difference: {jnp.max(jnp.abs(flow_field_gt_1 - flow_field_gt_2))}"
        )

    # Create the estimation state
    # Note: we use the same key for both images to ensure they are processed
    # in the same way, as they are from the same batch.
    estimation_state1 = create_state_fn(img1_1, key)
    estimation_state2 = create_state_fn(img1_2, key)

    # Compute the flow field estimate
    t1 = time.time()
    estimation_state1, metrics1 = compute_flow_fn(
        img2_1, estimation_state1, trained_state
    )
    t1 = time.time() - t1
    t2 = time.time()
    estimation_state2, metrics2 = compute_flow_fn(
        img2_2, estimation_state2, trained_state
    )
    t2 = time.time() - t2

    # Post process the metrics
    metrics1 = model.process_metrics(metrics1)
    metrics2 = model.process_metrics(metrics2)

    flow_field_1 = estimation_state1["estimates"][:, -1]
    flow_field_2 = estimation_state2["estimates"][:, -1]

    # Compute the average EPE between the two flow fields
    errors = jnp.linalg.norm(flow_field_1 - flow_field_2, axis=-1)
    errors = jnp.mean(errors, axis=(1, 2))
    errors = np.array(errors)

    # Merge metrics
    metrics = {}
    for k in metrics1.keys():
        metrics[k + "_batch1"] = metrics1[k]
        metrics[k + "_batch2"] = metrics2[k]
    metrics["epe_between_batches"] = errors

    # Log evaluation results
    log = "Comparison Metrics - "
    log += f"Time Batch1: {t1:.4f}s, Time Batch2: {t2:.4f}s, "
    log += f"Mean EPE between batches: {np.mean(errors):.5f}, "
    log += f"Max EPE between batches: {np.max(errors):.5f}, "
    log += f"Min EPE between batches: {np.min(errors):.5f}"
    logger.info(log)

    return metrics


def comparison(
    model_config: dict,
    sampler1: SyntheticImageSampler,
    sampler2: RealImageSampler,
    model: Estimator,
    create_state_fn: Callable,
    compute_estimate_fn: Callable,
    trainable_state: EstimatorTrainableState | None,
    key: PRNGKey = jax.random.PRNGKey(0),
) -> None:
    """Compare the flow field between real and synthetic images.

    Args:
        model_config: Configuration of the model.
        sampler1: The image sampler for comparison.
        sampler2: The second image sampler for comparison.
        model: The model to evaluate.
        create_state_fn: Function to create the state.
        compute_estimate_fn: Function to compute the estimate.
        trainable_state: The trained state of the model.
        key: Random key for JAX operations.
    """
    if model_config["estimate_type"] != "flow":
        raise ValueError(
            f"Invalid estimate type: {model_config['estimate_type']}. "
            "Only 'flow' is supported for full comparison."
        )

    with (GracefulShutdown("Stop detected, finishing batch...") as g,):
        epe_list = []
        for i, batch1 in enumerate(sampler1):
            try:
                # Get the corresponding batch from the second sampler
                batch2 = next(sampler2)
            except StopIteration:
                logger.warning(
                    f"Sampler2 has no more batches while sampler1 has {i + 1} batches."
                )
                break

            metrics = compare_performances_on_batches(
                model=model,
                trained_state=trainable_state,
                create_state_fn=create_state_fn,
                compute_flow_fn=compute_estimate_fn,
                batch1=batch1,
                batch2=batch2,
                key=key,
            )

            # If the batch was padded, we need to ignore the padded part
            if isinstance(metrics["epe_between_batches"], np.ndarray):
                errors = metrics["epe_between_batches"][batch1.mask]
            for k, v in metrics.items():
                if isinstance(v, np.ndarray) and v.shape[0] == batch1.images1.shape[0]:
                    v = v[batch1.mask]
                if k != "epe_between_batches":
                    logger.info(f"Batch {i + 1} - {k}: {v}")
            logger.push(metrics=metrics, step=i)

            # Unzip the epe_per_flow from a batch of epe per flow fields
            # to a list of epe per flow field
            if errors.ndim == 0:
                epe_list.append(float(errors))
            else:
                epe_list.extend(errors.tolist())

            if g.stop:
                logger.info("Stop requested, finishing comparison.")
                break

        # Calculate the mean EPE over all batches
        num_flows = len(epe_list)
        logger.info(f"Number of flow fields compared: {num_flows}")
        mean_epe = np.mean(epe_list)
        max_epe = np.max(epe_list)
        min_epe = np.min(epe_list)
        logger.info(f"Max EPE over {i + 1} batches: {max_epe:.5f}")
        logger.info(f"Mean EPE over {i + 1} batches: {mean_epe:.5f}")
        logger.info(f"Min EPE over {i + 1} batches: {min_epe:.5f}")

        # Calculate all the statistics for the EPE and put them in a csv file
        stats_dict = compute_stats(jnp.array(epe_list))
        write_dicts_to_csv(
            "comparison_results.csv",
            [stats_dict],
        )

        logger.info("Comparison completed successfully.")
