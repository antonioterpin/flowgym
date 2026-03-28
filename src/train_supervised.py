"""Train flow estimators with supervised learning."""

import contextlib
import itertools
import time
from typing import Literal

import goggles as gg
import jax
import numpy as np
from synthpix.sampler import Sampler

from eval import evaluate_batches
from flowgym.common.base import Estimator, NNEstimatorTrainableState
from flowgym.make import save_model
from flowgym.training.caching import CacheManager, enrich_batch
from flowgym.training.replay import ReplayBuffer
from flowgym.types import (
    CompiledComputeEstimateFn,
    CompiledCreateStateFn,
    PRNGKey,
    SupervisedExperience,
    SupervisedTrainStep,
)
from flowgym.utils import DEBUG, GracefulShutdown

logger = gg.get_logger(__name__, with_metrics=True)


def train_supervised(
    model: Estimator,
    model_config: dict,
    trainable_state: NNEstimatorTrainableState,
    out_dir: str,
    create_state_fn: CompiledCreateStateFn,
    compute_estimate_fn: CompiledComputeEstimateFn,
    sampler: Sampler,
    val_sampler: Sampler | None = None,
    val_interval: int | None = None,
    val_num_batches: int = 1,
    num_batches: int = 1000,
    estimate_type: Literal["flow", "density"] = "flow",
    save_every: int = 10000,
    log_every: int = 1,
    save_only_best: bool = False,
    key: PRNGKey | None = None,
    replay_buffer_capacity: int = 0,
    replay_ratio: float = 0.0,
    prefetch_replay_size: int = 0,
    cache_manager: CacheManager | None = None,
) -> None:
    """Train the flow estimator.

    Args:
        model: The flow estimator model.
        model_config: Configuration for the model.
        trainable_state: The initial state of the model.
        out_dir: Directory to save the model.
        create_state_fn: Function to create the initial state of the estimator.
        compute_estimate_fn: Function to compute the estimate.
        sampler: The sampler to use to generate images.
        val_sampler: The sampler to use to generate images for validation.
        val_interval: Frequency to validate the model.
        val_num_batches: Number of batches to validate for.
        num_batches: Number of batches to train for.
        estimate_type: Type of estimate to compute.
        save_every: Frequency to save the model.
        log_every: Frequency to log the training progress.
        save_only_best: Whether to save only the best model
                    by evaluating it on the validation set.
        key: Random key for JAX operations.
        replay_buffer_capacity: Capacity of the replay buffer.
        replay_ratio: Ratio of replay samples to collected samples.
        prefetch_replay_size: Prefetch buffer size for replay. If > 0,
            enables GPU prefetch.
        cache_manager: Optional CacheManager for read-through caching.

    Raises:
        ValueError: If estimate_type is not "flow" or "density", or if
            required ground truth data is missing from batch.
        Exception: Any exception raised during training is logged and re-raised.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Create the training step function
    train_step_fn = model.create_train_step()
    assert isinstance(train_step_fn, SupervisedTrainStep), (
        "Expected SupervisedTrainStep callable"
    )

    if model_config["config"].get("jit", False) and not DEBUG:
        if model.supports_train_step_jit():
            train_step_fn = jax.jit(train_step_fn)
        else:
            logger.warning(
                "Disabling JIT for supervised train step: "
                "model reports non-jittable train-step execution path."
            )

    logger.info("Training step function compiled successfully.")

    best_validation_score = float("-inf")
    has_saved_best_checkpoint = False
    last_val_metrics: dict | None = None

    # Initialize the replay buffer
    replay_buffer = None
    if replay_buffer_capacity > 0:
        assert key is not None
        key, subkey = jax.random.split(key)
        replay_buffer = ReplayBuffer(
            capacity=replay_buffer_capacity, key=subkey
        )
        logger.info(
            f"Replay buffer initialized with capacity {replay_buffer_capacity}."
        )

    with contextlib.ExitStack() as stack:
        g = stack.enter_context(
            GracefulShutdown("Stop detected, finishing epoch...")
        )
        if cache_manager is not None:
            # We use a contextlib.ExitStack instead of a simple with-statement
            # to ensure the cache manager gets added as a context manager only
            # if it's not None, since it may be optional.
            stack.enter_context(cache_manager)

        # Run initial validation before any training
        if val_sampler is not None and val_interval and val_interval > 0:
            logger.info(
                "Running initial validation (batch 0, before training)..."
            )
            assert key is not None
            key, val_key = jax.random.split(key)
            try:
                val_metrics = evaluate_batches(
                    model=model,
                    sampler=val_sampler,
                    create_state_fn=create_state_fn,
                    compute_estimate_fn=compute_estimate_fn,
                    trainable_state=trainable_state,
                    estimate_type=estimate_type,
                    key=val_key,
                    num_batches=val_num_batches,
                    cache_manager=cache_manager,
                )
                # Log all validation metrics generically
                for metric_name, value in val_metrics.items():
                    if metric_name == "errors":
                        continue
                    scalar = None
                    if isinstance(value, (int, float)):
                        scalar = float(value)
                    elif isinstance(value, np.ndarray) and value.ndim == 0:
                        scalar = float(value)
                    if scalar is not None:
                        logger.scalar(f"val/{metric_name}", scalar, step=0)

                # Build validation message with all scalar metrics
                mean_error = float(val_metrics.get("mean_error", float("nan")))
                current_score = float(
                    val_metrics.get("mask_f1", -mean_error)
                )
                init_val_msg = (
                    f"Initial validation (before training): "
                    f"mean_error={mean_error:.5f}"
                )
                # Append all other scalar metrics
                for k, v in val_metrics.items():
                    if k in (
                        "errors",
                        "mean_error",
                        "num_batches",
                        "num_samples",
                    ):
                        continue
                    if isinstance(v, (int, float)):
                        init_val_msg += f", {k}={float(v):.5f}"
                    elif isinstance(v, np.ndarray) and v.ndim == 0:
                        init_val_msg += f", {k}={float(v):.5f}"
                logger.info(init_val_msg)
                last_val_metrics = val_metrics
                best_validation_score = current_score
            except Exception as e:
                logger.error(f"Initial validation failed: {e}")

        t_sample_start = time.time()
        last_batch_idx = -1
        try:
            for batch_idx, batch in enumerate(
                itertools.islice(sampler, num_batches)
            ):
                last_batch_idx = batch_idx
                t_sample = time.time() - t_sample_start
                if g.stop:
                    break

                # Unpack the batch
                images1 = batch.images1
                images2 = batch.images2
                if estimate_type == "flow":
                    ground_truth = batch.flow_fields
                elif estimate_type == "density":
                    if (
                        batch.params is None
                        or batch.params.seeding_densities is None
                    ):
                        raise ValueError(
                            "Seeding densities not found in batch."
                        )
                    ground_truth = batch.params.seeding_densities
                else:
                    raise ValueError(f"Unknown estimate type: {estimate_type}")

                # Caching Logic - unified interface
                t_enrich = time.time()
                cache_payload = enrich_batch(
                    batch,
                    model,
                    cache_manager=cache_manager,
                    trainable_state=trainable_state,
                )
                t_enrich = time.time() - t_enrich

                # Create the estimator state
                assert key is not None
                key, subkey = jax.random.split(key)
                estimator_state = create_state_fn(images1, subkey)

                # Create the experience object
                experience = SupervisedExperience(
                    state=estimator_state,
                    obs=(images1, images2),
                    ground_truth=ground_truth,
                    cache_payload=cache_payload,
                )
                experience = model.prepare_experience_for_training(
                    experience, trainable_state
                )
                # Do one training step
                t = time.time()
                loss, trainable_state, metrics = train_step_fn(
                    trainable_state=trainable_state,
                    experience=experience,
                )
                t_train = time.time() - t

                # Log filenames for debugging coverage/randomization
                batch_files = getattr(batch, "files", [])
                if batch_files:
                    # Log up to 4 files from the batch
                    files_str = ", ".join(
                        [str(f).split("/")[-1] for f in batch_files[:4]]
                    )
                    logger.info(f"Batch {batch_idx} files: {files_str}")

                logger.info(
                    f"Batch {batch_idx} timing: "
                    f"Sample={t_sample:.4f}s, "
                    f"Enrich={t_enrich:.4f}s, "
                    f"Train={t_train:.4f}s. "
                    f"Total={t_sample + t_enrich + t_train:.4f}s"
                )

                # Store experience in the replay buffer
                if replay_buffer is not None:
                    # Allow model to enrich experience before storing
                    enriched_experience = model.prepare_experience_for_replay(
                        experience, trainable_state
                    )
                    # We store unbatched experiences
                    B = images1.shape[0]
                    for i in range(B):
                        exp = SupervisedExperience(
                            state=jax.tree_util.tree_map(
                                lambda x, i=i: x[i], enriched_experience.state
                            ),
                            obs=(
                                enriched_experience.obs[0][i],
                                enriched_experience.obs[1][i],
                            ),
                            ground_truth=enriched_experience.ground_truth[i],
                        )
                        replay_buffer.push(exp)

                    # Replay training steps
                    # We perform replay steps with probability replay_ratio % 1
                    # and int(replay_ratio) steps
                    num_replay_steps = int(replay_ratio)
                    assert key is not None
                    key, subkey = jax.random.split(key)
                    if jax.random.uniform(subkey) < (replay_ratio % 1):
                        num_replay_steps += 1

                    for _ in range(num_replay_steps):
                        if len(replay_buffer) >= B:
                            # Use sample_iter with prefetch for GPU prefetch
                            replay_iter = replay_buffer.sample_iter(
                                batch_size=B,
                                num_batches=1,
                                prefetch=prefetch_replay_size,
                            )
                            for replay_exp in replay_iter:
                                replay_exp_prepared = (
                                    model.prepare_experience_for_training(
                                        replay_exp, trainable_state
                                    )
                                )
                                t_replay = time.time()
                                (
                                    replay_loss,
                                    trainable_state,
                                    _,
                                ) = train_step_fn(
                                    trainable_state=trainable_state,
                                    experience=replay_exp_prepared,
                                )
                                t_replay = time.time() - t_replay
                                logger.info(
                                    f"Replay step took {t_replay:.4f} seconds."
                                )
                                logger.scalar(
                                    "loss/replay",
                                    float(replay_loss),
                                    step=batch_idx,
                                )

                if (
                    val_sampler is not None
                    and val_interval
                    and val_interval > 0
                    and batch_idx > 0
                    and batch_idx % val_interval == 0
                ):
                    assert key is not None
                    key, val_key = jax.random.split(key)

                    try:
                        val_metrics = evaluate_batches(
                            model=model,
                            sampler=val_sampler,
                            create_state_fn=create_state_fn,
                            compute_estimate_fn=compute_estimate_fn,
                            trainable_state=trainable_state,
                            estimate_type=estimate_type,
                            key=val_key,
                            num_batches=val_num_batches,
                            cache_manager=cache_manager,
                        )
                    except Exception as e:
                        logger.error(f"Validation failed: {e}")
                        continue

                    # Log all validation metrics generically
                    for metric_name, value in val_metrics.items():
                        if metric_name == "errors":
                            continue
                        scalar = None
                        if isinstance(value, (int, float)):
                            scalar = float(value)
                        elif isinstance(value, np.ndarray) and value.ndim == 0:
                            scalar = float(value)
                        if scalar is not None:
                            logger.scalar(
                                f"val/{metric_name}", scalar, step=batch_idx
                            )

                    # Build validation message with all scalar metrics
                    mean_error = float(
                        val_metrics.get("mean_error", float("nan"))
                    )
                    val_msg = (
                        f"Validation after batch {batch_idx}: "
                        f"mean_error={mean_error:.5f}"
                    )
                    # Append all other scalar metrics
                    for k, v in val_metrics.items():
                        if k in (
                            "errors",
                            "mean_error",
                            "num_batches",
                            "num_samples",
                        ):
                            continue
                        if isinstance(v, (int, float)):
                            val_msg += f", {k}={float(v):.5f}"
                        elif isinstance(v, np.ndarray) and v.ndim == 0:
                            val_msg += f", {k}={float(v):.5f}"
                    logger.info(val_msg)
                    last_val_metrics = val_metrics

                if batch_idx % log_every == 0:
                    logger.scalar("loss", float(loss), step=batch_idx)
                    for metric_name, metric_value in metrics.items():
                        logger.scalar(
                            metric_name, float(metric_value), step=batch_idx
                        )
                    msg = f"Batch {batch_idx} finished.\nloss: {loss:.6f}."
                    for metric_name, metric_value in metrics.items():
                        if metric_name != "loss":
                            msg += f"\n{metric_name}: {float(metric_value):.6f}"
                    logger.info(msg)

                if batch_idx > 0 and (
                    batch_idx % save_every == 0 or batch_idx == num_batches - 1
                ):
                    if not save_only_best:
                        save_model(
                            state=trainable_state,
                            out_dir=out_dir,
                            step=batch_idx,
                            model=model,
                            model_name=model.__class__.__name__,
                            sampler=sampler,
                        )

                    elif last_val_metrics is None:
                        logger.info(
                            f"Skipping best-model save at batch {batch_idx}: "
                            f"no validation computed yet."
                        )
                    else:
                        # Select best checkpoint by validation mask_f1
                        # (higher is better). Fallback to -mean_error
                        # if mask_f1 is unavailable.
                        mean_error = float(
                            last_val_metrics.get("mean_error", float("nan"))
                        )
                        current_score = float(
                            last_val_metrics.get("mask_f1", -mean_error)
                        )

                        should_save = (not has_saved_best_checkpoint) or (
                            current_score > best_validation_score
                        )
                        if should_save:
                            best_validation_score = current_score
                            has_saved_best_checkpoint = True

                            save_model(
                                state=trainable_state,
                                out_dir=out_dir,
                                step=batch_idx,
                                model=model,
                                model_name=model.__class__.__name__,
                                sampler=sampler,
                            )

                            logger.info(
                                f"New best model saved at batch {batch_idx} "
                                f"(mask_f1={current_score:.6f}, "
                                f"mean_error={mean_error:.6f})"
                            )

                # Reset sampler timer for next batch
                t_sample_start = time.time()
        except Exception as e:
            logger.warning("Exception during training loop:")
            logger.warning(str(e))
            raise e

        logger.info("Training complete.")

        # Run final validation after training completes
        # (inside context to keep cache_manager active)
        if val_sampler is not None and last_batch_idx >= 0:
            logger.info("Running final validation after training...")
            assert key is not None
            key, val_key = jax.random.split(key)
            try:
                final_val_metrics = evaluate_batches(
                    model=model,
                    sampler=val_sampler,
                    create_state_fn=create_state_fn,
                    compute_estimate_fn=compute_estimate_fn,
                    trainable_state=trainable_state,
                    estimate_type=estimate_type,
                    key=val_key,
                    num_batches=val_num_batches,
                    cache_manager=cache_manager,
                )
                # Use last_batch_idx + 1 as the step for final validation
                # to appear after the last training batch
                final_step = last_batch_idx + 1

                # Log all validation metrics generically
                for metric_name, value in final_val_metrics.items():
                    if metric_name == "errors":
                        continue
                    scalar = None
                    if isinstance(value, (int, float)):
                        scalar = float(value)
                    elif isinstance(value, np.ndarray) and value.ndim == 0:
                        scalar = float(value)
                    if scalar is not None:
                        logger.scalar(
                            f"val/{metric_name}", scalar, step=final_step
                        )

                # Build validation message with all scalar metrics
                mean_error = float(
                    final_val_metrics.get("mean_error", float("nan"))
                )
                final_val_msg = (
                    f"Final validation (step {final_step}): "
                    f"mean_error={mean_error:.5f}"
                )
                # Append all other scalar metrics
                for k, v in final_val_metrics.items():
                    if k in (
                        "errors",
                        "mean_error",
                        "num_batches",
                        "num_samples",
                    ):
                        continue
                    if isinstance(v, (int, float)):
                        final_val_msg += f", {k}={float(v):.5f}"
                    elif isinstance(v, np.ndarray) and v.ndim == 0:
                        final_val_msg += f", {k}={float(v):.5f}"
                logger.info(final_val_msg)
            except Exception as e:
                logger.error(f"Final validation failed: {e}")

        # Final cache flush (inside context while cache_manager is still active)
        if cache_manager is not None:
            cache_manager.flush()
