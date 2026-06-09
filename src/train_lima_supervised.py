"""Supervised kinematic training driver for LIMA (non-episodic sampler).

`src/main.py --mode train-supervised` routes through the episodic ``FluidEnv``,
whose ``reset()`` calls ``next_episode()``. That caps a run at one episode
(``episode_length <= num_files`` batches) and is unnecessary for plain
supervised training. This driver mirrors main.py's assembly but builds a
non-episodic sampler directly (``episode_length: 0`` + ``loop: true`` =>
infinite iteration) and calls ``train_supervised`` on it.
"""

import argparse
import time

import goggles as gg
import jax
import synthpix

from flowgym.checkpointing import CheckpointConfig
from flowgym.common.base import NNEstimatorTrainableState
from flowgym.make import make_estimator, select_gt
from flowgym.run_setup import prepare_configs
from flowgym.utils import setup_logging
from train_supervised import train_supervised

logger = setup_logging(debug=False, use_wandb=False)


def main() -> None:
    """Parse args, build a non-episodic sampler, and run supervised training.

    Raises:
        ValueError: If the output directory or trainable state is missing.
    """
    parser = argparse.ArgumentParser(description="LIMA supervised training.")
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    args.mode = "train-supervised"

    (
        dataset_config,
        _dataset_to_compare,
        estimator_config,
        out_dir,
        validation_settings,
        _caching_config,
        _wandb_setup,
    ) = prepare_configs(args)

    key = jax.random.PRNGKey(dataset_config["seed"])
    key, subkey = jax.random.split(key)

    # Non-episodic sampler (iterates indefinitely with loop: true).
    sampler = synthpix.make(
        dataset_config, load_from=dataset_config.get("load_from")
    )
    batch = next(sampler)
    gt = select_gt(estimator_config["estimate_type"], batch)

    estimate_shape = estimator_config.get("estimate_shape", None)
    if estimate_shape is not None:
        estimate_shape = (batch.images1.shape[0], *tuple(estimate_shape))
    else:
        estimate_shape = gt.shape

    trainable_state, create_state_fn, compute_estimate_fn, estimator = (
        make_estimator(
            estimator_config,
            image_shape=(
                batch.images1.shape[0],
                *dataset_config["image_shape"],
            ),
            estimate_shape=estimate_shape,
            load_from=estimator_config.get("load_from", None),
            rng=subkey,
        )
    )

    val_sampler = None
    val_interval = None
    val_num_batches = 1
    if validation_settings is not None:
        val_dataset_config = validation_settings["dataset_config"]
        val_interval = validation_settings.get("interval")
        val_num_batches = validation_settings.get("num_batches", 1)
        val_sampler = synthpix.make(
            val_dataset_config, load_from=val_dataset_config.get("load_from")
        )

    if out_dir is None:
        raise ValueError("out_dir must be provided for training.")
    if not isinstance(trainable_state, NNEstimatorTrainableState):
        raise ValueError(
            "trainable_state must be an NNEstimatorTrainableState; "
            "lima_piv must be a trainable NN estimator."
        )

    ckpt_cfg = CheckpointConfig.from_configs(estimator_config, dataset_config)
    try:
        train_supervised(
            estimator=estimator,
            estimator_config=estimator_config,
            trainable_state=trainable_state,
            out_dir=out_dir,
            checkpoint_config=ckpt_cfg,
            create_state_fn=create_state_fn,
            compute_estimate_fn=compute_estimate_fn,
            sampler=sampler,
            val_sampler=val_sampler,
            val_interval=val_interval,
            val_num_batches=val_num_batches,
            num_batches=dataset_config.get("num_batches", 1000),
            estimate_type=estimator_config["estimate_type"],
            save_every=dataset_config.get("save_every", 100),
            log_every=dataset_config.get("log_every", 1),
            save_only_best=dataset_config.get("save_only_best", False),
            key=key,
            cache_manager=None,
        )
    finally:
        sampler.shutdown()
        time.sleep(3)
        gg.finish()


if __name__ == "__main__":
    main()
