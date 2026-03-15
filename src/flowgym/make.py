"""Module for compiling, saving, and loading flow field estimator models."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from synthpix.types import SynthpixBatch

try:
    import torch
except ImportError:
    torch = None  # type: ignore

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import synthpix
from flax.core import FrozenDict, freeze
from goggles import get_logger
from goggles.history.types import History

# Models
from flowgym.common.base import Estimator
from flowgym.common.base.trainable_state import (
    EstimatorTrainableState,
    NNEstimatorTrainableState,
)
from flowgym.training.optimizer import build_optimizer_from_config
from flowgym.types import (
    CachePayload,
    CompiledComputeEstimateFn,
    CompiledCreateStateFn,
    PRNGKey,
)
from flowgym.utils import DEBUG, MissingDependency

logger = get_logger(__name__)


def make_manager(ckpt_dir: Path, keep: int = 3) -> ocp.CheckpointManager:
    """Make a CheckpointManager backed by a PyTreeCheckpointer.

    Args:
        ckpt_dir: Root output directory for this experiment/run.
        keep: Number of checkpoints to keep.

    Returns:
        The CheckpointManager instance.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=keep, create=True)
    return ocp.CheckpointManager(
        ckpt_dir,
        options=options,
    )


@overload
def compile_model(
    model: Estimator, estimates: None, jit: bool = True, history_size: int = 1
) -> tuple[
    None,
    CompiledComputeEstimateFn,
]: ...


@overload
def compile_model(
    model: Estimator,
    estimates: jnp.ndarray,
    jit: bool = True,
    history_size: int = 1,
) -> tuple[
    CompiledCreateStateFn,
    CompiledComputeEstimateFn,
]: ...


def compile_model(
    model: Estimator,
    estimates: jnp.ndarray | None,
    jit: bool = True,
    history_size: int = 1,
) -> tuple[
    CompiledCreateStateFn | None,
    CompiledComputeEstimateFn,
]:
    """Compile the model for JAX.

    Args:
        model: The flow field estimator model.
        estimates: Example estimates for shape inference.
        jit: Whether to use JIT compilation.
        history_size: The size of the history for the model.

    Returns:
        Compiled functions.
    """
    create_state_fn: CompiledCreateStateFn | None
    if estimates is not None:

        def create_state_fn_impl(images: jnp.ndarray, rng: PRNGKey) -> History:
            return model.create_state(
                images,
                estimates=estimates,
                image_history_size=history_size,
                estimate_history_size=history_size,
                rng=rng,
            )

        create_state_fn = create_state_fn_impl
    else:
        create_state_fn = None

    def compute_estimate_fn(
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
        cache_payload: CachePayload | None = None,
    ) -> tuple[History, dict]:
        return model(
            images, state, trainable_state, cache_payload=cache_payload
        )

    if jit and not DEBUG:
        return (
            jax.jit(create_state_fn) if create_state_fn is not None else None
        ), jax.jit(compute_estimate_fn)
    return create_state_fn, compute_estimate_fn


def save_model(
    state: NNEstimatorTrainableState,
    out_dir: str | Path,
    step: int | None = None,
    model: Estimator | None = None,
    model_name: str | None = None,
    sampler: Any | None = None,
    keep: int = 3,
) -> str:
    """Save a training checkpoint using Orbax.

    Checkpoint saved to out_dir/checkpoints/<model_name>/step_<step>.

    Args:
        state: The trainable state to save (PyTree).
        out_dir: Root output directory for this experiment/run.
        step: Training step/batch index. If None, reads from `state.step`.
        model: The model instance (to extract optimizer_config).
        model_name: Optional model name for directory nesting.
        sampler: The sampler instance to save (must be Sampler with
            Grain scheduler for full state saving).
        keep: Number of checkpoints to keep.

    Returns:
        The saved step directory path.

    Raises:
        ValueError: If step is not provided and state has no 'step' attr.
    """
    out_dir = Path(out_dir)
    out_dir = out_dir.resolve()

    # Decide on a step number
    if step is None:
        if hasattr(state, "step"):
            step = int(state.step)
        else:
            raise ValueError("step not provided and state has no 'step' attr")
    step = int(step)

    # Nesting: out_dir/checkpoints/<model_name>/step_<step>
    parts = [out_dir, "checkpoints"]
    if model_name is not None:
        parts.append(model_name)

    ckpt_root = Path(*parts)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # We use the 'ocp.args' API
    item = {
        "step": step,
        "params": state.params,
        "opt_state": state.opt_state,
        "extras": state.extras,
    }

    # Define the save arguments
    save_args: dict[str, ocp.args.CheckpointArgs] = {
        "state": ocp.args.StandardSave(item)  # pyright: ignore[reportCallIssue]
    }

    # Extract and save optimizer config separately (as it contains strings)
    if model is not None:
        opt_cfg = getattr(
            model, "optimizer_config", getattr(model, "opt_config", None)
        )
        if opt_cfg is not None:
            save_args["opt_config"] = ocp.args.JsonSave(opt_cfg)  # pyright: ignore[reportCallIssue]

    if sampler is not None:
        try:
            sampler_args = synthpix.checkpoint_args(sampler)
            # Flatten Composite args to root level for atomic saving
            # of (state, sampler, grain) side-by-side.
            save_args.update(
                sampler_args.items()  # pyright: ignore[reportAttributeAccessIssue]
            )
        except Exception as e:
            logger.warning(f"Failed to prepare sampler checkpoint args: {e}")

    with make_manager(ckpt_root, keep=keep) as mngr:
        mngr.save(step=step, args=ocp.args.Composite(**save_args))
        mngr.wait_until_finished()

    return str(ckpt_root / str(step))


def load_model(
    ckpt_dir: str | Path,
    template_state: NNEstimatorTrainableState,
    mode: Literal["resume", "params_only"] = "params_only",
) -> EstimatorTrainableState:
    """Load a checkpoint for EstimatorTrainableState using Orbax.

    Args:
        ckpt_dir: Path to checkpoint directory with 'extras', 'params',
            etc.
        template_state: A template EstimatorTrainableState with the
            correct structure, dtypes and static fields.
        mode:
          - "resume": restore full state (params + opt + step + extras).
          - "params_only": restore only params, reset step to 0.

    Returns:
        The restored EstimatorTrainableState.

    Raises:
        FileNotFoundError: If checkpoint path does not exist.
        ValueError: If no checkpoints found or required keys missing.
        Exception: If restore/casting operations fail.
    """
    ckpt_dir = Path(ckpt_dir).absolute()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_dir}")

    logger.info(f"Loading checkpoint from {ckpt_dir} in mode {mode}")

    # If ckpt_dir is a specific step directory (e.g. ends in a number),
    # point the manager to the parent directory and restore that specific step.
    if ckpt_dir.name.isdigit():
        mngr_dir = ckpt_dir.parent
        step = int(ckpt_dir.name)
    else:
        mngr_dir = ckpt_dir
        step = None

    # Use a context manager for the manager
    with make_manager(mngr_dir, keep=1) as mngr:
        if step is None:
            step = mngr.latest_step()

        if step is None:
            raise ValueError(f"No checkpoints found in: {ckpt_dir}")

        # Generic global restore (StandardRestore() loads everything)
        try:
            # Use SingleDeviceSharding as fallback to handle multi-GPU
            # checkpoints on single-GPU systems.
            fallback_sharding = jax.sharding.SingleDeviceSharding(
                jax.devices()[0]
            )

            # Create StandardRestore with kwargs to bypass type checking
            # limitations in older Orbax type stubs
            restore_kwargs: dict[str, Any] = {
                "fallback_sharding": fallback_sharding,
            }
            restore_args_dict: dict[str, ocp.args.CheckpointArgs] = {
                "state": ocp.args.StandardRestore(**restore_kwargs)
            }
            # Attempt to restore opt_config if it exists
            if (ckpt_dir / "opt_config").exists() or (
                mngr_dir / str(step) / "opt_config"
            ).exists():
                restore_args_dict["opt_config"] = ocp.args.JsonRestore()

            restore_args = ocp.args.Composite(**restore_args_dict)
            # Restore step matching directory if possible
            full_payload = mngr.restore(step, args=restore_args)
            state_dict = full_payload.get("state", {})
            opt_config = full_payload.get("opt_config", None)
        except Exception as e:
            logger.error(f"Failed to restore with StandardRestore: {e}")
            raise

        # Now manually reconstruct based on mode
        if mode == "params_only":
            # Just extract params, ignore the rest
            if "params" not in state_dict:
                raise ValueError(
                    f"Checkpoint missing 'params': {state_dict.keys()}"
                )

            # Use the loaded params to update the template
            # Note: state_dict['params'] structure should match template.
            restored_params = freeze(state_dict["params"])
            new_opt_state = template_state.tx.init(restored_params)

            ts = cast(Any, template_state).replace(
                params=restored_params,
                opt_state=new_opt_state,
                step=0,
            )
            return ts

        elif mode == "resume":
            logger.info("Resuming full state from checkpoint.")

            extras = FrozenDict(state_dict.get("extras", {}))
            opt_state = state_dict.get("opt_state", None)
            params = FrozenDict(state_dict.get("params"))
            loaded_step = state_dict.get("step", 0)

            # If optimization state is present, we might want to check it.
            # However, StandardRestore returns the raw structure.
            # Flax TrainState expects opt_state to be passed to replace().

            # We don't restore 'tx' from checkpoint usually (it shouldn't
            # exist unless explicitly saved; we skip it).

            replacements = {
                "step": loaded_step,
                "params": params,
                "extras": extras,
            }

            # If we restored an optimizer config, rebuild the optimizer
            # to match the structure of the restored opt_state.
            if opt_config is not None:
                try:
                    tx = build_optimizer_from_config(opt_config)
                    replacements["tx"] = tx
                    logger.info("Restored and rebuilt optimizer from config.")
                except Exception as e:
                    logger.warning(
                        f"Failed to rebuild optimizer from config: {e}"
                    )

            if opt_state is not None:
                # StandardRestore restores tuples as lists (EmptyState->[]).
                # Interpret restored structure using template's structure
                # for type safety (Tuple vs List).
                if "tx" in replacements:
                    template_opt_state = replacements["tx"].init(params)
                else:
                    template_opt_state = getattr(
                        template_state, "opt_state", None
                    )

                if template_opt_state is not None:
                    try:
                        treedef = jax.tree_util.tree_structure(
                            template_opt_state
                        )
                        opt_state = jax.tree_util.tree_unflatten(
                            treedef, jax.tree_util.tree_leaves(opt_state)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cast opt_state: {e}")

                replacements["opt_state"] = opt_state

            return cast(Any, template_state).replace(**replacements)

        else:
            raise ValueError(f"Unknown mode: {mode}")


@overload
def make_estimator(
    model_config: dict,
    image_shape: tuple,
    estimate_shape: tuple,
    load_from: str | None = None,
    rng: PRNGKey | int | None = None,
) -> tuple[
    EstimatorTrainableState,
    CompiledCreateStateFn,
    CompiledComputeEstimateFn,
    Estimator,
]: ...


@overload
def make_estimator(
    model_config: dict,
    image_shape: tuple,
    estimate_shape: None,
    load_from: str | None = None,
    rng: PRNGKey | int | None = None,
) -> tuple[
    EstimatorTrainableState,
    None,
    CompiledComputeEstimateFn,
    Estimator,
]: ...


@overload
def make_estimator(
    model_config: dict,
    image_shape: tuple | None = None,
    estimate_shape: tuple | None = None,
    load_from: str | None = None,
    rng: PRNGKey | int | None = None,
) -> tuple[
    EstimatorTrainableState | None,
    CompiledCreateStateFn | None,
    CompiledComputeEstimateFn,
    Estimator,
]: ...


def make_estimator(
    model_config: dict,
    image_shape: tuple | None = None,
    estimate_shape: tuple | None = None,
    load_from: str | None = None,
    rng: PRNGKey | int | None = None,
) -> tuple[
    EstimatorTrainableState | None,
    CompiledCreateStateFn | None,
    CompiledComputeEstimateFn,
    Estimator,
]:
    """Create an instance of the flow field estimator.

    If load_from is not provided, a new trainable state is created.

    model_config keys:
    - estimator: Name of the estimator.
    - estimator_type: Type of the estimator ("flow" or "density").
    - config: Configuration dictionary for the estimator.

    Args:
        model_config: Configuration dictionary for the estimator.
        image_shape: Shape of the input images (B, H, W).
        estimate_shape: Shape of the estimate. Defaults to (B, H, W, 2).
        load_from: Path to load the trained model state.
        rng: Random number generator key or seed.

    Returns:
        EstimatorTrainableState: The trainable state of the model.
        callable: Function to create the model state.
        callable: Function to compute the model estimate.
        Estimator: The model instance.

    Raises:
        ValueError: If estimator not found or model loading fails.
    """
    # Import here to avoid circular dependency
    from flowgym import ALL_ESTIMATORS as ESTIMATORS  # noqa: PLC0415

    # Extract the estimator class from the config
    if model_config["estimator"] not in ESTIMATORS:
        raise ValueError(f"Estimator {model_config['estimator']} not found.")
    model_class = ESTIMATORS.get(model_config["estimator"])
    if model_class is None:
        raise ValueError(f"Estimator {model_config['estimator']} not found.")
    elif isinstance(model_class, MissingDependency):
        model_class()  # Raises MissingDependency error
        # Type narrowing: model_class is not MissingDependency here
        raise ValueError("Unreachable")  # pragma: no cover

    if rng is None:
        rng = jax.random.PRNGKey(0)
    elif isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)

    # Create the model instance
    model = cast(type[Estimator], model_class).from_config(
        model_config["config"]
        | {
            "estimate_shape": estimate_shape,
            "image_shape": image_shape,
            "rng": rng,
        }
    )
    logger.info("Model created successfully.")

    # Load or create the trainable state
    if load_from:
        if model_config["estimator"] == "raft_torch":
            if torch is None:
                raise ValueError("torch required for raft_torch model")
            checkpoint = torch.load(load_from, map_location="cuda")
            model_any = model  # type: Any
            model_any.raft.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            trained_state = None
        else:
            mode = model_config.get("load_mode", "params_only")
            template_state = model.create_trainable_state(
                jnp.zeros(image_shape, dtype=jnp.float32), key=rng
            )
            if isinstance(template_state, NNEstimatorTrainableState):
                trained_state = load_model(load_from, template_state, mode=mode)
            else:
                raise ValueError("Model is not a neural network estimator.")
        logger.info("Trainable state loaded successfully.")
    # Create a dummy input to initialize the trainable state
    elif image_shape is None:
        logger.warning(
            "image_shape not provided, trainable state cannot be created."
        )
        trained_state = None
    else:
        sample_images = jnp.zeros(image_shape, dtype=jnp.float32)
        trained_state = model.create_trainable_state(sample_images, key=rng)
        logger.info("Trainable state created successfully.")

    if estimate_shape is None:
        if image_shape is None:
            estimate_shape = None
        else:
            logger.info(
                "Estimate shape not provided, using default (B, H, W, 2)."
            )
            estimate_shape = (*tuple(image_shape), 2)

    # Create dummy estimates for shape inference
    if estimate_shape is None:
        dummy_estimates = None
    else:
        dummy_estimates = jnp.zeros(estimate_shape, dtype=jnp.float32)
    create_state_fn, compute_estimate_fn = compile_model(
        model,
        dummy_estimates,
        model_config["config"].get("jit", False) and not DEBUG,
        history_size=model_config["config"].get("history_size", 1),
    )
    logger.info("Model compiled successfully.")

    return trained_state, create_state_fn, compute_estimate_fn, model


def select_gt(estimate_type: str, batch: "SynthpixBatch") -> jnp.ndarray:
    """Select the ground truth based on the mode.

    Args:
        estimate_type: The type of the estimator.
        batch: The batch of data.

    Returns:
        The ground truth.

    Raises:
        ValueError: If batch params are None or invalid estimate_type.
    """
    if estimate_type == "flow":
        return batch.flow_fields
    elif estimate_type == "density":
        if batch.params is None:
            raise ValueError("Batch params None, cannot select density gt")
        return batch.params.seeding_densities
    else:
        raise ValueError(
            f"Invalid mode: {estimate_type}. Use 'flow' or 'density'."
        )
