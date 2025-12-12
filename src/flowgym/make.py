"""Module for compiling, saving, and loading flow field estimator models."""

from collections.abc import Callable
from pathlib import Path
import jax
import os
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flowgym.common.base import Estimator
from flowgym.common.base.trainable_states import EstimatorTrainableState
from flowgym.utils import DEBUG, MissingDependency
from goggles import get_logger
from goggles.history.types import History
try:
    from synthpix.types import SynthpixBatch
except ImportError:
    SynthpixBatch = None
from flowgym.types import PRNGKey

# Models
from flowgym import ALL_ESTIMATORS as ESTIMATORS

logger = get_logger(__name__)


def compile_model(
    model: Estimator,
    estimates: jnp.ndarray | None,
    jit: bool = True,
    history_size: int = 1,
) -> tuple[Callable | None, Callable]:
    """Compile the model for JAX.

    Args:
        model: The flow field estimator model.
        estimates: Example estimates for shape inference.
        jit: Whether to use JIT compilation.
        history_size: The size of the history for the model.

    Returns:
        Compiled functions.
    """
    if estimates is not None:

        def create_state_fn(images: jnp.ndarray, rng: PRNGKey) -> History:
            return model.create_state(
                images,
                estimates=estimates,
                image_history_size=history_size,
                estimate_history_size=history_size,
                rng=rng,
            )

    else:
        create_state_fn = None  # type: ignore

    def compute_estimate_fn(
        images: jnp.ndarray,
        state: History,
        trainable_state: EstimatorTrainableState,
    ) -> tuple[History, dict]:
        return model(images, state, trainable_state)

    if jit and not DEBUG:
        return (
            jax.jit(create_state_fn) if create_state_fn is not None else None
        ), jax.jit(compute_estimate_fn)
    return create_state_fn, compute_estimate_fn


def save_model(
    trainable_state: EstimatorTrainableState,
    out_dir: str,
    model_name: str,
) -> None:
    """Save a trainable state as an Orbax checkpoint.

    Args:
        trainable_state:
            The trainable state of the model.
        out_dir: Directory where to save the model.
        model_name: Name of the model.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, model_name)

    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(path, trainable_state)
    logger.info(f"Model saved to {path}.")


def load_model(
    trainable_state_path: str,
    template_state: EstimatorTrainableState,
) -> EstimatorTrainableState:
    """Load the model state from an Orbax checkpoint.

    Args:
        trainable_state_path: Path to the saved model.
        template_state: A template trainable state for restoring.
    
    Returns:
        The loaded trainable state of the model.
    """
    checkpointer = ocp.StandardCheckpointHandler()
    
    restored_state = checkpointer.restore(
        Path(trainable_state_path), template_state # type: ignore
    )
    logger.info(f"Model loaded from {trainable_state_path}.")
    return restored_state

def make_estimator(
    model_config: dict,
    image_shape: tuple | None = None,
    estimate_shape: tuple | None = None,
    load_from: str | None = None,
    rng: PRNGKey | int = jax.random.PRNGKey(0),
) -> tuple[EstimatorTrainableState | None, Callable | None, Callable, Estimator]:
    """Create an instance of the flow field estimator.

    If load_from is not provided, a new trainable state is created, even for
    models that don't require training. The trainable state is then simply a None.

    model_config is a dictionary with the following keys:
    - estimator: Name of the estimator.
    - estimator_type: Type of the estimator ("flow" or "density").
    - config: Configuration dictionary for the estimator.


    Args:
        model_config: Configuration dictionary for the estimator.
        image_shape: Shape of the input images (B, H, W).
        estimate_shape:
            Shape of the estimate. If not provided, it will default to (B, H, W, 2).
        load_from: Path to load the trained model state. Defaults to None.
        rng: Random number generator key or seed.

    Returns:
        EstimatorTrainableState: The trainable state of the model.
        callable: Function to create the model state.
        callable: Function to compute the model estimate.
        Estimator: The model instance.
    """
    # Extract the estimator class from the config
    if model_config["estimator"] not in ESTIMATORS:
        raise ValueError(f"Estimator {model_config['estimator']} not found.")
    model_class = ESTIMATORS.get(model_config["estimator"])
    if model_class is None:
        raise ValueError(f"Estimator {model_config['estimator']} not found.")
    elif isinstance(model_class, MissingDependency):
        model_class()
        
    # Create the model instance
    model = model_class.from_config( # type: ignore
        model_config["config"] | {"estimate_shape": estimate_shape}
    )
    logger.info("Model created successfully.")
    
    # Load or create the trainable state
    if load_from:
        if model_config["estimator"] == "raft_torch":
            import torch

            checkpoint = torch.load(load_from, map_location="cuda")  # or "cpu"
            model.raft.load_state_dict(checkpoint["model_state_dict"], strict=False)
            trained_state = None
        else:
            if isinstance(rng, int):
                rng = jax.random.PRNGKey(rng)
            template_state = model.create_trainable_state(
                jnp.zeros(image_shape, dtype=jnp.float32), key=rng
            )
            trained_state = load_model(load_from, template_state)
        logger.info("Trainable state loaded successfully.")
    else:
        # Create a dummy input to initialize the trainable state
        if image_shape is None:
            logger.warning(
                "image_shape not provided, trainable state cannot be created."
            )
            trained_state = None
        else:
            sample_images = jnp.zeros(image_shape, dtype=jnp.float32)
            if isinstance(rng, int):
                rng = jax.random.PRNGKey(rng)
            trained_state = model.create_trainable_state(sample_images, key=rng)
            logger.info("Trainable state created successfully.")

    if estimate_shape is None:
        if image_shape is None:
            estimate_shape = None
        else:
            logger.info("Estimate shape not provided, using default (B, H, W, 2).")
            estimate_shape = tuple(image_shape) + (2,)

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


def select_gt(estimate_type: str, batch: SynthpixBatch) -> jnp.ndarray:
    """Select the ground truth based on the mode.

    Args:
        estimate_type: The type of the estimator.
        batch: The batch of data.

    Returns:
        The ground truth.
    """
    if estimate_type == "flow":
        return batch.flow_fields
    elif estimate_type == "density":
        if batch.params is None:
            raise ValueError(
                "Batch params are None, cannot select density ground truth."
            )
        return batch.params.seeding_densities
    else:
        raise ValueError(f"Invalid mode: {estimate_type}. Choose 'flow' or 'density'.")
