"""Optax loss functions registry and builder."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from goggles import get_logger
from optax import losses as optax_losses

logger = get_logger(__name__)

# Type aliases
LossFn = Callable[[jax.Array, jax.Array], jax.Array]


def _l1_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute element-wise L1 (absolute) loss.

    Args:
        predictions: Predicted values.
        targets: Target values.

    Returns:
        Element-wise L1 loss.
    """
    return jnp.abs(jnp.subtract(predictions, targets))


# Registry uses Any since optax loss functions have heterogeneous signatures
LOSS_REGISTRY: dict[str, Any] = {
    # Basic regression losses
    "l1": _l1_loss,
    "l2": optax.l2_loss,
    "squared_error": optax.squared_error,
    "huber": optax.huber_loss,
    "smooth_l1": lambda preds, targets: optax.huber_loss(
        preds, targets, delta=1.0
    ),
    "log_cosh": optax.log_cosh,
    # Similarity losses
    "cosine_distance": optax.cosine_distance,
    "cosine_similarity": optax.cosine_similarity,
    # Classification losses
    "hinge": optax.hinge_loss,
    "perceptron": optax_losses.perceptron_loss,
    "sigmoid_binary_cross_entropy": optax.sigmoid_binary_cross_entropy,
    "sigmoid_focal": optax.sigmoid_focal_loss,
    "softmax_cross_entropy": optax.softmax_cross_entropy,
    "softmax_cross_entropy_with_integer_labels": (
        optax.softmax_cross_entropy_with_integer_labels
    ),
    "safe_softmax_cross_entropy": optax.safe_softmax_cross_entropy,
    "poly_loss_cross_entropy": optax_losses.poly_loss_cross_entropy,
    # Multiclass losses
    "multiclass_hinge": optax_losses.multiclass_hinge_loss,
    "multiclass_perceptron": optax_losses.multiclass_perceptron_loss,
    # Dice losses (segmentation)
    "dice": optax_losses.dice_loss,
    "binary_dice": optax_losses.binary_dice_loss,
    "multiclass_generalized_dice": (
        optax_losses.multiclass_generalized_dice_loss
    ),
    # KL divergence
    "kl_divergence": optax.kl_divergence,
    "kl_divergence_with_log_targets": (
        optax_losses.kl_divergence_with_log_targets
    ),
    "convex_kl_divergence": optax.convex_kl_divergence,
    # Contrastive/ranking losses
    "ntxent": optax.ntxent,
    "triplet_margin": optax_losses.triplet_margin_loss,
    "ranking_softmax": optax_losses.ranking_softmax_loss,
    # CTC loss
    "ctc": optax.ctc_loss,
}

REDUCTION_REGISTRY: dict[str, Callable[[jax.Array], jax.Array]] = {
    "mean": jnp.mean,
    "sum": jnp.sum,
    "none": lambda x: x,
}


def build_loss_from_config(config: dict[str, Any]) -> LossFn:
    """Build a loss function from a config.

    Config format:
        name: The loss function name (from LOSS_REGISTRY)
        kwargs: Optional kwargs to pass to the base loss function
        reduction: Optional reduction to apply ("mean", "sum", "none")

    Example:
        config = {
            "name": "huber",
            "kwargs": {"delta": 1.0},
            "reduction": "mean"
        }

    Args:
        config: Configuration dictionary for the loss function.

    Returns:
        A loss function that takes (predictions, targets) and returns scalar.

    Raises:
        ValueError: If config is invalid or loss name is unknown.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Loss config must be a dictionary; got {type(config).__name__}"
        )

    if "name" not in config:
        raise ValueError("Loss config must contain a 'name' field.")

    cfg = config.copy()
    name = str(cfg.pop("name")).lower()
    kwargs = dict(cfg.get("kwargs", {}) or {})
    reduction_name = str(cfg.get("reduction", "mean")).lower()

    base_loss_fn = LOSS_REGISTRY.get(name)
    if base_loss_fn is None:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss name '{name}'. Available: {available}")

    reduction_fn = REDUCTION_REGISTRY.get(reduction_name)
    if reduction_fn is None:
        available = ", ".join(sorted(REDUCTION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reduction '{reduction_name}'. Available: {available}"
        )

    def loss_fn(predictions: jax.Array, targets: jax.Array) -> jax.Array:
        elementwise = base_loss_fn(predictions, targets, **kwargs)
        return reduction_fn(elementwise)

    return loss_fn
