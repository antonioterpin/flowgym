"""Consensus algorithm for robust estimation of flow parameters."""

import jax
import jax.numpy as jnp
from goggles import get_logger
from jax import lax

from flowgym.common.filters import sobel
from flowgym.flow.consensus.regularizers import total_regularization_loss
from flowgym.flow.dis.process import (
    extract_patches,
    photometric_error_with_patches,
)

logger = get_logger(__name__)


def z_objective(
    consensus_flow: jnp.ndarray,
    flows: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    rho: float = 1.0,
    regularizer_list: list | None = None,
    regularizer_weights: dict[str, float] | None = None,
) -> jnp.ndarray:
    r"""Compute the Z objective function for consensus-based flow estimation.

    In the Boyd et al. notation, this corresponds to the function
    minimized over z in the second step of the ADMM algorithm.
    It includes the consensus term and regularization on the consensus
    flow estimate. It is expressed in the scaled form:

    .. math::
        reg(z) + \\frac{\\rho}{2} \\| x - z + u \\|^2

    where
        - :math:`x` is the current flow estimate,
        - :math:`u` is the dual variable,

    Args:
        consensus_flow: Current consensus flow estimate.
        flows: Array of flow estimates from different agents.
        consensus_dual: Dual variable for consensus.
        rho: Penalty parameter for the consensus term.
        regularizer_list: List of regularization functions to apply.
        regularizer_weights: Weights for each regularization term.

    Returns:
        Computed Z objective value.
    """
    # avoid mutable default arguments
    if regularizer_list is None:
        regularizer_list = []
    if regularizer_weights is None:
        regularizer_weights = {}

    # Compute the consensus term
    residuals = (
        flows - consensus_flow[None, ...] + consensus_dual
    )  # (N, H, W, 2)

    # Consensus term scaled by rho / 2
    consensus_term = 0.5 * rho * jnp.sum(residuals**2)

    # Regularization on z only
    # Note: this is the g(z) if following Boyd et al. notation
    reg_term = total_regularization_loss(
        consensus_flow, regularizer_list, regularizer_weights
    )

    return consensus_term + reg_term


def flows_objective(
    flows: jnp.ndarray,
    consensus_flow: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    initial_flows: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    objective_type: str = "l2",
    rho: float = 1.0,
) -> jnp.ndarray:
    """Compute the objective function for flow estimates.

    Args:
        flows: Array of flow estimates from different agents.
        consensus_flow: Current consensus flow estimate.
        consensus_dual: Dual variable for consensus.
        initial_flows: Initial flow estimates for comparison.
        weights: Weights to apply to the anchor term.
        objective_type: Type of objective function to compute,
            either "l2", "l1" or "huber".
        rho: Penalty parameter for the consensus term.

    Returns:
        Computed objective value for flow estimates.
    """
    # Calculate the anchor term
    if objective_type == "l2":
        anchor = flows - initial_flows
        data_term = anchor**2
        if weights is not None:
            # If weights are provided, apply them to the anchor term
            data_term *= weights[..., jnp.newaxis]
        data_term = jnp.sum(data_term)
    elif objective_type == "l1":
        anchor = flows - initial_flows
        data_term = jnp.abs(anchor)
        if weights is not None:
            # If weights are provided, apply them to the anchor term
            data_term *= weights[..., jnp.newaxis]
        data_term = jnp.sum(data_term)
    elif objective_type == "huber":
        delta = 1.0  # Huber parameter, can be adjusted or passed as an argument
        if weights is None:
            weights = jnp.ones(flows.shape[:-1])  # (N,H,W)
        sqrt_p = jnp.sqrt(weights)[..., None]  # (N,H,W,1)
        diff = sqrt_p * (flows - initial_flows)  # (N,H,W,2)
        abs_diff = jnp.abs(diff)
        quadratic = jnp.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        data_term = 0.5 * quadratic**2 + delta * linear
        data_term = jnp.sum(data_term)

    # Residuals for the consensus term
    residuals = flows - consensus_flow[jnp.newaxis, ...] + consensus_dual

    # Consensus term
    consensus_term = 0.5 * rho * jnp.sum(residuals**2)

    return data_term + consensus_term


def weights_and_anchors(
    anchor_flows: jnp.ndarray,
    weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wrapper function to return weights and anchors.

    Args:
        anchor_flows: Array of anchor flow estimates.
        weights: Array of weights for the flow estimates.

    Returns:
        A tuple containing the weights and anchors.
    """
    return weights, anchor_flows


def _extract_patch_params(cfg: dict) -> tuple[int, int]:
    """Extract and validate patch size and stride from configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Tuple of (patch_size, patch_stride).

    Raises:
        ValueError: If parameters are invalid.
    """
    if "patch_size" not in cfg:
        logger.warning(
            "No patch_size specified in the configuration. Using 3 as default."
        )
        patch_size = 3
    else:
        patch_size = cfg["patch_size"]
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be an odd number to ensure a center pixel."
            )

    if "patch_stride" not in cfg:
        logger.warning(
            "No patch_stride specified in the configuration. Using 1 as default."
        )
        patch_stride = 1
    else:
        patch_stride = cfg["patch_stride"]
        if not isinstance(patch_stride, int) or patch_stride <= 0:
            raise ValueError("patch_stride must be a positive integer.")

    return patch_size, patch_stride


def _make_weights_photometric(
    flows: jnp.ndarray,
    prevs: jnp.ndarray,
    currs: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    """Compute weights based on photometric error.

    Args:
        flows: Flow estimates, shape (B, N, H, W, 2).
        prevs: Previous frames, shape (B, H, W).
        currs: Current frames, shape (B, H, W).
        cfg: Configuration dictionary.

    Returns:
        Weights, shape (B, N, H, W).
    """
    B, N, H, W, _ = flows.shape
    patch_size, patch_stride = _extract_patch_params(cfg)
    half = patch_size // 2

    # Tile prevs and currs to match the number of flow estimates
    prevs = jnp.tile(prevs[:, jnp.newaxis, ...], (1, N, 1, 1))
    currs = jnp.tile(currs[:, jnp.newaxis, ...], (1, N, 1, 1))

    # Reshape to use vmap
    prevs_flat = prevs.reshape(-1, H, W)
    currs_flat = currs.reshape(-1, H, W)
    flows_flat = flows.reshape(-1, H, W, 2)

    # Compute the photometric error
    photometric_errors = jax.vmap(
        photometric_error_with_patches, in_axes=(0, 0, 0, None, None, None)
    )(prevs_flat, currs_flat, flows_flat, patch_size, patch_stride, True)

    # Compute weights based on the inverse of the photometric errors
    weights = 1.0 / jnp.maximum(photometric_errors, 1e-6)

    # Pad weights to match the original shape
    weights = jnp.pad(
        weights,
        ((0, 0), (half, half), (half, half)),
        mode="constant",
        constant_values=0,
    )

    weights = weights.reshape(B, N, H, W)
    if "weights" in cfg:
        logger.warning(
            "Weights specified in the configuration will be ignored "
            "when using 'photometric' as weights_type."
        )

    return weights


def _make_weights_photograd(
    flows: jnp.ndarray,
    prevs: jnp.ndarray,
    currs: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    """Compute weights based on photometric error and gradient magnitude.

    Args:
        flows: Flow estimates, shape (B, N, H, W, 2).
        prevs: Previous frames, shape (B, H, W).
        currs: Current frames, shape (B, H, W).
        cfg: Configuration dictionary.

    Returns:
        Weights, shape (B, N, H, W).
    """
    B, N, H, W, _ = flows.shape
    patch_size, patch_stride = _extract_patch_params(cfg)
    half = patch_size // 2

    # Tile prevs and currs to match the number of flow estimates
    prevs = jnp.tile(prevs[:, jnp.newaxis, ...], (1, N, 1, 1))
    currs = jnp.tile(currs[:, jnp.newaxis, ...], (1, N, 1, 1))

    # Reshape to use vmap
    prevs_flat = prevs.reshape(-1, H, W)
    currs_flat = currs.reshape(-1, H, W)
    flows_flat = flows.reshape(-1, H, W, 2)

    # Compute the photometric error
    photometric_errors = jax.vmap(
        photometric_error_with_patches, in_axes=(0, 0, 0, None, None, None)
    )(prevs_flat, currs_flat, flows_flat, patch_size, patch_stride, True)

    # Normalize the photometric errors by number of pixels in a patch
    num_pixels = patch_size * patch_size
    photometric_errors /= num_pixels

    # Compute gradients using Sobel filter
    kx, ky = sobel()
    Ix = lax.conv_general_dilated(
        prevs_flat[:, jnp.newaxis, ...],
        kx[jnp.newaxis, jnp.newaxis, ...],
        (1, 1),
        padding="VALID",
    )[:, 0]
    Iy = lax.conv_general_dilated(
        prevs_flat[:, jnp.newaxis, ...],
        ky[jnp.newaxis, jnp.newaxis, ...],
        (1, 1),
        padding="VALID",
    )[:, 0]

    # Pad gradients to match the original shape
    padded_grads = jnp.pad(
        jnp.stack([Ix, Iy], axis=1),
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )
    grad_patches = jax.vmap(extract_patches, in_axes=[0, None, None])(
        padded_grads,
        patch_size,
        patch_stride,
    ).reshape(B * N, H - 2 * half, W - 2 * half, patch_size, patch_size, 2)

    # Compute weights based on the inverse of the photometric errors
    # and proportional to the gradient magnitudes
    weights = jnp.sum(grad_patches**2, axis=(3, 4, 5)) / (
        photometric_errors + 1e-4
    )

    weights = jnp.pad(
        weights,
        ((0, 0), (half, half), (half, half)),
        mode="constant",
        constant_values=0,
    )

    weights = weights.reshape(B, N, H, W)
    if "weights" in cfg:
        logger.warning(
            "Weights specified in the configuration will be ignored "
            "when using 'photograd' as weights_type."
        )

    return weights


def _make_weights_gradient(
    flows: jnp.ndarray,
    prevs: jnp.ndarray,
    currs: jnp.ndarray,
    cfg: dict,
) -> jnp.ndarray:
    """Compute weights based on gradient magnitude.

    Args:
        flows: Flow estimates, shape (B, N, H, W, 2).
        prevs: Previous frames, shape (B, H, W).
        currs: Current frames, shape (B, H, W).
        cfg: Configuration dictionary.

    Returns:
        Weights, shape (B, N, H, W).
    """
    B, N, H, W, _ = flows.shape
    patch_size, patch_stride = _extract_patch_params(cfg)
    half = patch_size // 2

    # Tile prevs and currs to match the number of flow estimates
    prevs = jnp.tile(prevs[:, jnp.newaxis, ...], (1, N, 1, 1))
    currs = jnp.tile(currs[:, jnp.newaxis, ...], (1, N, 1, 1))

    # Reshape to use vmap
    prevs_flat = prevs.reshape(-1, H, W)
    currs_flat = currs.reshape(-1, H, W)
    flows_flat = flows.reshape(-1, H, W, 2)

    # Compute gradients using Sobel filter
    kx, ky = sobel()
    Ix = lax.conv_general_dilated(
        prevs_flat[:, jnp.newaxis, ...],
        kx[jnp.newaxis, jnp.newaxis, ...],
        (1, 1),
        padding="VALID",
    )[:, 0]
    Iy = lax.conv_general_dilated(
        prevs_flat[:, jnp.newaxis, ...],
        ky[jnp.newaxis, jnp.newaxis, ...],
        (1, 1),
        padding="VALID",
    )[:, 0]

    # Pad gradients to match the original shape
    padded_grads = jnp.pad(
        jnp.stack([Ix, Iy], axis=1),
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )
    grad_patches = jax.vmap(extract_patches, in_axes=[0, None, None])(
        padded_grads,
        patch_size,
        patch_stride,
    ).reshape(B * N, H - 2 * half, W - 2 * half, patch_size, patch_size, 2)

    # Compute weights proportionally to the gradient magnitudes
    weights = jnp.sum(grad_patches**2, axis=(3, 4, 5))

    weights = jnp.pad(
        weights,
        ((0, 0), (half, half), (half, half)),
        mode="constant",
        constant_values=0,
    )

    weights = weights.reshape(B, N, H, W)
    if "weights" in cfg:
        logger.warning(
            "Weights specified in the configuration will be ignored "
            "when using 'gradient' as weights_type."
        )

    return weights


def _make_weights_list(flows: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    """Compute weights from a list in the configuration.

    Args:
        flows: Flow estimates, shape (B, N, H, W, 2).
        cfg: Configuration dictionary.

    Returns:
        Weights, shape (B, N, H, W).

    Raises:
        ValueError: If weights are not provided in configuration.
    """
    if "weights" not in cfg:
        raise ValueError(
            "Weights must be provided when weights_type is 'list'."
        )
    weights = jnp.array(cfg["weights"])
    if weights.ndim == 1:
        # If weights are 1D, we assume they are for each flow estimate
        weights = weights[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        weights = jnp.broadcast_to(weights, flows.shape[:-1])

    return weights


def _make_weights_none(flows: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    """Compute uniform weights.

    Args:
        flows: Flow estimates, shape (B, N, H, W, 2).
        cfg: Configuration dictionary.

    Returns:
        Weights, shape (B, N, H, W).
    """
    weights = jnp.ones(flows.shape[:-1])
    if "weights" in cfg:
        logger.warning(
            "Weights specified in the configuration will be ignored "
            "when using 'none' as weights_type."
        )

    return weights


def _normalize_weights(
    weights: jnp.ndarray,
    normalization: str,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Apply normalization technique to weights.

    Args:
        weights: Weights to normalize, shape (B, N, H, W).
        normalization: Normalization technique to apply.
        epsilon: Small value to avoid division by zero.

    Returns:
        Normalized weights, shape (B, N, H, W).

    Raises:
        ValueError: If normalization technique is unknown.
    """
    if normalization == "per_batch":
        weights = weights / (
            jnp.sum(weights, axis=(1, 2, 3))[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            + epsilon
        )

    elif normalization == "per_pixel":
        weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + epsilon)

    elif normalization == "softmax_per_batch":
        # Flatten (N, H, W) into one axis for each batch
        B, N, H, W = weights.shape
        flat_weights = weights.reshape(B, -1)
        shifted = flat_weights - jnp.max(flat_weights, axis=1, keepdims=True)
        exp_weights = jnp.exp(shifted)
        softmax_flat = exp_weights / jnp.sum(exp_weights, axis=1, keepdims=True)
        weights = softmax_flat.reshape(B, N, H, W)

    elif normalization == "softmax_per_pixel":
        # Softmax over agent axis (N), per-pixel
        weights_exp = jnp.exp(weights - jnp.max(weights, axis=1, keepdims=True))
        weights = weights_exp / (jnp.sum(weights_exp, axis=1, keepdims=True))

    elif normalization == "max":
        # Winner-take-all, ties are split uniformly
        max_mask = (weights == jnp.max(weights, axis=1, keepdims=True)).astype(
            weights.dtype
        )
        weights = max_mask / (jnp.sum(max_mask, axis=1, keepdims=True) + epsilon)

    elif normalization != "none":
        raise ValueError(f"Unknown normalization technique: {normalization}")

    return weights


def make_weights(
    flows: jnp.ndarray,
    prevs: jnp.ndarray,
    currs: jnp.ndarray,
    cfg: dict | None = None,
    mask: jnp.ndarray | None = None,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Create weights for the flow estimates based on the specific method.

    This function computes weights according to the method specified
    in the configuration, which can be one of "list", "photometric",
    "photograd", "gradient", or "none".
    If "list" is specified, it uses the provided weights directly.
    If "photometric" is specified, it computes weights based on
    the photometric error of the flow estimates.
    If "photograd" is specified, it combines photometric error with
    gradient magnitude.
    If "gradient" is specified, it computes weights based on gradient magnitude.
    If "none" is specified, it returns uniform weights.

    Args:
        flows: Array of flow estimates from different agents.
            shape (B, N, H, W, 2) where B is the batch size,
            N is the number of agents.
        prevs: Previous frame images, shape (B, H, W).
        currs: Current frame images, shape (B, H, W).
        cfg: Configuration parameters for weight computation.
            It should contain the key "weights_type" to specify the method.
        mask: Mask to apply to the weights.
        epsilon: Small value to avoid division by zero in normalization.

    Returns:
        Weights for each flow estimate, shape (B, N, H, W).

    Raises:
        ValueError: If the configuration is invalid.
    """
    # Avoid mutable default argument
    if cfg is None:
        cfg = {}

    if "weights_type" not in cfg:
        logger.warning(
            "No weights_type specified in the configuration. "
            "Using 'none' as default."
        )
        weights_type = "none"
    else:
        weights_type = cfg["weights_type"]

    if flows.ndim == 4:
        # If flows is 4D, we assume it has shape (N, H, W, 2)
        flows = flows[jnp.newaxis, ...]

    # Compute weights based on the specified type
    if weights_type == "photometric":
        weights = _make_weights_photometric(flows, prevs, currs, cfg)
    elif weights_type == "photograd":
        weights = _make_weights_photograd(flows, prevs, currs, cfg)
    elif weights_type == "gradient":
        weights = _make_weights_gradient(flows, prevs, currs, cfg)
    elif weights_type == "list":
        weights = _make_weights_list(flows, cfg)
    elif weights_type == "none":
        weights = _make_weights_none(flows, cfg)
    else:
        raise ValueError(f"Unknown weights_type: {weights_type}")

    # Extract and validate normalization technique from the configuration
    if "normalization" not in cfg:
        logger.warning(
            "No normalization specified in the configuration. "
            "Using 'none' as default."
        )
        normalization = "none"
    else:
        normalization = cfg["normalization"]
        if normalization not in [
            "per_batch",
            "per_pixel",
            "none",
            "softmax_per_batch",
            "softmax_per_pixel",
            "max",
        ]:
            raise ValueError(
                f"Unknown normalization technique: {normalization}, "
                "expected one of 'per_batch', 'per_pixel', 'none', "
                "'softmax_per_batch', 'softmax_per_pixel' or 'max'"
            )

    # Apply the mask if provided
    if mask is not None:
        if mask.shape != weights.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match weights shape {weights.shape}."
            )
        weights = jnp.where(mask, weights, 0.0)

    # Normalize weights
    weights = _normalize_weights(weights, normalization, epsilon)

    # Apply mask again after normalization
    if mask is not None:
        if mask.shape != weights.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match weights shape {weights.shape}."
            )
        weights = jnp.where(mask, weights, 0.0)

    return weights
