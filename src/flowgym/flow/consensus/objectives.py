"""Consensus algorithm for robust estimation of flow parameters."""

import jax
import jax.numpy as jnp
import jax.lax as lax
from flowgym.flow.consensus.regularizers import total_regularization_loss
from flowgym.common.filters import sobel
from flowgym.flow.dis.process import extract_patches, photometric_error_with_patches

from goggles import get_logger

logger = get_logger(__name__)


def z_objective(
    consensus_flow: jnp.ndarray,
    flows: jnp.ndarray,
    consensus_dual: jnp.ndarray,
    rho: float = 1.0,
    regularizer_list: list = [],
    regularizer_weights: dict[str, float] = {},
) -> jnp.ndarray:
    r"""Compute the Z objective function for consensus-based flow estimation.

    In the Boyd et al. notation, this corresponds to the function minimized over z
    in the second step of the ADMM algorithm. It includes the consensus term
    and regularization on the consensus flow estimate. It is expressed in the
    scaled form:

    .. math::
        reg(z) + \\frac{\\rho}{2} \\| x - z + u \\|^2

    where :math:`x` is the current flow estimate, :math:`u` is the dual variable,

    Args:
        flows (jnp.ndarray): Array of flow estimates from different agents.
        consensus_flow (jnp.ndarray): Current consensus flow estimate.
        consensus_dual (jnp.ndarray): Dual variable for consensus.
        rho (float): Penalty parameter for the consensus term.
        regularizer_list (list): List of regularization functions to apply.
        regularizer_weights (list): Weights for each regularization term.

    Returns:
        jnp.ndarray: Computed Z objective value.
    """
    # Compute the consensus term
    residuals = flows - consensus_flow[None, ...] + consensus_dual  # (N, H, W, 2)

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
        flows (jnp.ndarray): Array of flow estimates from different agents.
        consensus_flow (jnp.ndarray): Current consensus flow estimate.
        consensus_dual (jnp.ndarray): Dual variable for consensus.
        regularizer_list (list): List of regularization functions to apply.
        regularizer_weights (list): Weights for each regularization term.
        initial_flows (jnp.ndarray): Initial flow estimates for comparison.
        weights (jnp.ndarray, optional): Weights to apply to the anchor term.
        objective_type (str): Type of objective function to compute, either "l2" or "l1".
        rho (float): Penalty parameter for the consensus term.

    Returns:
        jnp.ndarray: Computed objective value for flow estimates.
    """
    # Calculate the anchor term
    if objective_type == "l2":
        anchor = flows - initial_flows
        data_term = anchor**2
        if weights is not None:
            # If weights are provided, apply them to the anchor term
            data_term *= weights[..., None]
        data_term = jnp.sum(data_term)
    elif objective_type == "l1":
        anchor = flows - initial_flows
        data_term = jnp.abs(anchor)
        if weights is not None:
            # If weights are provided, apply them to the anchor term
            data_term *= weights[..., None]
        data_term = jnp.sum(data_term)

    # Residuals for the consensus term
    residuals = flows - consensus_flow[None, ...] + consensus_dual

    # Consensus term
    consensus_term = 0.5 * rho * jnp.sum(residuals**2)

    return data_term + consensus_term


def weights_and_anchors(
    anchor_flows: jnp.ndarray,
    weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wrapper function to return weights and anchors.

    Args:
        anchor_flows (jnp.ndarray): Array of anchor flow estimates.
        weights (jnp.ndarray): Array of weights for the flow estimates.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the weights and anchors.
    """
    return weights, anchor_flows


def make_weights(
    flows: jnp.ndarray,
    prevs: jnp.ndarray,
    currs: jnp.ndarray,
    cfg: dict = {},
    mask: jnp.ndarray | None = None,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Create weights for the flow estimates based on the specific method.

    This function computes weights according to the method specified in the configuration,
    which can be one of "list", "photometric", or "none".
    If "list" is specified, it uses the provided weights directly.
    If "photometric" is specified, it computes weights based on the mean squared error of
    the flow estimates.
    If "none" is specified, it returns uniform weights.

    Args:
        flows (jnp.ndarray): Array of flow estimates from different agents.
            shape (B, N, H, W, 2) where B is the batch size, N is the number of agents.
        prevs (jnp.ndarray): Previous frame images, shape (B, H, W).
        currs (jnp.ndarray): Current frame images, shape (B, H, W).
        cfg (dict): Configuration parameters for weight computation.
            It should contain the key "weights_type" to specify the method.
        mask (jnp.ndarray, optional): Mask to apply to the weights.
        epsilon (float): Small value to avoid division by zero in normalization.

    Returns:
        jnp.ndarray: Weights for each flow estimate, shape (B, N, H, W).
    """
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
        flows = flows[None, ...]

    if weights_type == "photometric":
        B, N, H, W, _ = flows.shape

        # Extract patch size and stride from the configuration
        if "patch_size" not in cfg:
            logger.warning(
                "No patch_size specified in the configuration. " "Using 3 as default."
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
                "No patch_stride specified in the configuration. " "Using 1 as default."
            )
            patch_stride = 1
        else:
            patch_stride = cfg["patch_stride"]
            if not isinstance(patch_stride, int) or patch_stride <= 0:
                raise ValueError("patch_stride must be a positive integer.")

        # Calculate half of the patch size for padding
        half = patch_size // 2

        # Tile prevs and currs to match the number of flow estimates
        prevs = jnp.tile(prevs[:, None, ...], (1, N, 1, 1))
        currs = jnp.tile(currs[:, None, ...], (1, N, 1, 1))

        # Reshape to use vmap
        prevs = prevs.reshape(-1, H, W)  # shape (B*N, H, W)
        currs = currs.reshape(-1, H, W)  # shape (B*N, H, W)
        flows = flows.reshape(-1, H, W, 2)  # shape (B*N, H, W, 2)

        # Compute the photometric error
        photometric_errors = jax.vmap(
            photometric_error_with_patches, in_axes=(0, 0, 0, None, None)
        )(
            prevs, currs, flows, patch_size, patch_stride
        )  # shape (B*N, H - half, W - half)

        # Compute weights based on the inverse of the photometric errors
        weights = 1.0 / jnp.maximum(photometric_errors, 1)  # Avoid division by zero

        # Pad weights to match the original shape
        weights = jnp.pad(
            weights,
            (
                (0, 0),
                (half, half),
                (half, half),
            ),
            mode="constant",
            constant_values=0,
        )

        weights = weights.reshape(B, N, H, W)
        if "weights" in cfg:
            logger.warning(
                "Weights specified in the configuration will be ignored "
                "when using 'photometric' as weights_type."
            )
    elif weights_type == "photograd":
        B, N, H, W, _ = flows.shape

        # Extract patch size and stride from the configuration
        if "patch_size" not in cfg:
            logger.warning(
                "No patch_size specified in the configuration. " "Using 3 as default."
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
                "No patch_stride specified in the configuration. " "Using 1 as default."
            )
            patch_stride = 1
        else:
            patch_stride = cfg["patch_stride"]
            if not isinstance(patch_stride, int) or patch_stride <= 0:
                raise ValueError("patch_stride must be a positive integer.")

        # Calculate half of the patch size for padding
        half = patch_size // 2

        # Tile prevs and currs to match the number of flow estimates
        prevs = jnp.tile(prevs[:, None, ...], (1, N, 1, 1))
        currs = jnp.tile(currs[:, None, ...], (1, N, 1, 1))

        # Reshape to use vmap
        prevs = prevs.reshape(-1, H, W)  # shape (B*N, H, W)
        currs = currs.reshape(-1, H, W)  # shape (B*N, H, W)
        flows = flows.reshape(-1, H, W, 2)  # shape (B*N, H, W, 2)

        # Compute the photometric error
        photometric_errors = jax.vmap(
            photometric_error_with_patches, in_axes=(0, 0, 0, None, None, None)
        )(
            prevs, currs, flows, patch_size, patch_stride, True
        )  # shape (B*N, H - half, W - half)

        # Normalize the photometric errors by number of pixels in a patch
        num_pixels = patch_size * patch_size
        photometric_errors /= num_pixels

        # Compute gradients using Sobel filter
        kx, ky = sobel()
        Ix = lax.conv_general_dilated(
            prevs[:, None, ...], kx[None, None, ...], (1, 1), padding="VALID"
        )[:, 0]
        Iy = lax.conv_general_dilated(
            prevs[:, None, ...], ky[None, None, ...], (1, 1), padding="VALID"
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
        )  # Avoid division by zero

        weights = jnp.pad(
            weights,
            (
                (0, 0),
                (half, half),
                (half, half),
            ),
            mode="constant",
            constant_values=0,
        )

        weights = weights.reshape(B, N, H, W)
        if "weights" in cfg:
            logger.warning(
                "Weights specified in the configuration will be ignored "
                "when using 'photometric' as weights_type."
            )

    elif weights_type == "list":
        # Use the provided weights directly
        if "weights" not in cfg:
            raise ValueError("Weights must be provided when weights_type is 'list'.")
        weights = jnp.array(cfg["weights"])
        if weights.ndim == 1:
            # If weights are 1D, we assume they are for each flow estimate
            weights = weights[None, :, None, None]  # Shape (1, N, 1, 1)
            weights = jnp.broadcast_to(weights, flows.shape[:-1])  # (B, N, H, W)
        else:
            pass
    elif weights_type == "none" or weights_type is None:
        # If no weights are specified, use uniform weights
        weights = jnp.ones(flows.shape[:-1])
        if "weights" in cfg:
            logger.warning(
                "Weights specified in the configuration will be ignored "
                "when using 'none' as weights_type."
            )
    else:
        raise ValueError(f"Unknown weights_type: {weights_type}")

    # Extract normalization technique from the configuration
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
                " expected one of 'per_batch', 'per_pixel', 'none', "
                "'softmax_per_batch', 'softmax_per_pixel' or 'max'"
            )

    # Apply the mask if provided
    if mask is not None:
        if mask.shape != weights.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match weights shape {weights.shape}."
            )
        weights = jnp.where(mask, weights, 0.0)

    if normalization == "per_batch":
        weights = weights / (
            jnp.sum(weights, axis=(1, 2, 3))[:, None, None, None] + epsilon
        )

    elif normalization == "per_pixel":
        weights = weights / (jnp.sum(weights, axis=(1), keepdims=True) + epsilon)

    elif normalization == "softmax_per_batch":
        # Flatten (N, H, W) into one axis for each batch
        B, N, H, W = weights.shape
        flat_weights = weights.reshape(B, -1)  # Shape (B, N*H*W)
        shifted = flat_weights - jnp.max(flat_weights, axis=1, keepdims=True)
        exp_weights = jnp.exp(shifted)
        softmax_flat = exp_weights / jnp.sum(exp_weights, axis=1, keepdims=True)
        # Reshape back to (B, N, H, W)
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

    # apply mask again
    if mask is not None:
        if mask.shape != weights.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match weights shape {weights.shape}."
            )
        weights = jnp.where(mask, weights, 0.0)

    return weights
