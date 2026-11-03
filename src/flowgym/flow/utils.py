"""Utils for analyzing and visualizing flow fields."""

import jax.numpy as jnp
from jax import lax


def compute_gradients(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute discrete-difference gradients of a scalar field.

    Args:
        field: Input field of shape (B, H, W).

    Returns:
        Tuple of gradients in x and y directions, each of shape (B, H-2, W-2).
    """
    # Ignore boundary pixels; use centered finite differences
    df_dx = (field[:, :, 2:] - field[:, :, :-2]) / 2
    df_dy = (field[:, 2:, :] - field[:, :-2, :]) / 2
    return df_dx[:, 1:-1, :], df_dy[:, :, 1:-1]


def compute_divergence_and_vorticity(
    flow: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes divergence and vorticity of a 2D flow field.

    Args:
        flow: Flow field of shape (B, H, W, 2).

    Returns:
        Tuple of divergence and vorticity fields, each shape (B, H-2, W-2).
    """
    u = flow[..., 0]
    v = flow[..., 1]

    du_dx, du_dy = compute_gradients(u)
    dv_dx, dv_dy = compute_gradients(v)

    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy

    return divergence, vorticity


def hessian(
    Ix: jnp.ndarray, Iy: jnp.ndarray, patch_size: int, patch_stride: int
) -> jnp.ndarray:
    """Compute the Hessian matrix for each patch.

    Args:
        Ix: x gradient of the image.
        Iy: y gradient of the image.
        patch_size: size of the patches to extract.
        patch_stride: stride between patches.

    Returns:
        Hessian matrix of shape (num_patches, 2, 2).
    """
    win = (patch_size, patch_size)
    st = (patch_stride, patch_stride)

    def wsum(x: jnp.ndarray) -> jnp.ndarray:
        """Sum all values within a sliding window.

        Args:
            x: Input array.

        Returns:
            Summed array from reduce_window operation.
        """
        return lax.reduce_window(x, 0.0, lax.add, win, st, "VALID")

    Sxx = wsum(Ix * Ix)
    Syy = wsum(Iy * Iy)
    Sxy = wsum(Ix * Iy)

    H = jnp.stack([jnp.stack([Sxx, Sxy], -1), jnp.stack([Sxy, Syy], -1)], -2)

    return H


def inv_hessian(
    Ix: jnp.ndarray,
    Iy: jnp.ndarray,
    patch_size: int,
    patch_stride: int,
    eps: float,
) -> jnp.ndarray:
    """Compute the inverse of the Hessian matrix for each patch.

    Args:
        Ix: x gradient of the image.
        Iy: y gradient of the image.
        patch_size: size of the patches to extract.
        patch_stride: stride between patches.
        eps: small value to avoid division by zero in inversion.

    Returns:
        Inverted Hessian matrix of shape (num_patches, 2, 2).
    """
    win = (patch_size, patch_size)
    st = (patch_stride, patch_stride)

    def wsum(x: jnp.ndarray) -> jnp.ndarray:
        """Sum all values within a sliding window.

        Args:
            x: Input array.

        Returns:
            Summed array from reduce_window operation.
        """
        return lax.reduce_window(x, 0.0, lax.add, win, st, "VALID")

    # Construct the Hessian matrices and invert them
    Sxx = wsum(Ix * Ix)
    Syy = wsum(Iy * Iy)
    Sxy = wsum(Ix * Iy)
    det = Sxx * Syy - Sxy**2 + eps

    invH = (
        jnp.stack([jnp.stack([Syy, -Sxy], -1), jnp.stack([-Sxy, Sxx], -1)], -2)
        / det[..., None, None]  # (N,2,2)
    )
    invH = invH.reshape(-1, 2, 2)  # (N,2,2)

    return invH


def compute_vector_gradients(
    flow: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute centered discrete gradients for multi-channel fields.

    Args:
        flow: Flow field of shape (B, H, W) or (B, H, W, C).

    Returns:
        Tuple of gradients in x and y directions, each shape (B, H-2, W-2, C).
    """
    if flow.ndim == 3:
        # (B, H, W) -> (B, H, W, 1)
        flow = flow[..., jnp.newaxis]

    # centered differences
    df_dx = (flow[:, :, 2:, :] - flow[:, :, :-2, :]) / 2  # x-direction (W)
    df_dy = (flow[:, 2:, :, :] - flow[:, :-2, :, :]) / 2  # y-direction (H)

    # crop to (B, H-2, W-2, C)
    df_dx = df_dx[:, 1:-1, :, :]
    df_dy = df_dy[:, :, 1:-1, :]

    return df_dx, df_dy


def compute_divergence(flow: jnp.ndarray) -> jnp.ndarray:
    """Compute divergence of a batch of flow fields.

    Args:
        flow: Flow field of shape (B, H, W, 2).

    Returns:
        Divergence of the flow field of shape (B, H-2, W-2).
    """
    dfx, dfy = compute_vector_gradients(flow)
    return dfx[..., 0] + dfy[..., 1]


def compute_laplacian(flow: jnp.ndarray) -> jnp.ndarray:
    """Compute the Laplacian of a flow field.

    Args:
        flow: Flow field of shape (B, H, W, 2).

    Returns:
        Laplacian of the flow field of shape (B, H-2, W-2, 2).
    """
    # Neumann boundary conditions via finite differences
    lap = (
        flow[:, 2:, 1:-1, :]  # down
        + flow[:, :-2, 1:-1, :]  # up
        + flow[:, 1:-1, 2:, :]  # right
        + flow[:, 1:-1, :-2, :]  # left
        - 4 * flow[:, 1:-1, 1:-1, :]  # center
    )
    return lap
