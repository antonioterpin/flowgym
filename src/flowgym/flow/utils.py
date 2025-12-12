"""Utils for analyzing and visualizing flow fields."""

import jax.numpy as jnp
from jax import lax


def compute_gradients(field):
    """Compute discrete-difference gradients of a scalar field.

    Args:
        field (jnp.ndarray): Input field of shape (B, H, W).

    Returns:
        df_dx (jnp.ndarray): Gradient in x-direction of shape (B, H, W).
        df_dy (jnp.ndarray): Gradient in y-direction of shape (B, H, W).
    """
    # wrap around with periodic boundary conditions
    # df_dx = (jnp.roll(field, -1, axis=2) - jnp.roll(field, 1, axis=2)) / (2 * dx)
    # df_dy = (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2 * dy)

    # ignore boundary pixels
    df_dx = (field[:, :, 2:] - field[:, :, :-2]) / 2
    df_dy = (field[:, 2:, :] - field[:, :-2, :]) / 2
    return df_dx[:, 1:-1, :], df_dy[:, :, 1:-1]


def compute_divergence_and_vorticity(flow):
    """Computes divergence and vorticity of a 2D flow field.

    Args:
        flow (jnp.ndarray): Flow field of shape (B, H, W, 2).
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.

    Returns:
        divergence (jnp.ndarray): Divergence of the flow field of shape (B, H, W).
        vorticity (jnp.ndarray): Vorticity of the flow field of shape (B, H, W).
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
        Ix (jnp.ndarray): x gradient of the image
        Iy (jnp.ndarray): y gradient of the image
        patch_size (int): size of the patches to extract
        patch_stride (int): stride between patches

    Returns:
        H (jnp.ndarray): Hessian matrix of shape (num_patches, 2, 2)
    """
    win = (patch_size, patch_size)
    st = (patch_stride, patch_stride)

    def wsum(x):
        """Helper that computes the sum of all values within a sliding window."""
        return lax.reduce_window(
            x, 0.0, lax.add, win, st, "VALID"
        )  # img is already padded

    Sxx = wsum(Ix * Ix)
    Syy = wsum(Iy * Iy)
    Sxy = wsum(Ix * Iy)

    H = jnp.stack([jnp.stack([Sxx, Sxy], -1), jnp.stack([Sxy, Syy], -1)], -2)

    return H


def inv_hessian(
    Ix: jnp.ndarray, Iy: jnp.ndarray, patch_size: int, patch_stride: int, eps: float
) -> jnp.ndarray:
    """Compute the inverse of the Hessian matrix for each patch.

    Args:
        Ix (jnp.ndarray): x gradient of the image
        Iy (jnp.ndarray): y gradient of the image
        patch_size (int): size of the patches to extract
        patch_stride (int): stride between patches
        eps (float): small value to avoid division by zero in the hessian inversion

    Returns:
        invH (jnp.ndarray): inverted Hessian matrix of shape (num_patches, 2, 2)
    """
    win = (patch_size, patch_size)
    st = (patch_stride, patch_stride)

    def wsum(x):
        """Helper that computes the sum of all values within a sliding window."""
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


def compute_vector_gradients(flow: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute centered discrete gradients for multi-channel fields.

    Args:
        flow (jnp.ndarray): Flow field of shape (B, H, W) or (B, H, W, C)

    Returns:
        df_dx (jnp.ndarray): Gradient in x-direction of shape (B, H-2, W-2, C)
        df_dy (jnp.ndarray): Gradient in y-direction of shape (B, H-2, W-2, C)
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


def compute_divergence(flow) -> jnp.ndarray:
    """Compute divergence of a batch of flow fields with shape (B, H, W, 2).

    Args:
        flow (jnp.ndarray): Flow field of shape (B, H, W, 2).

    Returns:
        jnp.ndarray: Divergence of the flow field of shape (B, H-2, W-2).
    """
    # Compute gradients using centered differences
    dfx, dfy = compute_vector_gradients(flow)

    du_x_dx = dfx[..., 0]  # Gradient in x-direction for u
    du_y_dy = dfy[..., 1]  # Gradient in y-direction for v

    return du_x_dx + du_y_dy  # Shape: (B, H-2, W-2)


# TODO: consider using a more efficient implementation,
# i.e. using jax.lax.conv_general_dilated
def compute_laplacian(flow):
    """Compute the Laplacian of a (B, H, W, 2) flow field.

    Args:
        flow (jnp.ndarray): Flow field of shape (B, H, W, 2).

    Returns:
        jnp.ndarray: Laplacian of the flow field of shape (B, H-2, W-2, 2).
    """
    # Pad for boundary conditions (Neumann/reflective)
    lap = (
        flow[:, 2:, 1:-1, :]  # down
        + flow[:, :-2, 1:-1, :]  # up
        + flow[:, 1:-1, 2:, :]  # right
        + flow[:, 1:-1, :-2, :]  # left
        - 4 * flow[:, 1:-1, 1:-1, :]  # center
    )
    return lap
