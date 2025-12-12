"""Filters for velocity field processing."""

import jax.numpy as jnp
from jax import vmap
from jax.scipy.signal import convolve as jax_convolve
from flowgym.utils import DEBUG


def gaussian_kernel(sigma: float, truncate: float, n_channels: int) -> jnp.ndarray:
    """Generate a 2D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian kernel.
        truncate: Truncate parameter for the kernel.
        n_channels: Number of channels in the input data.

    Returns:
        Normalized 2D Gaussian kernel.
    """
    if DEBUG:
        assert isinstance(sigma, float), "Sigma must be a float."
        assert sigma > 0, "Sigma must be positive."
        assert isinstance(truncate, float), "Truncate must be a float."
        assert truncate > 0, "Truncate must be positive."
        assert isinstance(n_channels, int), "Number of channels must be an integer."
        assert n_channels > 0, "Number of channels must be positive."

    radius = int(truncate * sigma + 0.5)
    x = jnp.arange(-radius, radius + 1)
    y = jnp.arange(-radius, radius + 1)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = jnp.repeat(kernel[:, :, jnp.newaxis], n_channels, axis=2)

    if DEBUG:
        assert kernel.shape == (
            2 * radius + 1,
            2 * radius + 1,
            n_channels,
        ), "Kernel shape mismatch."
        assert jnp.isclose(
            kernel.sum(axis=(0, 1)), jnp.ones((n_channels,))
        ).all(), "Kernel normalization error."

    return kernel


def gaussian_smoothing(
    data: jnp.ndarray, sigma: float, truncate: float = 4.0, mode: str = "same"
) -> jnp.ndarray:
    """Smooth the channel (e.g., u of the velocity field) using a Gaussian kernel.

    Args:
        data: batched array to be smoothed along the channels.
        sigma: Standard deviation of the Gaussian kernel.
        truncate: Truncate parameter for the kernel.
        mode: Convolution mode, either 'same', 'valid', or 'reflect'.

    Returns:
        Smoothed 2D array.
    """
    kernel = gaussian_kernel(sigma, truncate, n_channels=data.shape[-1])
    if DEBUG:
        assert isinstance(data, jnp.ndarray), "Data must be a jnp.ndarray."
        assert data.ndim == 4, "Data must be 4D (batch_size, height, width, channels)."
        assert kernel.ndim == 3, "Kernel must be 3D (height, width, channels)."
        assert (
            kernel.shape[-1] == data.shape[-1]
        ), "Kernel channels must match data channels."
        assert (
            data.shape[1] > kernel.shape[0]
        ), "Data height must be greater than kernel height."
        assert (
            data.shape[2] > kernel.shape[1]
        ), "Data width must be greater than kernel width."
    if mode == "reflect":
        mode = "valid"
        radius = kernel.shape[0] // 2
        data = jnp.pad(
            data, ((0, 0), (radius, radius), (radius, radius), (0, 0)), mode="reflect"
        )
    return vmap(lambda x: jax_convolve(x, kernel, mode=mode))(data)


def uniform_kernel(kernel_size: int, n_channels: int) -> jnp.ndarray:
    """Generate a uniform kernel for local mean filtering, excluding the center.

    Args:
        kernel_size: Half-size of the kernel (kernel size is 2*kernel_size + 1).
        n_channels: Number of channels in the input data.

    Returns:
        Normalized uniform kernel.
    """
    if DEBUG:
        assert isinstance(kernel_size, int), "Kernel size must be an integer."
        assert kernel_size > 0, "Kernel size must be positive."
        assert isinstance(n_channels, int), "Number of channels must be an integer."
        assert n_channels > 0, "Number of channels must be positive."
    size = 2 * kernel_size + 1
    kernel = jnp.ones((size, size, n_channels)) / (size**2 - 1)
    kernel = kernel.at[kernel_size, kernel_size].set(0)
    if DEBUG:
        assert kernel.shape == (size, size, n_channels), "Kernel shape mismatch."
        assert jnp.isclose(
            kernel.sum(axis=(0, 1)), jnp.ones((n_channels,))
        ).all(), "Kernel normalization error."
    return kernel


def sobel() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Sobel kernels for x and y directions."""
    kx = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], jnp.float32) * 0.125
    return (
        kx,
        kx.T,
    )
