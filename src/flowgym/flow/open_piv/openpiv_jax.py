"""Module that implements DeepFlow for use in the Estimator framework."""

from typing import Any
from jax import lax, vmap
from jax.scipy.signal import convolve as jax_convolve
import jax.numpy as jnp
from goggles import get_logger
from goggles.history.types import History

from flowgym.flow.open_piv import process
from flowgym.flow.base import FlowFieldEstimator
from flowgym.common.filters import uniform_kernel
from flowgym.utils import DEBUG

logger = get_logger(__name__)


def replace_invalid_single(
    field: jnp.ndarray, flags: jnp.ndarray, kernel: jnp.ndarray, max_iter: int
) -> jnp.ndarray:
    """Replace invalid values in velocity field.

    Args:
        field: Single velocity field with invalid values.
        flags: Boolean mask indicating invalid values.
        kernel: Convolution kernel for local mean filtering.
        max_iter: Maximum number of iterations for replacement.

    Returns:
        Updated velocity field with invalid values replaced.
    """
    if DEBUG:
        assert isinstance(field, jnp.ndarray), "Field must be a jnp.ndarray."
        assert field.ndim == 3, f"Wrong -single- field shape {field.shape}."
        assert field.shape[:-1] == flags.shape, (
            "Field and flags must have the same spatial dimensions."
            + f" Got {field.shape} and {flags.shape}."
        )
        assert kernel.ndim == 3, "Kernel must be 3D (height, width, channels)."
        assert (
            kernel.shape[-1] == field.shape[-1]
        ), "Kernel channels must match field channels."

    def body_fun(_, field):
        average_neighbors = jax_convolve(field, kernel, mode="same")
        field = jnp.where(flags[..., None], average_neighbors, field)
        return field

    return lax.fori_loop(0, max_iter, body_fun, jnp.nan_to_num(field, nan=0.0))


def replace_outliers(
    field: jnp.ndarray, flags: jnp.ndarray, n_iter: int, kernel_size: int
) -> jnp.ndarray:
    """Replace invalid vectors in batched velocity fields.

    Args:
        field: Batched velocity fields with invalid vectors.
        flags: Boolean mask indicating invalid vectors.
        n_iter: Maximum number of iterations for replacement.
        kernel_size: Half-size of the kernel for local mean filtering.

    Returns:
        Updated velocity fields with invalid vectors replaced.
    """
    if DEBUG:
        assert isinstance(field, jnp.ndarray), "Field must be a jnp.ndarray."
        assert (
            field.ndim == 4
        ), "Field must be 4D (batch_size, height, width, channels)."
        assert field.shape[:-1] == flags.shape, (
            "Field and flags must have the same spatial dimensions."
            + f" Got {field.shape} and {flags.shape}."
        )
        assert kernel_size > 0, "Kernel size must be positive."
        assert isinstance(n_iter, int), "Number of iterations must be an integer."
        assert n_iter > 0, "Number of iterations must be positive."
    kernel = uniform_kernel(kernel_size, n_channels=field.shape[-1])
    return vmap(lambda x, y: replace_invalid_single(x, y, kernel, n_iter))(field, flags)


class OpenPIVJAXEstimator(FlowFieldEstimator):
    """OpenPIV flow field estimator implemented in JAX."""

    velocity_filters: list = ["Manual", "Adaptive", "None"]

    def __init__(
        self,
        window_size: int,
        search_size: int,
        overlap: int,
        openpiv_outlier_replacement_kernel_size: int,
        openpiv_outlier_replacement_n_iter: int,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenPIVJAX estimator.

        Args:
            window_size:
                The size of each interrogation window (square).
            search_size:
                The size of the search area for the flow field.
            overlap:
                Overlap between consecutive windows.
            openpiv_outlier_replacement_kernel_size:
                Kernel size for outlier replacement.
            openpiv_outlier_replacement_n_iter:
                Number of iterations for outlier replacement.
            **kwargs: Additional keyword arguments for the base class.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        # Check class-specific input parameters
        if (
            openpiv_outlier_replacement_kernel_size is None
            or openpiv_outlier_replacement_n_iter is None
        ):
            raise ValueError(
                "OpenPIVJAXEstimator requires outlier_replacement_kernel_size and "
                "outlier_replacement_n_iter to be set."
            )

        if (
            not isinstance(openpiv_outlier_replacement_kernel_size, int)
            or openpiv_outlier_replacement_kernel_size <= 0
        ):
            raise ValueError(
                "openpiv_outlier_replacement_kernel_size must be a positive integer."
            )
        if (
            not isinstance(openpiv_outlier_replacement_n_iter, int)
            or openpiv_outlier_replacement_n_iter <= 0
        ):
            raise ValueError(
                "openpiv_outlier_replacement_n_iter must be a positive integer."
            )
        self.openpiv_outlier_replacement_kernel_size = (
            openpiv_outlier_replacement_kernel_size
        )
        self.openpiv_outlier_replacement_n_iter = openpiv_outlier_replacement_n_iter

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size

        if not isinstance(search_size, int) or search_size <= 0:
            raise ValueError("search_size must be a positive integer.")
        self.search_size = search_size

        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if overlap >= window_size:
            raise ValueError("overlap must be less than window_size.")
        self.overlap = overlap

        super().__init__(**kwargs)

    def _estimate(
        self,
        images: jnp.ndarray,
        state: History,
        _: None,
        __: None,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field using OpenPIV.

        Args:
            images: The input images.
            state: Current state of the estimator.
            _: Unused parameter.
            __: Unused parameter.

        Returns:
            The computed flow field.
            Placeholder for additional outputs.
            Placeholder for metrics.
        """
        image1 = state["images"][:, -1, ...]

        if DEBUG:
            logger.debug(f"image1 shape: {image1.shape}, images shape: {images.shape}")

        flow_field = process.extended_search_area_piv(
            image1,
            images,
            window_size=self.window_size,
            overlap=self.overlap,
            search_area_size=self.search_size,
        )

        if DEBUG:
            logger.debug(f"Flow field shape: {flow_field.shape}")
            n_outliers = jnp.sum(jnp.isnan(flow_field).any(axis=-1))
            logger.debug(f"Number of outliers before post-processing: {n_outliers}")

        flow_field = replace_outliers(
            flow_field,
            jnp.isnan(flow_field).any(axis=-1),
            self.openpiv_outlier_replacement_n_iter,
            self.openpiv_outlier_replacement_kernel_size,
        )

        if DEBUG:
            logger.debug(f"Flow field shape after filling NaNs: {flow_field.shape}")
            n_outliers = jnp.sum(jnp.isnan(flow_field).any(axis=-1))
            logger.debug(f"Number of remaining outliers: {n_outliers}")

        flow_field = process.upsample_flow(
            flow_field, (images.shape[1], images.shape[2])
        )
        if DEBUG:
            logger.debug(f"Flow field shape after upsampling: {flow_field.shape}")

        return flow_field, {}, {}
