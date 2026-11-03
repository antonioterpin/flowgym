"""Module that implements OpenPIV for use in the Estimator framework."""

from __future__ import annotations

from typing import Any, ClassVar

import jax.numpy as jnp
from goggles import get_logger
from goggles.history.types import History
from jax import lax, vmap
from jax.scipy.signal import convolve as jax_convolve

from flowgym.common.filters import uniform_kernel
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.open_piv import process
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
        assert kernel.shape[-1] == field.shape[-1], (
            "Kernel channels must match field channels."
        )

    # Convolve every channel independently with its own 2D kernel. Passing the
    # full (H, W, C) field with a (k, k, C) kernel to a single ND convolution
    # would convolve across the channel axis as well, leaking values between
    # the u and v components; vmapping over the trailing axis keeps them
    # separate so each channel is a pure local mean of its own neighbours.
    def body_fun(_, field):
        average_neighbors = vmap(
            lambda channel, kernel_2d: jax_convolve(
                channel, kernel_2d, mode="same"
            ),
            in_axes=-1,
            out_axes=-1,
        )(field, kernel)
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
        assert field.ndim == 4, (
            "Field must be 4D (batch_size, height, width, channels)."
        )
        assert field.shape[:-1] == flags.shape, (
            "Field and flags must have the same spatial dimensions."
            + f" Got {field.shape} and {flags.shape}."
        )
        assert kernel_size > 0, "Kernel size must be positive."
        assert isinstance(n_iter, int), (
            "Number of iterations must be an integer."
        )
        assert n_iter > 0, "Number of iterations must be positive."
    kernel = uniform_kernel(kernel_size, n_channels=field.shape[-1])
    return vmap(lambda x, y: replace_invalid_single(x, y, kernel, n_iter))(
        field, flags
    )


def _as_static_pair(value: Any) -> int | tuple[int, int]:
    """Normalize int / [h, w] / (h, w) to an int or a static tuple.

    Args:
        value: A scalar size or a two-element height/width sequence.

    Returns:
        The value as an int, or a ``(height, width)`` tuple of ints.

    Raises:
        ValueError: If a sequence does not have exactly two elements, or a
            size is not integer-like (e.g. ``16.9``).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"Expected an int or a (height, width) pair, got {value!r}."
            )
        return (process._as_int(value[0]), process._as_int(value[1]))
    return process._as_int(value)


class OpenPIVJAXEstimator(FlowFieldEstimator):
    """OpenPIV flow field estimator implemented in JAX.

    Exposes the full openPIV-JAX feature set as estimator config: sub-pixel
    method, correlation method, rectangular windows, signal-to-noise outlier
    rejection, and iterative window-deformation multipass
    (:func:`process.multipass_deform`). :meth:`get_config` lists the
    parameters, so each configuration gets its own EPE cache namespace.

    ``sig2noise_threshold`` only takes effect when ``n_passes == 1``. With
    multipass the per-pass NaNs are zeroed while accumulating and no outlier
    mask is applied, so the threshold has no effect. Existing caches and
    selected libraries depend on this behavior, so it is kept as is.

    Attributes:
        velocity_filters: Available velocity filter types.
    """

    velocity_filters: ClassVar[list] = ["Manual", "Adaptive", "None"]

    def __init__(
        self,
        window_size: int | tuple[int, int] = 16,
        search_size: int | tuple[int, int] = 21,
        overlap: int | tuple[int, int] = 8,
        subpixel_method: process.SubpixelMethod = "gaussian",
        correlation_method: process.CorrelationMethod = "circular",
        n_passes: int = 1,
        sig2noise_threshold: float | None = None,
        openpiv_outlier_replacement_kernel_size: int = 3,
        openpiv_outlier_replacement_n_iter: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenPIVJAX estimator.

        Args:
            window_size: Interrogation window size (int or (height, width)).
            search_size: Search-area size (int or (height, width)).
            overlap: Window overlap (int or (height, width)).
            subpixel_method: Sub-pixel estimator: "gaussian", "parabolic"
                or "centroid".
            correlation_method: "circular" or "linear".
            n_passes: Window-deformation passes (1 = single pass).
            sig2noise_threshold: If set, vectors with peak-to-peak S/N below
                this are flagged as outliers (in addition to NaNs); if None,
                only NaN vectors are replaced.
            openpiv_outlier_replacement_kernel_size: Local-mean fill kernel.
            openpiv_outlier_replacement_n_iter: Local-mean fill iterations.
            **kwargs: Additional keyword arguments for the base class.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        self.window_size = _as_static_pair(window_size)
        self.search_size = _as_static_pair(search_size)
        self.overlap = _as_static_pair(overlap)
        window_hw = process._as_pair(self.window_size)
        overlap_hw = process._as_pair(self.overlap)
        if min(window_hw) <= 0:
            raise ValueError("window_size must be positive.")
        if min(process._as_pair(self.search_size)) <= 0:
            raise ValueError("search_size must be positive.")
        if min(overlap_hw) < 0:
            raise ValueError("overlap must be non-negative.")
        if any(o >= w for o, w in zip(overlap_hw, window_hw, strict=True)):
            raise ValueError("overlap must be less than window_size.")
        subpixel_method = process.parse_subpixel_method(subpixel_method)
        correlation_method = process.parse_correlation_method(
            correlation_method
        )
        if not isinstance(n_passes, int) or n_passes < 1:
            raise ValueError("n_passes must be a positive integer.")
        if (
            not isinstance(openpiv_outlier_replacement_kernel_size, int)
            or openpiv_outlier_replacement_kernel_size <= 0
        ):
            raise ValueError(
                "openpiv_outlier_replacement_kernel_size must be a "
                "positive integer."
            )
        if (
            not isinstance(openpiv_outlier_replacement_n_iter, int)
            or openpiv_outlier_replacement_n_iter <= 0
        ):
            raise ValueError(
                "openpiv_outlier_replacement_n_iter must be a positive integer."
            )
        self.subpixel_method: process.SubpixelMethod = subpixel_method
        self.correlation_method: process.CorrelationMethod = correlation_method
        self.n_passes = n_passes
        self.sig2noise_threshold = (
            None if sig2noise_threshold is None else float(sig2noise_threshold)
        )
        self.openpiv_outlier_replacement_kernel_size = (
            openpiv_outlier_replacement_kernel_size
        )
        self.openpiv_outlier_replacement_n_iter = (
            openpiv_outlier_replacement_n_iter
        )

        super().__init__(**kwargs)

    def _single_pass(
        self, img1: jnp.ndarray, img2: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run one correlation pass.

        Args:
            img1: First-frame images (batch, height, width).
            img2: Second-frame images (batch, height, width).

        Returns:
            The flow field and a boolean outlier mask.
        """
        if self.sig2noise_threshold is None:
            flow = process.extended_search_area_piv(
                img1,
                img2,
                window_size=self.window_size,
                search_area_size=self.search_size,
                overlap=self.overlap,
                subpixel_method=self.subpixel_method,
                correlation_method=self.correlation_method,
            )
            return flow, jnp.isnan(flow).any(axis=-1)
        flow, s2n = process.extended_search_area_piv(
            img1,
            img2,
            window_size=self.window_size,
            search_area_size=self.search_size,
            overlap=self.overlap,
            subpixel_method=self.subpixel_method,
            correlation_method=self.correlation_method,
            sig2noise_method="peak2peak",
        )
        mask = process.sig2noise_val(s2n, self.sig2noise_threshold)
        return flow, mask | jnp.isnan(flow).any(axis=-1)

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

        Returns:
            The computed flow field.
            Placeholder for additional outputs.
            Placeholder for metrics.
        """
        image1 = state["images"][:, -1, ...]

        if self.n_passes > 1:
            # Multipass output has no NaNs and is not masked, so
            # sig2noise_threshold has no effect here (see class docstring).
            flow_field = process.multipass_deform(
                image1,
                images,
                self.window_size,
                self.search_size,
                self.overlap,
                self.n_passes,
                subpixel_method=self.subpixel_method,
                correlation_method=self.correlation_method,
            )
            mask = jnp.zeros(flow_field.shape[:-1], dtype=bool)
        else:
            flow_field, mask = self._single_pass(image1, images)

        flow_field = replace_outliers(
            flow_field,
            mask,
            self.openpiv_outlier_replacement_n_iter,
            self.openpiv_outlier_replacement_kernel_size,
        )

        flow_field = process.upsample_flow(
            flow_field, (images.shape[1], images.shape[2])
        )

        return flow_field, {}, {}

    def get_config(self) -> dict[str, Any]:
        """Return the parameter dict that namespaces the cache.

        Returns:
            A JSON-serializable parameter dictionary.
        """
        return {
            "window_size": list(self.window_size)
            if isinstance(self.window_size, tuple)
            else self.window_size,
            "search_size": list(self.search_size)
            if isinstance(self.search_size, tuple)
            else self.search_size,
            "overlap": list(self.overlap)
            if isinstance(self.overlap, tuple)
            else self.overlap,
            "subpixel_method": self.subpixel_method,
            "correlation_method": self.correlation_method,
            "n_passes": self.n_passes,
            "sig2noise_threshold": self.sig2noise_threshold,
            "outlier_kernel_size": (
                self.openpiv_outlier_replacement_kernel_size
            ),
            "outlier_n_iter": self.openpiv_outlier_replacement_n_iter,
        }
