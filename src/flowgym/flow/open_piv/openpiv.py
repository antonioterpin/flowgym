"""Module that implements OpenPIV for use in the Estimator framework."""

from typing import Any
import numpy as np
import jax.numpy as jnp
from openpiv import pyprocess, validation, filters

from goggles.history.types import History
from flowgym.flow.base import FlowFieldEstimator
from flowgym.flow.open_piv.process import upsample_flow, get_field_shape


class OpenPIVEstimator(FlowFieldEstimator):
    """DeepFlow flow field estimator."""

    def __init__(
        self,
        window_size: int = 32,
        search_size: int = 64,
        overlap: int = 16,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenPIV estimator.

        Args:
            window_size: The size of each interrogation window (square).
            search_size: The size of the search area for the flow field.
            overlap: Overlap between consecutive windows.
            kwargs: Additional keyword arguments for the base class.
        """

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
        self, images: jnp.ndarray, state: History, _: None, __: None
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field using OpenPIV.

        Args:
            images: The input images.
            state: The state object containing historical images.
            _: Unused parameter.
            __: Unused parameter.

        Returns:
            - The computed flow field.
            - Placeholder for additional outputs.
            - Placeholder for metrics.
        """
        # Note: Host callbacks, pure_functions calls, etc. seem
        # to be not particularly more efficient and also they are not as supported.
        # The support depends on the version of JAX, on the GPUs, etc.
        # After experimenting with them, I think it is ok to stick to a for loop...
        num_rows, num_cols = get_field_shape(
            (images.shape[1], images.shape[2]),
            (self.search_size, self.search_size),
            (self.overlap, self.overlap),
        )
        flows = np.zeros((images.shape[0], num_rows, num_cols, 2), dtype=np.float32)
        for i in range(images.shape[0]):
            img1 = state["images"][i, -1, ...]
            img2 = images[i, ...]
            # Convert to numpy array
            img1_np = np.asarray(img1)
            img2_np = np.asarray(img2)

            u0, v0, sig2noise = pyprocess.extended_search_area_piv(
                img1_np,
                img2_np,
                window_size=self.window_size,
                overlap=self.overlap,
                dt=1,
                search_area_size=self.search_size,
                sig2noise_method="peak2peak",
            )

            # Post-process the flow field
            invalid_mask = validation.sig2noise_val(
                sig2noise,
                threshold=1.3,
            )

            u0, v0 = filters.replace_outliers(
                u0,
                v0,
                invalid_mask,
                method="localmean",
                max_iter=30,
                kernel_size=3,
            )

            flows[i, ..., 0] = u0.filled()
            flows[i, ..., 1] = v0.filled()

        flows = jnp.asarray(flows)
        return (
            upsample_flow(
                flows,
                (images.shape[1], images.shape[2]),
            ),
            {},
            {},
        )
