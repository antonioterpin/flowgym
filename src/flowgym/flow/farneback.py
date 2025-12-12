"""Module that implements Farneback for use in the Estimator framework."""

from typing import Any
import cv2
import numpy as np
import jax.numpy as jnp

from flowgym.flow.base import FlowFieldEstimator
from goggles.history.types import History


class FarnebackEstimator(FlowFieldEstimator):
    """Farneback flow field estimator."""

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        **kwargs: Any,
    ):
        """Initialize the Farneback estimator."""
        if pyr_scale < 0.0 or pyr_scale > 1.0:
            raise ValueError(f"pyr_scale {pyr_scale} must be in [0, 1].")
        self.pyr_scale = pyr_scale

        if levels < 0 or not isinstance(levels, int):
            raise TypeError(f"levels {levels} must be a positive integer.")
        self.levels = levels

        if winsize < 0 or not isinstance(winsize, int):
            raise ValueError(f"winsize {winsize} must be a positive integer.")
        self.winsize = winsize

        if iterations < 0 or not isinstance(iterations, int):
            raise ValueError(f"iterations {iterations} must be a positive integer.")
        self.iterations = iterations

        if poly_n < 0 or not isinstance(poly_n, int):
            raise ValueError(f"poly_n {poly_n} must be a positive integer.")
        self.poly_n = poly_n

        if poly_sigma < 0.0:
            raise ValueError(f"poly_sigma {poly_sigma} must be >= 0.")
        self.poly_sigma = poly_sigma

        super().__init__(**kwargs)

    def _estimate(
        self, image: jnp.ndarray, state: History, _, __
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field using Farneback.

        Args:
            image: Input image.
            state: Current state of the estimator.
            _: Unused parameter.
            __: Unused parameter.

        Returns:
            - Computed flow field.
            - placeholder for additional outputs.
            - placeholder for metrics.
        """
        # Note: Host callbacks, pure_functions calls, etc. seem
        # to be not particularly more efficient and also they are not as supported.
        # The support depends on the version of JAX, on the GPUs, etc.
        # After experimenting with them, I think it is ok to stick to a for loop...
        flows = np.zeros(image.shape + (2,), dtype=np.float32)
        for i in range(image.shape[0]):
            img1 = state["images"][i, -1, ...]
            img2 = image[i, ...]
            # Convert to numpy array
            img1 = np.asarray(img1)
            img2 = np.asarray(img2)
            flows[i, ...] = cv2.calcOpticalFlowFarneback(  # type: ignore
                prev=img1,
                next=img2,
                flow=None,  # type: ignore
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=self.iterations,
                poly_n=self.poly_n,
                poly_sigma=self.poly_sigma,
                flags=0,
            )

        return jnp.asarray(flows), {}, {}
