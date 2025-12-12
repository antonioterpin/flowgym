"""Module that implements DeepFlow for use in the Estimator framework."""

import cv2
import numpy as np
import jax.numpy as jnp

from flowgym.flow.base import FlowFieldEstimator
from goggles.history.types import History


class DeepFlowEstimator(FlowFieldEstimator):
    """DeepFlow flow field estimator."""

    def __init__(self, **kwargs):
        """Initialize the DeepFlow estimator."""
        self.est = cv2.optflow.createOptFlow_DeepFlow()  # type: ignore
        super().__init__(**kwargs)

    def _estimate(
        self, image: jnp.ndarray, state: History, _, __
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field using DeepFlow.

        Args:
            image: The input image.
            state: The state object containing historical images.
            _: Unused parameter.
            __: Unused parameter.

        Returns:
            - The computed flow field.
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
            # Convert to numpy arrays for external library compatibility
            img1_np = np.asarray(img1)
            img2_np = np.asarray(img2)
            # Compute flow using the external library
            flows[i, ...] = self.est.calc(img1_np, img2_np, None)  # type: ignore
        return jnp.array(flows), {}, {}
