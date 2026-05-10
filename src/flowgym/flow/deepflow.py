"""Module that implements DeepFlow for use in the Estimator framework."""

from typing import Any

import cv2
import jax.numpy as jnp
import numpy as np
from goggles.history.types import History

from flowgym.flow.base import FlowFieldEstimator


class DeepFlowEstimator(FlowFieldEstimator):
    """DeepFlow flow field estimator."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the DeepFlow estimator.

        Args:
            **kwargs: Additional arguments passed to FlowFieldEstimator.
        """
        self.est = cv2.optflow.createOptFlow_DeepFlow()  # pyright: ignore[reportAttributeAccessIssue]
        super().__init__(**kwargs)

    def _estimate(
        self, image: jnp.ndarray, state: History, _, __
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the flow field using DeepFlow.

        Args:
            image: The input image.
            state: The state object containing historical images.

        Returns:
            Tuple of computed flow field, additional outputs, and metrics.
        """
        # Note: Host callbacks, pure_functions calls, etc. are not as
        # efficient and not as well supported. The support depends on the
        # JAX version, GPUs, etc. After experimentation, a for loop is used.
        flows = np.zeros((*image.shape, 2), dtype=np.float32)
        for i in range(image.shape[0]):
            img1 = state["images"][i, -1, ...]
            img2 = image[i, ...]
            # Convert to numpy arrays for external library compatibility
            img1_np = np.asarray(img1)
            img2_np = np.asarray(img2)
            # Compute flow using the external library
            flows[i, ...] = self.est.calc(img1_np, img2_np, None)  # type: ignore
        return jnp.array(flows), {}, {}
