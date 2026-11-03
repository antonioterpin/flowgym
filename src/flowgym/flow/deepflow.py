"""Module that implements DeepFlow for use in the Estimator framework."""

from collections.abc import Callable
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
        self.est = self._create_deepflow()
        super().__init__(**kwargs)

    @staticmethod
    def _create_deepflow() -> Any:
        """Create a DeepFlow estimator across OpenCV API layouts.

        Returns:
            OpenCV DeepFlow estimator instance.

        Raises:
            ImportError: If DeepFlow cannot be created from the
                installed OpenCV.
        """
        factories: list[tuple[str, Callable[[], Any] | None]] = [
            (
                "cv2.optflow.createOptFlow_DeepFlow",
                getattr(
                    getattr(cv2, "optflow", None),
                    "createOptFlow_DeepFlow",
                    None,
                ),
            ),
            (
                "cv2.createOptFlow_DeepFlow",
                getattr(cv2, "createOptFlow_DeepFlow", None),
            ),
            (
                "cv2.legacy.createOptFlow_DeepFlow",
                getattr(
                    getattr(cv2, "legacy", None),
                    "createOptFlow_DeepFlow",
                    None,
                ),
            ),
        ]
        for _, factory in factories:
            if callable(factory):
                return factory()

        available = ", ".join(path for path, _ in factories)
        raise ImportError(
            "DeepFlow requires OpenCV contrib modules. None of these factories "
            "were found: "
            f"{available}. Install 'opencv-contrib-python-headless' "
            "and ensure 'opencv-python-headless' is not shadowing it."
        )

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
