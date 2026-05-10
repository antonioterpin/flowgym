"""Simple density estimator using a threshold discrimination."""

from typing import Any

import jax.numpy as jnp
from goggles.history.types import History

from flowgym.common.base import (
    Estimator,
    EstimatorTrainableState,
)


class SimpleDensityEstimator(Estimator):
    """Simple density estimator using a threshold discrimination."""

    def __init__(
        self,
        threshold: float,
        **kwargs: Any,
    ) -> None:
        """Initialize the estimator with the bandwidth.

        Args:
            threshold: Threshold for considering a pixel as occupied.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If threshold is not in the range [0, 255].
        """
        if not (0 <= threshold <= 255):
            raise ValueError("threshold must be a float in the range [0, 255].")
        self.threshold = threshold

        super().__init__(**kwargs)

    def _estimate(
        self,
        img: jnp.ndarray,
        _: History,
        __: EstimatorTrainableState,
        ___: dict,
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Compute the density from the image.

        Args:
            img: Input image.

        Returns:
            - Computed density.
            - Empty dict for cache outputs.
            - Empty dict for extras.
        """
        return (
            (
                jnp.sum(img > self.threshold, axis=(1, 2))
                / (img.shape[1] * img.shape[2])
            )[:, None],
            {},
            {},
        )
