"""Cross-correlation flow estimator using a block-based approach."""

import jax.numpy as jnp

from flowgym.flow.base import FlowFieldEstimator


class DummyEstimator(FlowFieldEstimator):
    """Dummy flow field estimator that does not perform any estimation."""

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the Dummy estimator."""
        super().__init__(**kwargs)

    def _estimate(
        self, image: jnp.ndarray, _, __, ___
    ) -> tuple[jnp.ndarray, dict, dict]:
        """Dummy estimation function that returns zeros.

        Args:
            image: The input image.
            _: Unused parameter.
            __: Unused parameter.
            ___: Unused parameter.

        Returns:
            - A zero-filled flow field.
            - placeholder for additional outputs.
            - placeholder for metrics.
        """
        return jnp.zeros(image.shape + (2,)), {}, {}

    def create_train_step(self):
        """Dummy training step that does nothing."""

        def train_step(image, state, trainable_state, flow_gt=None):
            return state, trainable_state, 0.0

        return train_step
