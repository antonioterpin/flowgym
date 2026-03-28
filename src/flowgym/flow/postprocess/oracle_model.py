"""Shared model architecture for learned oracle outlier rejection."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class OracleMaskCNN(nn.Module):
    """Small CNN producing per-pixel mask logits from a flow field."""

    features: tuple[int, ...] = (16, 32)

    @nn.compact
    def __call__(self, flow_field: jnp.ndarray) -> jnp.ndarray:
        x = flow_field
        for feature in self.features:
            x = nn.Conv(feature, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
        logits = nn.Conv(1, kernel_size=(1, 1), padding="SAME")(x)
        return jnp.squeeze(logits, axis=-1)
