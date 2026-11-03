"""Tests for the ground-truth-free flow quality metrics."""

import jax
import jax.numpy as jnp

from flowgym.flow.quality import correlation_peak_quality


def test_correlation_peak_quality_ranks_flows_by_error():
    """The true flow scores better than a slightly wrong one, then zero."""
    H = W = 64
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    cy = jax.random.uniform(k1, (200,)) * H
    cx = jax.random.uniform(k2, (200,)) * W
    d2 = (ys[..., None] - cy) ** 2 + (xs[..., None] - cx) ** 2
    image1 = jnp.exp(-d2 / 2.0).sum(-1)[None]
    image2 = jnp.roll(image1, 2, axis=-1)
    true_flow = jnp.zeros((1, H, W, 2)).at[..., 0].set(2.0)

    scores = [
        float(correlation_peak_quality(image1, image2, f)[0])
        for f in (true_flow, true_flow + 0.5, jnp.zeros_like(true_flow))
    ]

    assert scores[0] < scores[1] < scores[2]
