"""Tests for how RAFT blends patch estimates into the full flow field."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from flowgym.flow.raft.raft256_jax import RaftJax256Estimator


def test_single_patch_flow_is_not_blended():
    """With one patch per image the model output is returned unchanged."""
    estimator = RaftJax256Estimator(train=False)
    flow = jax.random.normal(jax.random.PRNGKey(0), (2, 256, 256, 2))
    # The model returns per-iteration predictions; the last is used.
    estimator.model = SimpleNamespace(apply=lambda *_: flow[None])
    images = jnp.zeros((2, 256, 256))

    out, _, _ = estimator._estimate(
        images,
        {"images": images[:, None]},
        SimpleNamespace(params=None),
        {},
    )

    assert bool(jnp.array_equal(out, flow))
