"""Regression coverage for consensus children with independent image grids."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import flowgym
from flowgym.flow.dummy import DummyEstimator
from flowgym.make import make_estimator


class HistoryEstimator(DummyEstimator):
    """Use the previous image and flow so malformed child histories fail."""

    def _estimate(self, image, state, trainable_state, extras):
        previous = state["images"][:, -1]
        assert previous.shape == image.shape, "Child image grid must match"
        delta = image - previous
        initial = jnp.mean(state["estimates"], axis=(1, 2, 3, 4))
        flow = jnp.stack((delta, delta), axis=-1) + initial[:, None, None, None]
        return flow, {}, {}


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("supports_jit", [False, True])
def test_consensus_initializes_children_on_their_own_grid(
    tmp_path, monkeypatch, jit, temporal, supports_jit
):
    """Parent output resizing must not corrupt differently sized children."""
    monkeypatch.setattr(
        HistoryEstimator, "supports_jit", lambda self: supports_jit
    )
    monkeypatch.setitem(
        flowgym.ALL_ESTIMATORS, "history_test", HistoryEstimator
    )
    children = []
    for size in (4, 8):
        children.append(
            {
                "estimator": "history_test",
                "config": {
                    "jit": jit,
                    "preprocess": [
                        {
                            "name": "resize_image",
                            "target_height": size,
                            "target_width": size,
                        }
                    ],
                    "postprocess": [
                        {
                            "name": "resize_flow",
                            "target_height": 8,
                            "target_width": 8,
                        }
                    ],
                },
            }
        )
    child_path = tmp_path / "children.yaml"
    child_path.write_text(yaml.safe_dump({"estimators": children}))
    config = {
        "estimator": "consensus",
        "config": {
            "jit": jit,
            "consensus_algorithm": "mean",
            "estimators_list_path": str(child_path),
            "use_temporal_propagation": temporal,
            "postprocess": [
                {
                    "name": "resize_flow",
                    "target_height": 4,
                    "target_width": 4,
                }
            ],
        },
    }
    trained, create, compute, _ = make_estimator(
        config, image_shape=(2, 8, 8), estimate_shape=(2, 4, 4, 2)
    )
    assert create is not None, "Factory must supply state initialization"
    images = jnp.ones((2, 8, 8))
    state = create(images, jax.random.PRNGKey(0))
    state, _ = compute(images * 2, state, trained)
    assert state["estimates"].shape == (2, 1, 4, 4, 2), (
        "Parent grid must survive"
    )
    # resize_flow interpolates the grid without changing flow magnitudes.
    np.testing.assert_allclose(state["estimates"], 1.0, atol=1e-6)
    state, _ = compute(images * 3, state, trained)
    np.testing.assert_allclose(
        state["estimates"], 2.0 if temporal else 1.0, atol=1e-6
    )
