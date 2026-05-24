"""Tests for RAFT estimator model module."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from flowgym.nn import raft_model


def test_coords_grid_shape_values_and_dtype():
    """Coordinate grid uses (x, y) order and tiles across batch."""
    model = raft_model.RaftEstimatorModel(
        hidden_dim=4,
        context_dim=4,
        corr_levels=2,
        corr_radius=1,
        iters=2,
        norm_fn="none",
    )

    coords = model._coords_grid(batch=2, ht=2, wd=3)

    expected_single = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    assert coords.shape == (2, 2, 3, 2)
    assert coords.dtype == jnp.float32
    assert jnp.array_equal(coords[0], expected_single)
    assert jnp.array_equal(coords[1], expected_single)


def test_raft_estimator_forward_path_is_fully_exercised(monkeypatch):
    """Run RaftEstimatorModel.__call__ with lightweight mocked blocks."""
    captured_encoder_inputs: list[object] = []
    captured_corr_args: dict[str, object] = {}
    captured_scan_args: dict[str, object] = {}

    class DummyEncoderBlock(nn.Module):
        output_dim: int
        norm_fn: str
        dropout: float
        train: bool

        @nn.compact
        def __call__(self, x):
            del self.norm_fn, self.dropout, self.train
            captured_encoder_inputs.append(x)
            if isinstance(x, list):
                return [
                    jnp.full((*x[0].shape[:-1], self.output_dim), 3.0),
                    jnp.full((*x[1].shape[:-1], self.output_dim), 4.0),
                ]
            return jnp.full((*x.shape[:-1], self.output_dim), 5.0)

    def fake_build_corr_pyramid(
        fmap1: jnp.ndarray, fmap2: jnp.ndarray, corr_levels: int
    ) -> list[jnp.ndarray]:
        captured_corr_args["fmap1_shape"] = fmap1.shape
        captured_corr_args["fmap2_shape"] = fmap2.shape
        captured_corr_args["corr_levels"] = corr_levels
        return [
            jnp.zeros((*fmap1.shape[:3], 2), dtype=fmap1.dtype)
        ] * corr_levels

    def fake_scan(scan_body, variable_broadcast, split_rngs, length, out_axes):
        captured_scan_args["scan_body"] = scan_body
        captured_scan_args["variable_broadcast"] = variable_broadcast
        captured_scan_args["split_rngs"] = split_rngs
        captured_scan_args["length"] = length
        captured_scan_args["out_axes"] = out_axes

        class DummyScanModule(nn.Module):
            update_block: nn.Module
            coords0: jnp.ndarray
            corr_radius: int
            inp: jnp.ndarray
            corr_pyramid: list[jnp.ndarray]

            @nn.compact
            def __call__(self, carry):
                net, coords1 = carry
                del net, self.update_block, self.corr_radius, self.inp
                del self.corr_pyramid
                flow = coords1 - self.coords0
                flows = jnp.stack([flow] * length, axis=0)
                return carry, flows

        return DummyScanModule

    monkeypatch.setattr(raft_model, "EncoderBlock", DummyEncoderBlock)
    monkeypatch.setattr(
        raft_model, "build_corr_pyramid", fake_build_corr_pyramid
    )
    monkeypatch.setattr(raft_model.nn, "scan", fake_scan)

    model = raft_model.RaftEstimatorModel(
        hidden_dim=5,
        context_dim=7,
        corr_levels=3,
        corr_radius=2,
        iters=4,
        norm_fn="none",
        dropout=0.1,
        train=True,
    )
    images = jnp.arange(2 * 3 * 4 * 2, dtype=jnp.float32).reshape(2, 3, 4, 2)
    flow_init = jnp.full((2, 3, 4, 2), 2.5, dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), images, flow_init)
    flows = model.apply(variables, images, flow_init)  # type: ignore[misc]

    normalized = images / 256.0
    assert flows.shape == (4, 2, 3, 4, 2)
    assert jnp.allclose(flows, flow_init[None, ...])
    assert isinstance(captured_encoder_inputs[-2], list)
    assert jnp.array_equal(captured_encoder_inputs[-2][0], normalized[..., :1])
    assert jnp.array_equal(captured_encoder_inputs[-2][1], normalized[..., 1:])
    assert jnp.array_equal(captured_encoder_inputs[-1], normalized[..., :1])
    assert captured_corr_args["corr_levels"] == 3
    assert captured_corr_args["fmap1_shape"] == (2, 3, 4, 256)
    assert captured_corr_args["fmap2_shape"] == (2, 3, 4, 256)
    assert captured_scan_args["variable_broadcast"] == "params"
    assert captured_scan_args["split_rngs"] == {"params": False}
    assert captured_scan_args["length"] == 4
    assert captured_scan_args["out_axes"] == 0
