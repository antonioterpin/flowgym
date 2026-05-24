"""Tests for neural network block builders.

This module tests the config-driven builders for:
- ConvBlock
- ResidualBlock
- Block dispatcher
"""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from flax.core import freeze, unfreeze

from flowgym.nn.blocks import (
    BLOCK_REGISTRY,
    ClampedInstanceNorm,
    ConvBlock,
    EncoderBlock,
    FlowHeadBlock,
    MotionEncoderBlock,
    ResidualBlock,
    ScanBodyBlock,
    SepConvGRUBlock,
    UpdateBlock,
    UpsampleBlock,
    build_block_from_config,
    build_conv_block_from_config,
    build_residual_block_from_config,
)


class TestConvBlockBuilder:
    """Tests for ConvBlock builder."""

    def test_minimal_config(self):
        """Build ConvBlock with only required fields."""
        config = {"features": 64}
        block = build_conv_block_from_config(config)
        assert isinstance(block, ConvBlock)
        assert block.features == 64
        assert block.kernel_size == (3, 3)
        assert block.strides == (1, 1)
        assert block.norm_fn == "none"
        assert block.activation is not None  # Default relu

    def test_features_required(self):
        """ConvBlock builder requires 'features' field."""
        with pytest.raises(ValueError, match="must contain 'features'"):
            build_conv_block_from_config({})

    def test_features_must_be_positive(self):
        """ConvBlock features must be positive."""
        with pytest.raises(ValueError, match="features must be positive"):
            build_conv_block_from_config({"features": 0})
        with pytest.raises(ValueError, match="features must be positive"):
            build_conv_block_from_config({"features": -10})

    def test_kernel_size_as_int(self):
        """Build ConvBlock with kernel_size as int."""
        config = {"features": 32, "kernel_size": 5}
        block = build_conv_block_from_config(config)
        assert block.kernel_size == (5, 5)

    def test_kernel_size_as_list(self):
        """Build ConvBlock with kernel_size as [H, W]."""
        config = {"features": 32, "kernel_size": [7, 3]}
        block = build_conv_block_from_config(config)
        assert block.kernel_size == (7, 3)

    def test_kernel_size_invalid_list(self):
        """Invalid kernel_size list raises error."""
        with pytest.raises(ValueError, match="kernel_size must be int or"):
            build_conv_block_from_config(
                {"features": 32, "kernel_size": [1, 2, 3]}
            )

    def test_strides_as_int(self):
        """Build ConvBlock with strides as int."""
        config = {"features": 32, "strides": 2}
        block = build_conv_block_from_config(config)
        assert block.strides == (2, 2)

    def test_strides_as_list(self):
        """Build ConvBlock with strides as [H, W]."""
        config = {"features": 32, "strides": [2, 1]}
        block = build_conv_block_from_config(config)
        assert block.strides == (2, 1)

    def test_strides_invalid_list(self):
        """Invalid strides list raises error."""
        with pytest.raises(ValueError, match="strides must be int or"):
            build_conv_block_from_config({"features": 32, "strides": [1, 2, 3]})

    def test_norm_fn_from_registry(self):
        """Build ConvBlock with various normalization types."""
        for norm_type in ["batch", "instance", "group", "none"]:
            config = {"features": 32, "norm_fn": norm_type}
            block = build_conv_block_from_config(config)
            assert block.norm_fn == norm_type

    def test_invalid_norm_fn(self):
        """Invalid norm_fn raises error."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            build_conv_block_from_config({"features": 32, "norm_fn": "layer"})

    def test_activation_from_registry(self):
        """Build ConvBlock with various activation functions."""
        for act_name in ["relu", "gelu", "tanh", "sigmoid"]:
            config = {"features": 32, "activation": act_name}
            block = build_conv_block_from_config(config)
            assert block.activation is not None

    def test_activation_none(self):
        """Build ConvBlock with no activation."""
        config = {"features": 32, "activation": "none"}
        block = build_conv_block_from_config(config)
        assert block.activation is None

    def test_invalid_activation(self):
        """Invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            build_conv_block_from_config(
                {"features": 32, "activation": "invalid"}
            )

    def test_group_size(self):
        """Build ConvBlock with custom group_size."""
        config = {"features": 64, "norm_fn": "group", "group_size": 16}
        block = build_conv_block_from_config(config)
        assert block.group_size == 16

    def test_group_size_must_be_positive_for_group_norm(self):
        """Non-positive group_size is rejected for group normalization."""
        with pytest.raises(
            ValueError,
            match="group_size must be positive when norm_fn='group'",
        ):
            build_conv_block_from_config(
                {"features": 64, "norm_fn": "group", "group_size": 0}
            )

    def test_full_config(self):
        """Build ConvBlock with all fields specified."""
        config = {
            "features": 128,
            "kernel_size": [5, 5],
            "strides": [2, 2],
            "norm_fn": "batch",
            "activation": "gelu",
            "group_size": 32,
        }
        block = build_conv_block_from_config(config)
        assert block.features == 128
        assert block.kernel_size == (5, 5)
        assert block.strides == (2, 2)
        assert block.norm_fn == "batch"
        assert block.activation is not None
        assert block.group_size == 32

    def test_block_is_callable(self):
        """Built ConvBlock can be called."""
        config = {"features": 16}
        block = build_conv_block_from_config(config)

        # Initialize and apply
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 3))
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]

        assert y.shape == (1, 8, 8, 16)  # type: ignore[union-attr]
        assert y.dtype == x.dtype  # type: ignore[union-attr]
        assert isinstance(block, ConvBlock)
        assert block.features == 16


class TestResidualBlockBuilder:
    """Tests for ResidualBlock builder."""

    def test_minimal_config(self):
        """Build ResidualBlock with only required fields."""
        config = {"features": 64}
        block = build_residual_block_from_config(config)
        assert isinstance(block, ResidualBlock)
        assert block.features == 64
        assert block.kernel_size == (3, 3)
        assert block.strides == (1, 1)
        assert block.norm_fn == "none"

    def test_features_required(self):
        """ResidualBlock builder requires 'features' field."""
        with pytest.raises(ValueError, match="must contain 'features'"):
            build_residual_block_from_config({})

    def test_features_must_be_positive(self):
        """ResidualBlock features must be positive."""
        with pytest.raises(ValueError, match="features must be positive"):
            build_residual_block_from_config({"features": 0})

    def test_kernel_size_as_int(self):
        """Build ResidualBlock with kernel_size as int."""
        config = {"features": 32, "kernel_size": 5}
        block = build_residual_block_from_config(config)
        assert block.kernel_size == (5, 5)

    def test_kernel_size_as_list(self):
        """Build ResidualBlock with kernel_size as [H, W]."""
        config = {"features": 32, "kernel_size": [7, 3]}
        block = build_residual_block_from_config(config)
        assert block.kernel_size == (7, 3)

    def test_strides_as_int(self):
        """Build ResidualBlock with strides as int."""
        config = {"features": 32, "strides": 2}
        block = build_residual_block_from_config(config)
        assert block.strides == (2, 2)

    def test_strides_as_list(self):
        """Build ResidualBlock with strides as [H, W]."""
        config = {"features": 32, "strides": [2, 1]}
        block = build_residual_block_from_config(config)
        assert block.strides == (2, 1)

    def test_kernel_size_invalid_list(self):
        """Invalid kernel_size list raises error."""
        with pytest.raises(ValueError, match="kernel_size must be int or"):
            build_residual_block_from_config(
                {"features": 32, "kernel_size": [1, 2, 3]}
            )

    def test_strides_invalid_list(self):
        """Invalid strides list raises error."""
        with pytest.raises(ValueError, match="strides must be int or"):
            build_residual_block_from_config(
                {"features": 32, "strides": [1, 2, 3]}
            )

    def test_norm_fn_from_registry(self):
        """Build ResidualBlock with various normalization types."""
        for norm_type in ["batch", "instance", "group", "none"]:
            config = {"features": 32, "norm_fn": norm_type}
            block = build_residual_block_from_config(config)
            assert block.norm_fn == norm_type

    def test_invalid_norm_fn(self):
        """Invalid norm_fn raises error."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            build_residual_block_from_config(
                {"features": 32, "norm_fn": "invalid"}
            )

    def test_full_config(self):
        """Build ResidualBlock with all fields specified."""
        config = {
            "features": 128,
            "kernel_size": [3, 3],
            "strides": [1, 1],
            "norm_fn": "instance",
        }
        block = build_residual_block_from_config(config)
        assert block.features == 128
        assert block.kernel_size == (3, 3)
        assert block.strides == (1, 1)
        assert block.norm_fn == "instance"

    def test_block_is_callable(self):
        """Built ResidualBlock can be called."""
        config = {"features": 16}
        block = build_residual_block_from_config(config)

        # Initialize and apply
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 8, 8, 3))
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]

        assert y.shape == (1, 8, 8, 16)  # type: ignore[union-attr]
        assert y.dtype == x.dtype  # type: ignore[union-attr]
        assert isinstance(block, ResidualBlock)
        assert block.features == 16


class TestBlockDispatcher:
    """Tests for block dispatcher."""

    def test_dispatch_conv_block(self):
        """Dispatcher routes to ConvBlock builder."""
        config = {"type": "conv", "features": 32}
        block = build_block_from_config(config)
        assert isinstance(block, ConvBlock)
        assert block.features == 32

    def test_dispatch_residual_block(self):
        """Dispatcher routes to ResidualBlock builder."""
        config = {"type": "residual", "features": 64}
        block = build_block_from_config(config)
        assert isinstance(block, ResidualBlock)
        assert block.features == 64

    def test_type_required(self):
        """Dispatcher requires 'type' field."""
        with pytest.raises(ValueError, match="must contain 'type'"):
            build_block_from_config({"features": 64})

    def test_unknown_type(self):
        """Unknown block type raises helpful error."""
        with pytest.raises(ValueError, match=r"Unknown block type.*invalid"):
            build_block_from_config({"type": "invalid", "features": 64})

    def test_error_shows_available_types(self):
        """Error message lists available block types."""
        with pytest.raises(ValueError, match=r"Available:.*conv"):
            build_block_from_config({"type": "unknown", "features": 64})

    def test_type_case_insensitive(self):
        """Block type lookup is case-insensitive."""
        for type_name in ["conv", "CONV", "Conv", "CoNv"]:
            config = {"type": type_name, "features": 32}
            block = build_block_from_config(config)
            assert isinstance(block, ConvBlock)

    def test_registry_complete(self):
        """BLOCK_REGISTRY contains expected entries."""
        assert "conv" in BLOCK_REGISTRY
        assert "residual" in BLOCK_REGISTRY
        assert len(BLOCK_REGISTRY) == 2


class TestBlockIntegration:
    """Integration tests for block builders."""

    def test_sequential_blocks(self):
        """Build and chain multiple blocks."""
        configs = [
            {"type": "conv", "features": 32, "activation": "relu"},
            {"type": "conv", "features": 64, "activation": "gelu"},
            {"type": "residual", "features": 64, "norm_fn": "batch"},
        ]

        blocks = [build_block_from_config(cfg) for cfg in configs]
        assert len(blocks) == 3
        assert isinstance(blocks[0], ConvBlock)
        assert isinstance(blocks[1], ConvBlock)
        assert isinstance(blocks[2], ResidualBlock)

    def test_conv_block_forward_pass(self):
        """ConvBlock processes input correctly."""
        config = {
            "features": 16,
            "kernel_size": 3,
            "strides": 1,
            "activation": "relu",
            "norm_fn": "none",
        }
        block = build_conv_block_from_config(config)

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 8, 8, 3))
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]

        assert y.shape == (2, 8, 8, 16)  # type: ignore[union-attr]
        # ReLU should ensure non-negative outputs (mostly)
        assert jnp.sum(y >= 0) > 0.9 * y.size  # type: ignore[union-attr]

    def test_residual_block_forward_pass(self):
        """ResidualBlock processes input correctly."""
        config = {
            "features": 32,
            "kernel_size": 3,
            "strides": 1,
            "norm_fn": "none",
        }
        block = build_residual_block_from_config(config)

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 8, 8, 16))
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]

        assert y.shape == (2, 8, 8, 32)  # type: ignore[union-attr]
        # Residual block applies ReLU at the end
        assert jnp.sum(y >= 0) > 0.9 * y.size  # type: ignore[union-attr]

    def test_different_activations_produce_different_outputs(self):
        """Different activations produce different results."""
        key = jax.random.PRNGKey(0)
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]).reshape(1, 1, 5, 1)

        configs = [
            {"features": 1, "activation": "relu", "kernel_size": 1},
            {"features": 1, "activation": "tanh", "kernel_size": 1},
        ]

        outputs = []
        for cfg in configs:
            block = build_conv_block_from_config(cfg)
            variables = block.init(key, x)
            y = block.apply(variables, x)  # type: ignore[misc]
            outputs.append(y)

        # Different activations should produce different outputs
        assert not jnp.allclose(outputs[0], outputs[1])

    def test_different_norms_produce_different_outputs(self):
        """Different normalization types produce different results."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 8, 8, 3))

        configs = [
            {"features": 16, "norm_fn": "none"},
            {
                "features": 16,
                "norm_fn": "instance",
            },  # instance norm doesn't need mutable state
        ]

        outputs = []
        for cfg in configs:
            block = build_conv_block_from_config(cfg)
            variables = block.init(key, x)
            y = block.apply(variables, x)  # type:ignore[misc]
            outputs.append(y)

        # Different norms should produce different outputs
        assert not jnp.allclose(outputs[0], outputs[1])

    def test_conv_block_group_norm_path(self):
        """ConvBlock group normalization branch runs and preserves shape."""
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (2, 8, 8, 8))
        block = ConvBlock(features=8, norm_fn="group", group_size=2)
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]
        assert y.shape == (2, 8, 8, 8)  # type: ignore[union-attr]

    def test_conv_block_batch_norm_path(self):
        """ConvBlock batch normalization branch updates batch stats."""
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (2, 8, 8, 4))
        block = ConvBlock(features=4, norm_fn="batch")
        variables = block.init(key, x)
        y, mutated = block.apply(  # type: ignore[misc]
            variables, x, mutable=["batch_stats"]
        )
        assert y.shape == (2, 8, 8, 4)
        assert "batch_stats" in mutated

    def test_conv_block_invalid_norm_raises(self):
        """Direct ConvBlock call rejects unsupported normalization."""
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (1, 4, 4, 2))
        block = ConvBlock(features=4, norm_fn="invalid_norm")
        with pytest.raises(ValueError, match="Unsupported normalization"):
            block.init(key, x)

    def test_conv_block_no_activation_branch(self):
        """ConvBlock supports activation=None branch."""
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (1, 4, 4, 3))
        block = ConvBlock(features=5, activation=None)
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]
        assert y.shape == (1, 4, 4, 5)  # type: ignore[union-attr]

    def test_residual_block_float32_cast_back_path(self):
        """ResidualBlock float32 input goes through cast-back branch."""
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (1, 8, 8, 4), dtype=jnp.float32)
        block = ResidualBlock(features=4, norm_fn="none")
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]
        assert y.dtype == jnp.float32  # type: ignore[union-attr]

    def test_residual_block_float16_skips_cast_back(self):
        """ResidualBlock float16 input keeps dtype without cast-back branch."""
        key = jax.random.PRNGKey(51)
        x = jax.random.normal(key, (1, 8, 8, 4), dtype=jnp.float16)
        block = ResidualBlock(features=4, norm_fn="none")
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]
        assert y.dtype == jnp.float16  # type: ignore[union-attr]

    def test_clamped_instance_norm_with_bias_path(self):
        """ClampedInstanceNorm executes bias masking branch."""
        key = jax.random.PRNGKey(6)
        x = jnp.ones((1, 4, 4, 2), dtype=jnp.float32)
        norm = ClampedInstanceNorm(use_bias=True, var_threshold=1e6)
        variables = norm.init(key, x)
        y = norm.apply(variables, x)  # type: ignore[misc]
        assert y.shape == x.shape

    def test_clamped_instance_norm_with_top_level_bias(self):
        """ClampedInstanceNorm supports optional top-level bias lookup path."""
        key = jax.random.PRNGKey(61)
        x = jnp.ones((1, 4, 4, 2), dtype=jnp.float32)
        norm = ClampedInstanceNorm(use_bias=True, var_threshold=1e6)
        variables = norm.init(key, x)

        mutable = unfreeze(variables)
        inst_bias = mutable["params"]["InstanceNorm_0"]["bias"]
        mutable["params"]["bias"] = jnp.array(inst_bias)
        variables_with_top_bias = freeze(mutable)

        y = norm.apply(variables_with_top_bias, x)  # type: ignore[misc]
        assert y.shape == x.shape

    def test_upsample_block_with_batch_norm(self):
        """UpsampleBlock runs use_bn=True path and doubles spatial dims."""
        key = jax.random.PRNGKey(7)
        x = jax.random.normal(key, (1, 4, 5, 3))
        block = UpsampleBlock(features=6, scale=2, use_bn=True)
        variables = block.init(key, x)
        y, mutated = block.apply(  # type: ignore[misc]
            variables, x, mutable=["batch_stats"]
        )
        assert y.shape == (1, 8, 10, 6)
        assert "batch_stats" in mutated

    def test_upsample_block_without_activation(self):
        """UpsampleBlock handles activation=None path."""
        key = jax.random.PRNGKey(8)
        x = jax.random.normal(key, (1, 3, 3, 2))
        block = UpsampleBlock(features=4, activation=None)
        variables = block.init(key, x)
        y = block.apply(variables, x)  # type: ignore[misc]
        assert y.shape == (1, 6, 6, 4)  # type: ignore[union-attr]

    def test_encoder_block_tensor_and_list_paths(self):
        """EncoderBlock supports tensor input and list/tuple split output."""
        key = jax.random.PRNGKey(9)
        x = jax.random.normal(key, (1, 16, 16, 3))

        # Tensor path
        enc_tensor = EncoderBlock(output_dim=8, dropout=0.0, train=False)
        vars_tensor = enc_tensor.init(key, x)
        y_tensor = enc_tensor.apply(vars_tensor, x)  # type: ignore[misc]
        assert y_tensor.shape == (1, 16, 16, 8)  # type: ignore[union-attr]

        # List/tuple path with train=True (dropout RNG required)
        enc_list = EncoderBlock(output_dim=8, dropout=0.2, train=True)
        x_list = [x, x]
        vars_list = enc_list.init({"params": key, "dropout": key}, x_list)
        y_list = enc_list.apply(  # type: ignore[misc]
            vars_list, x_list, rngs={"dropout": key}
        )
        assert isinstance(y_list, list)
        assert len(y_list) == 2
        assert y_list[0].shape == (1, 16, 16, 8)
        assert y_list[1].shape == (1, 16, 16, 8)

    def test_flow_head_motion_encoder_gru_and_update_blocks(self):
        """Core recurrent flow blocks run end-to-end with expected shapes."""
        key = jax.random.PRNGKey(10)
        net = jax.random.normal(key, (1, 8, 8, 96))
        inp = jax.random.normal(key, (1, 8, 8, 128))
        corr = jax.random.normal(key, (1, 8, 8, 64))
        flow = jax.random.normal(key, (1, 8, 8, 2))

        head = FlowHeadBlock(hidden_dim=64)
        h_vars = head.init(key, net)
        h_out = head.apply(h_vars, net)  # type: ignore[misc]
        assert h_out.shape == (1, 8, 8, 2)

        motion = MotionEncoderBlock()
        m_vars = motion.init(key, flow, corr)
        m_out = motion.apply(m_vars, flow, corr)  # type: ignore[misc]
        assert m_out.shape == (1, 8, 8, 128)

        gru = SepConvGRUBlock(hidden_dim=96)
        g_vars = gru.init(key, net, inp)
        g_out = gru.apply(g_vars, net, inp)  # type: ignore[misc]
        assert g_out.shape == (1, 8, 8, 96)

        update = UpdateBlock(hidden_dim=96, corr_levels=2, corr_radius=2)
        u_vars = update.init(key, net, inp, corr, flow)
        net2, mask, dflow = update.apply(  # type: ignore[misc]
            u_vars, net, inp, corr, flow
        )
        assert net2.shape == net.shape
        assert mask.shape == (1, 8, 8, 64 * 9)
        assert dflow.shape == flow.shape

    def test_scan_body_block_runs_with_mocked_correlation(self, monkeypatch):
        """ScanBodyBlock executes update flow path with mocked correlation."""

        class DummyUpdateBlock(nn.Module):
            @nn.compact
            def __call__(self, net, inp, corr, flow):
                del inp, corr
                mask = jnp.zeros((*net.shape[:3], 64 * 9), dtype=net.dtype)
                delta_flow = jnp.ones_like(flow)
                return net, mask, delta_flow

        def fake_correlation_block(corr_pyramid, coords1, corr_radius):
            del corr_pyramid, corr_radius
            return jnp.zeros((*coords1.shape[:3], 4), dtype=coords1.dtype)

        monkeypatch.setattr(
            "flowgym.flow.raft.process.correlation_block",
            fake_correlation_block,
        )

        key = jax.random.PRNGKey(11)
        net = jax.random.normal(key, (1, 4, 4, 8))
        coords0 = jax.random.normal(key, (1, 4, 4, 2))
        coords1 = jax.random.normal(key, (1, 4, 4, 2))
        inp = jax.random.normal(key, (1, 4, 4, 8))
        corr_pyramid = [jnp.zeros((1, 4, 4, 4))]

        block = ScanBodyBlock(
            update_block=DummyUpdateBlock(),
            coords0=coords0,
            corr_radius=2,
            inp=inp,
            corr_pyramid=corr_pyramid,
        )
        variables = block.init(key, (net, coords1))
        (net_out, coords_out), flow_out = block.apply(  # type: ignore[misc]
            variables, (net, coords1)
        )

        assert net_out.shape == net.shape
        assert coords_out.shape == coords1.shape
        assert flow_out.shape == coords1.shape
