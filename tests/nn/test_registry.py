"""Tests for neural network registries.

This module tests the registry lookups and primitive functions for:
- Activation functions
- Normalization types
- Pooling operations
"""

import jax.numpy as jnp
import pytest

from flowgym.nn.registry import (
    ACTIVATION_REGISTRY,
    NORM_REGISTRY,
    POOLING_REGISTRY,
    get_activation,
    get_norm,
    get_pooling,
)


class TestActivationRegistry:
    """Tests for activation function registry."""

    def test_all_activations_retrievable(self):
        """All registered activations can be retrieved."""
        for name in ACTIVATION_REGISTRY:
            fn = get_activation(name)
            assert callable(fn)

    def test_activation_case_insensitive(self):
        """Activation lookup is case-insensitive."""
        fn_lower = get_activation("relu")
        fn_upper = get_activation("RELU")
        fn_mixed = get_activation("ReLu")
        # All should return the same function
        assert fn_lower is fn_upper is fn_mixed

    def test_activation_unknown_raises(self):
        """Unknown activation raises ValueError with helpful message."""
        with pytest.raises(
            ValueError, match=r"Unknown activation.*invalid_act"
        ):
            get_activation("invalid_act")

    def test_activation_error_shows_available(self):
        """Error message lists available activations."""
        with pytest.raises(ValueError, match=r"Available options:.*relu"):
            get_activation("nonexistent")

    @pytest.mark.parametrize(
        "name",
        ["relu", "gelu", "silu", "swish", "tanh", "sigmoid", "elu", "softplus"],
    )
    def test_activation_functions_work(self, name):
        """Activation functions produce expected output shapes."""
        fn = get_activation(name)
        x = jnp.array([-1.0, 0.0, 1.0])
        y = fn(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_relu_clips_negatives(self):
        """ReLU activation zeros negative values."""
        relu = get_activation("relu")
        x = jnp.array([-1.0, 0.0, 1.0])
        y = relu(x)
        assert jnp.allclose(y, jnp.array([0.0, 0.0, 1.0]))

    def test_tanh_range(self):
        """Tanh activation is bounded in [-1, 1]."""
        tanh = get_activation("tanh")
        x = jnp.linspace(-10, 10, 100)
        y = tanh(x)
        assert jnp.all(y >= -1.0)
        assert jnp.all(y <= 1.0)

    def test_sigmoid_range(self):
        """Sigmoid activation is bounded in [0, 1]."""
        sigmoid = get_activation("sigmoid")
        x = jnp.linspace(-10, 10, 100)
        y = sigmoid(x)
        assert jnp.all(y >= 0.0)
        assert jnp.all(y <= 1.0)

    def test_leaky_relu_preserves_negatives(self):
        """Leaky ReLU allows small negative values through."""
        leaky_relu = get_activation("leaky_relu")
        x = jnp.array([-1.0, 0.0, 1.0])
        y = leaky_relu(x)
        # Negative value should be scaled, not zeroed
        assert y[0] < 0.0
        assert y[0] > -1.0
        assert jnp.allclose(y[1], 0.0)
        assert jnp.allclose(y[2], 1.0)

    def test_none_activation_is_identity(self):
        """'none' activation is identity function."""
        identity = get_activation("none")
        x = jnp.array([-5.0, 0.0, 5.0])
        y = identity(x)
        assert jnp.allclose(x, y)

    def test_swish_is_alias_for_silu(self):
        """'swish' is an alias for 'silu'."""
        swish = get_activation("swish")
        silu = get_activation("silu")
        # Both should reference the same underlying function
        assert swish is silu


class TestNormRegistry:
    """Tests for normalization type registry."""

    def test_all_norms_retrievable(self):
        """All registered normalization types can be retrieved."""
        for name in NORM_REGISTRY:
            norm_type = get_norm(name)
            assert isinstance(norm_type, str)

    def test_norm_case_insensitive(self):
        """Normalization lookup is case-insensitive."""
        norm_lower = get_norm("batch")
        norm_upper = get_norm("BATCH")
        norm_mixed = get_norm("BaTcH")
        assert norm_lower == norm_upper == norm_mixed == "batch"

    def test_norm_unknown_raises(self):
        """Unknown normalization raises ValueError with helpful message."""
        with pytest.raises(
            ValueError, match=r"Unknown normalization.*invalid_norm"
        ):
            get_norm("invalid_norm")

    def test_norm_error_shows_available(self):
        """Error message lists available normalization types."""
        with pytest.raises(ValueError, match=r"Available options:.*batch"):
            get_norm("layer")  # PyTorch has layer norm, we don't

    @pytest.mark.parametrize("name", ["batch", "instance", "group", "none"])
    def test_norm_returns_valid_string(self, name):
        """Each normalization returns its canonical string."""
        result = get_norm(name)
        assert result == name

    def test_norm_none_available(self):
        """'none' is a valid normalization option (no normalization)."""
        result = get_norm("none")
        assert result == "none"


class TestPoolingRegistry:
    """Tests for pooling operation registry."""

    def test_all_pooling_retrievable(self):
        """All registered pooling operations can be retrieved."""
        for name in POOLING_REGISTRY:
            fn = get_pooling(name)
            assert callable(fn)

    def test_pooling_case_insensitive(self):
        """Pooling lookup is case-insensitive."""
        fn_lower = get_pooling("max")
        fn_upper = get_pooling("MAX")
        fn_mixed = get_pooling("MaX")
        assert fn_lower is fn_upper is fn_mixed

    def test_pooling_unknown_raises(self):
        """Unknown pooling raises ValueError with helpful message."""
        with pytest.raises(ValueError, match=r"Unknown pooling.*invalid_pool"):
            get_pooling("invalid_pool")

    def test_pooling_error_shows_available(self):
        """Error message lists available pooling operations."""
        with pytest.raises(ValueError, match=r"Available options:.*max"):
            get_pooling("nonexistent")

    @pytest.mark.parametrize("name", ["max", "avg", "none"])
    def test_pooling_functions_work(self, name):
        """Pooling functions produce expected output shapes."""
        fn = get_pooling(name)
        # Create a simple 4x4 feature map
        x = jnp.arange(16.0).reshape(1, 4, 4, 1)
        y = fn(x)
        assert y.ndim == 4
        assert y.dtype == x.dtype

    def test_max_pool_reduces_spatial(self):
        """Max pooling reduces spatial dimensions."""
        max_pool = get_pooling("max")
        x = jnp.ones((1, 4, 4, 1))
        y = max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # With 2x2 window and stride 2, 4x4 -> 2x2
        assert y.shape == (1, 2, 2, 1)

    def test_max_pool_takes_maximum(self):
        """Max pooling selects maximum values."""
        max_pool = get_pooling("max")
        # Create a pattern where we know the max in each 2x2 window
        x = jnp.array(
            [
                [1.0, 2.0, 5.0, 6.0],
                [3.0, 4.0, 7.0, 8.0],
                [9.0, 10.0, 13.0, 14.0],
                [11.0, 12.0, 15.0, 16.0],
            ]
        ).reshape(1, 4, 4, 1)
        y = max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        expected = jnp.array([[4.0, 8.0], [12.0, 16.0]]).reshape(1, 2, 2, 1)
        assert jnp.allclose(y, expected)

    def test_avg_pool_reduces_spatial(self):
        """Average pooling reduces spatial dimensions."""
        avg_pool = get_pooling("avg")
        x = jnp.ones((1, 4, 4, 1))
        y = avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        assert y.shape == (1, 2, 2, 1)

    def test_avg_pool_computes_mean(self):
        """Average pooling computes mean values."""
        avg_pool = get_pooling("avg")
        # Each 2x2 window has values that average to a known value
        x = jnp.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [2.0, 4.0, 6.0, 8.0],
                [8.0, 10.0, 12.0, 14.0],
                [10.0, 12.0, 14.0, 16.0],
            ]
        ).reshape(1, 4, 4, 1)
        y = avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        # Top-left: (0+2+2+4)/4 = 2, Top-right: (4+6+6+8)/4 = 6
        # Bottom-left: (8+10+10+12)/4 = 10, Bottom-right: (12+14+14+16)/4 = 14
        expected = jnp.array([[2.0, 6.0], [10.0, 14.0]]).reshape(1, 2, 2, 1)
        assert jnp.allclose(y, expected)

    def test_none_pooling_is_identity(self):
        """'none' pooling is identity function."""
        no_pool = get_pooling("none")
        x = jnp.ones((1, 4, 4, 2))
        y = no_pool(x)
        assert jnp.array_equal(x, y)
        assert y.shape == x.shape


class TestRegistryConsistency:
    """Tests for consistency across registries."""

    def test_all_registries_have_none(self):
        """All registries support 'none' for no-op behavior."""
        assert "none" in ACTIVATION_REGISTRY
        assert "none" in NORM_REGISTRY
        assert "none" in POOLING_REGISTRY

    def test_registries_are_dictionaries(self):
        """All registries are dictionaries."""
        assert isinstance(ACTIVATION_REGISTRY, dict)
        assert isinstance(NORM_REGISTRY, dict)
        assert isinstance(POOLING_REGISTRY, dict)

    def test_registries_nonempty(self):
        """All registries contain at least some entries."""
        assert len(ACTIVATION_REGISTRY) >= 5
        assert len(NORM_REGISTRY) >= 3
        assert len(POOLING_REGISTRY) >= 2
