"""Tests for MLP builder.

tests/nn/test_mlp_builders.py
"""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

mlp_module = pytest.importorskip(
    "flowgym.nn.mlp", reason="flowgym.nn.mlp is required for these tests"
)
MLP = mlp_module.MLP
build_mlp_from_config = mlp_module.build_mlp_from_config


class TestMLPBuilder:
    """Test the build_mlp_from_config function."""

    def test_minimal_config(self):
        """Build MLP with minimal config (only hidden_dims)."""
        config = {"hidden_dims": [64, 32]}
        model = build_mlp_from_config(config)

        assert isinstance(model, MLP)
        assert model.hidden_dims == (64, 32)
        assert model.output_dim == 1  # default
        assert model.activation == nn.relu  # default
        assert model.use_bias is True  # default

    def test_full_config(self):
        """Build MLP with all config fields."""
        config = {
            "hidden_dims": [128, 64, 32],
            "output_dim": 5,
            "activation": "tanh",
            "use_bias": False,
        }
        model = build_mlp_from_config(config)

        assert model.hidden_dims == (128, 64, 32)
        assert model.output_dim == 5
        assert model.activation == nn.tanh
        assert model.use_bias is False

    def test_missing_hidden_dims_raises(self):
        """Missing hidden_dims raises KeyError."""
        config = {"output_dim": 1}
        with pytest.raises(KeyError, match="Config must contain 'hidden_dims'"):
            build_mlp_from_config(config)

    def test_hidden_dims_must_be_sequence(self):
        """hidden_dims must be list or tuple."""
        config = {"hidden_dims": 64}  # not a list/tuple
        with pytest.raises(TypeError, match="must be a list or tuple"):
            build_mlp_from_config(config)

    def test_hidden_dims_cannot_be_empty(self):
        """hidden_dims cannot be empty."""
        config = {"hidden_dims": []}
        with pytest.raises(ValueError, match="must be non-empty"):
            build_mlp_from_config(config)

    def test_hidden_dims_must_be_positive(self):
        """All hidden dimensions must be positive integers."""
        config = {"hidden_dims": [64, -32]}
        with pytest.raises(ValueError, match="must be positive integers"):
            build_mlp_from_config(config)

    def test_hidden_dims_must_be_integers(self):
        """All hidden dimensions must be integers."""
        config = {"hidden_dims": [64, 32.5]}
        with pytest.raises(ValueError, match="must be positive integers"):
            build_mlp_from_config(config)

    def test_output_dim_must_be_positive(self):
        """output_dim must be a positive integer."""
        config = {"hidden_dims": [64], "output_dim": -1}
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_mlp_from_config(config)

    def test_output_dim_must_be_integer(self):
        """output_dim must be an integer."""
        config = {"hidden_dims": [64], "output_dim": 1.5}
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_mlp_from_config(config)

    def test_activation_must_be_string(self):
        """activation must be a string."""
        config = {"hidden_dims": [64], "activation": nn.relu}
        with pytest.raises(TypeError, match="must be a string"):
            build_mlp_from_config(config)

    def test_invalid_activation_raises(self):
        """Invalid activation name raises ValueError."""
        config = {"hidden_dims": [64], "activation": "invalid_activation"}
        with pytest.raises(
            ValueError, match="Unknown activation 'invalid_activation'"
        ):
            build_mlp_from_config(config)

    def test_use_bias_must_be_bool(self):
        """use_bias must be a boolean."""
        config = {"hidden_dims": [64], "use_bias": "yes"}
        with pytest.raises(TypeError, match="must be a boolean"):
            build_mlp_from_config(config)

    def test_converts_list_to_tuple(self):
        """Builder converts list to tuple for hidden_dims."""
        config = {"hidden_dims": [64, 32]}
        model = build_mlp_from_config(config)
        assert isinstance(model.hidden_dims, tuple)
        assert model.hidden_dims == (64, 32)


class TestMLPForward:
    """Test MLP forward pass."""

    def test_forward_pass_1d_input(self):
        """MLP forward pass with 1D input."""
        config = {"hidden_dims": [8, 4], "output_dim": 1}
        model = build_mlp_from_config(config)

        # Initialize parameters
        key = jnp.array([0, 0], dtype=jnp.uint32)
        dummy_input = jnp.zeros((10,))  # 10 input features
        params = model.init(key, dummy_input)

        # Forward pass
        output = model.apply(params, dummy_input)
        assert output.shape == (1,)

    def test_forward_pass_batch(self):
        """MLP forward pass with batched input."""
        config = {"hidden_dims": [16, 8], "output_dim": 3}
        model = build_mlp_from_config(config)

        # Initialize parameters
        key = jnp.array([0, 0], dtype=jnp.uint32)
        dummy_input = jnp.zeros((5,))  # 5 input features
        params = model.init(key, dummy_input)

        # Batched forward pass
        batch_input = jnp.zeros((4, 5))  # batch of 4 samples
        output = model.apply(params, batch_input)
        assert output.shape == (4, 3)

    def test_different_activations(self):
        """Test MLP with different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu"]
        for act in activations:
            config = {"hidden_dims": [8], "activation": act}
            model = build_mlp_from_config(config)

            key = jnp.array([0, 0], dtype=jnp.uint32)
            dummy_input = jnp.zeros((5,))
            params = model.init(key, dummy_input)
            output = model.apply(params, dummy_input)
            assert output.shape == (1,)

    def test_no_bias(self):
        """Test MLP without bias terms."""
        config = {"hidden_dims": [8, 4], "use_bias": False}
        model = build_mlp_from_config(config)

        key = jnp.array([0, 0], dtype=jnp.uint32)
        dummy_input = jnp.zeros((5,))
        params = model.init(key, dummy_input)

        # Check that no bias parameters exist
        all_params_flat = jax.tree_util.tree_leaves(params["params"])
        # All parameters should be 2D (weight matrices only)
        for p in all_params_flat:
            assert p.ndim == 2  # Only weight matrices, no bias vectors

    def test_with_bias(self):
        """Test MLP with bias terms."""
        config = {"hidden_dims": [8, 4], "use_bias": True}
        model = build_mlp_from_config(config)

        key = jnp.array([0, 0], dtype=jnp.uint32)
        dummy_input = jnp.zeros((5,))
        params = model.init(key, dummy_input)

        # Check that bias parameters exist
        all_params_flat = jax.tree_util.tree_leaves(params["params"])
        # Should have both 2D (weights) and 1D (biases) parameters
        has_1d = any(p.ndim == 1 for p in all_params_flat)
        has_2d = any(p.ndim == 2 for p in all_params_flat)
        assert has_1d  # bias vectors
        assert has_2d  # weight matrices
