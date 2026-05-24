"""Tests for CNN model builders.

This module tests the config-driven builders for:
- CNNDensityModel
- CNNQEstimatorModel
- Model dispatcher
"""

import jax
import jax.numpy as jnp
import pytest

from flowgym.nn.cnn import (
    MODEL_REGISTRY,
    CNNDensityModel,
    CNNFlowFieldModel,
    CNNQEstimatorModel,
    build_cnn_density_model_from_config,
    build_cnn_flow_field_model_from_config,
    build_cnn_q_estimator_model_from_config,
    build_model_from_config,
)


class TestCNNDensityModelBuilder:
    """Tests for CNNDensityModel builder."""

    def test_minimal_config(self):
        """Build CNNDensityModel with only required fields."""
        config = {"features_list": [32, 64]}
        model = build_cnn_density_model_from_config(config)
        assert isinstance(model, CNNDensityModel)
        assert model.features_list == [32, 64]
        assert model.use_residual is False
        assert model.norm_fn == "none"

    def test_features_list_required(self):
        """CNNDensityModel builder requires 'features_list' field."""
        with pytest.raises(ValueError, match="must contain 'features_list'"):
            build_cnn_density_model_from_config({})

    def test_features_list_must_be_list(self):
        """CNNDensityModel features_list must be a list."""
        with pytest.raises(ValueError, match="features_list must be a list"):
            build_cnn_density_model_from_config({"features_list": (32, 64)})

    def test_features_list_must_be_non_empty(self):
        """CNNDensityModel features_list must be non-empty."""
        with pytest.raises(ValueError, match="features_list must be non-empty"):
            build_cnn_density_model_from_config({"features_list": []})

    def test_features_list_must_contain_positive_integers(self):
        """CNNDensityModel features_list must contain only positive integers."""
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            build_cnn_density_model_from_config({"features_list": [32, 0, 64]})
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            build_cnn_density_model_from_config({"features_list": [32, -5, 64]})
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            build_cnn_density_model_from_config(
                {"features_list": [32, 64.5, 128]}
            )

    def test_use_residual_default_false(self):
        """CNNDensityModel use_residual defaults to False."""
        config = {"features_list": [32]}
        model = build_cnn_density_model_from_config(config)
        assert model.use_residual is False

    def test_use_residual_can_be_set(self):
        """CNNDensityModel use_residual can be set to True."""
        config = {"features_list": [32, 64], "use_residual": True}
        model = build_cnn_density_model_from_config(config)
        assert model.use_residual is True

    @pytest.mark.parametrize(
        "raw_value", ["true", "false", "false-ish", 0, 1, 2, -1, 1.2, None]
    )
    def test_use_residual_invalid_values_raise(self, raw_value: object):
        """CNNDensityModel rejects ambiguous use_residual values."""
        with pytest.raises(ValueError, match="use_residual"):
            build_cnn_density_model_from_config(
                {"features_list": [32, 64], "use_residual": raw_value}
            )

    def test_norm_fn_default_none(self):
        """CNNDensityModel norm_fn defaults to 'none'."""
        config = {"features_list": [32]}
        model = build_cnn_density_model_from_config(config)
        assert model.norm_fn == "none"

    def test_norm_fn_from_registry(self):
        """Build CNNDensityModel with various normalization types."""
        for norm_type in ["batch", "instance", "group", "none"]:
            config = {"features_list": [32, 64], "norm_fn": norm_type}
            model = build_cnn_density_model_from_config(config)
            assert model.norm_fn == norm_type

    def test_invalid_norm_fn(self):
        """Invalid norm_fn raises error."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            build_cnn_density_model_from_config(
                {"features_list": [32], "norm_fn": "layer"}
            )

    def test_full_config(self):
        """Build CNNDensityModel with all fields specified."""
        config = {
            "features_list": [16, 32, 64, 128],
            "use_residual": True,
            "norm_fn": "batch",
        }
        model = build_cnn_density_model_from_config(config)
        assert model.features_list == [16, 32, 64, 128]
        assert model.use_residual is True
        assert model.norm_fn == "batch"

    def test_model_is_callable(self):
        """Built CNNDensityModel can be called and produces correct output."""
        config = {"features_list": [16, 32]}
        model = build_cnn_density_model_from_config(config)

        # Initialize with input (H, W)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 64))  # Batch of 2, 64x64 images
        variables = model.init(key, x)
        y = model.apply(variables, x)  # type: ignore[misc]

        # Model outputs scalar per example
        assert y.shape == (2,)  # type: ignore[union-attr]
        assert isinstance(model, CNNDensityModel)

    def test_model_is_callable_with_residual(self):
        """CNNDensityModel forward pass supports residual branch."""
        config = {"features_list": [8, 16], "use_residual": True}
        model = build_cnn_density_model_from_config(config)
        key = jax.random.PRNGKey(100)
        x = jnp.ones((2, 32, 32))
        variables = model.init(key, x)
        y = model.apply(variables, x)  # type: ignore[misc]
        assert y.shape == (2,)  # type: ignore[union-attr]


class TestCNNQEstimatorModelBuilder:
    """Tests for CNNQEstimatorModel builder."""

    def test_minimal_config(self):
        """Build CNNQEstimatorModel with all required fields."""
        config = {
            "features_list": (32, 64),
            "kernel_sizes": (4, 3),
            "strides": (2, 2),
            "num_q_values": 3,
        }
        model = build_cnn_q_estimator_model_from_config(config)
        assert isinstance(model, CNNQEstimatorModel)
        assert model.features_list == (32, 64)
        assert model.kernel_sizes == (4, 3)
        assert model.strides == (2, 2)
        assert model.num_q_values == 3
        assert model.norm_fn == "none"
        assert model.mlp_hidden_dim == 512

    def test_features_list_required(self):
        """CNNQEstimatorModel builder requires 'features_list' field."""
        with pytest.raises(ValueError, match="must contain 'features_list'"):
            build_cnn_q_estimator_model_from_config(
                {
                    "kernel_sizes": (4, 3),
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

    def test_kernel_sizes_required(self):
        """CNNQEstimatorModel builder requires 'kernel_sizes' field."""
        with pytest.raises(ValueError, match="must contain 'kernel_sizes'"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

    def test_strides_required(self):
        """CNNQEstimatorModel builder requires 'strides' field."""
        with pytest.raises(ValueError, match="must contain 'strides'"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3),
                    "num_q_values": 3,
                }
            )

    def test_num_q_values_required(self):
        """CNNQEstimatorModel builder requires 'num_q_values' field."""
        with pytest.raises(ValueError, match="must contain 'num_q_values'"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3),
                    "strides": (2, 2),
                }
            )

    def test_features_list_accepts_list_or_tuple(self):
        """CNNQEstimatorModel features_list can be list or tuple."""
        # Test with list - should be converted to tuple
        config_list = {
            "features_list": [32, 64],
            "kernel_sizes": [4, 3],
            "strides": [2, 2],
            "num_q_values": 3,
        }
        model_list = build_cnn_q_estimator_model_from_config(config_list)
        assert model_list.features_list == (32, 64)
        assert model_list.kernel_sizes == (4, 3)
        assert model_list.strides == (2, 2)

        # Test with tuple - should remain tuple
        config_tuple = {
            "features_list": (32, 64),
            "kernel_sizes": (4, 3),
            "strides": (2, 2),
            "num_q_values": 3,
        }
        model_tuple = build_cnn_q_estimator_model_from_config(config_tuple)
        assert model_tuple.features_list == (32, 64)

    def test_features_list_must_be_non_empty(self):
        """CNNQEstimatorModel features_list must be non-empty."""
        with pytest.raises(ValueError, match="features_list must be non-empty"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": [],
                    "kernel_sizes": [],
                    "strides": [],
                    "num_q_values": 3,
                }
            )

    def test_features_list_must_contain_positive_integers(self):
        """CNNQEstimatorModel features_list must contain positive integers."""
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 0),
                    "kernel_sizes": (4, 3),
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

    def test_kernel_sizes_must_contain_positive_integers(self):
        """CNNQEstimatorModel kernel_sizes must contain positive integers."""
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, -1),
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

    def test_strides_must_contain_positive_integers(self):
        """CNNQEstimatorModel strides must contain positive integers."""
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3),
                    "strides": (2, 0),
                    "num_q_values": 3,
                }
            )

    def test_lengths_must_match(self):
        """kernel_sizes and strides must match features_list length."""
        # Test kernel_sizes length mismatch
        with pytest.raises(
            ValueError, match="len\\(kernel_sizes\\)=3 must equal"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3, 3),
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

        # Test strides length mismatch
        with pytest.raises(ValueError, match="len\\(strides\\)=3 must equal"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3),
                    "strides": (2, 2, 1),
                    "num_q_values": 3,
                }
            )

    def test_num_q_values_must_be_positive(self):
        """CNNQEstimatorModel num_q_values must be positive."""
        with pytest.raises(ValueError, match="num_q_values must be positive"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3),
                    "strides": (2, 2),
                    "num_q_values": 0,
                }
            )

    def test_mlp_hidden_dim_default(self):
        """CNNQEstimatorModel mlp_hidden_dim defaults to 512."""
        config = {
            "features_list": (32,),
            "kernel_sizes": (4,),
            "strides": (2,),
            "num_q_values": 3,
        }
        model = build_cnn_q_estimator_model_from_config(config)
        assert model.mlp_hidden_dim == 512

    def test_mlp_hidden_dim_can_be_set(self):
        """CNNQEstimatorModel mlp_hidden_dim can be set."""
        config = {
            "features_list": (32,),
            "kernel_sizes": (4,),
            "strides": (2,),
            "num_q_values": 3,
            "mlp_hidden_dim": 256,
        }
        model = build_cnn_q_estimator_model_from_config(config)
        assert model.mlp_hidden_dim == 256

    def test_mlp_hidden_dim_must_be_positive(self):
        """CNNQEstimatorModel mlp_hidden_dim must be positive."""
        with pytest.raises(ValueError, match="mlp_hidden_dim must be positive"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32,),
                    "kernel_sizes": (4,),
                    "strides": (2,),
                    "num_q_values": 3,
                    "mlp_hidden_dim": -1,
                }
            )

    def test_norm_fn_default_none(self):
        """CNNQEstimatorModel norm_fn defaults to 'none'."""
        config = {
            "features_list": (32,),
            "kernel_sizes": (4,),
            "strides": (2,),
            "num_q_values": 3,
        }
        model = build_cnn_q_estimator_model_from_config(config)
        assert model.norm_fn == "none"

    def test_norm_fn_from_registry(self):
        """Build CNNQEstimatorModel with various normalization types."""
        for norm_type in ["batch", "instance", "group", "none"]:
            config = {
                "features_list": (32, 64),
                "kernel_sizes": (4, 3),
                "strides": (2, 2),
                "num_q_values": 3,
                "norm_fn": norm_type,
            }
            model = build_cnn_q_estimator_model_from_config(config)
            assert model.norm_fn == norm_type

    def test_invalid_norm_fn(self):
        """Invalid norm_fn raises error."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32,),
                    "kernel_sizes": (4,),
                    "strides": (2,),
                    "num_q_values": 3,
                    "norm_fn": "layer",
                }
            )

    def test_features_list_invalid_type_raises(self):
        """features_list must be list/tuple."""
        with pytest.raises(
            ValueError, match="features_list must be a tuple or list"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": "32,64",
                    "kernel_sizes": (4, 3),
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

    def test_kernel_sizes_invalid_type_raises(self):
        """kernel_sizes must be list/tuple."""
        with pytest.raises(
            ValueError, match="kernel_sizes must be a tuple or list"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": 3,
                    "strides": (2, 2),
                    "num_q_values": 3,
                }
            )

    def test_strides_invalid_type_raises(self):
        """strides must be list/tuple."""
        with pytest.raises(ValueError, match="strides must be a tuple or list"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32, 64),
                    "kernel_sizes": (4, 3),
                    "strides": 2,
                    "num_q_values": 3,
                }
            )

    def test_postprocess_default(self):
        """CNNQEstimatorModel postprocess defaults to empty list."""
        config = {
            "features_list": (32,),
            "kernel_sizes": (4,),
            "strides": (2,),
            "num_q_values": 3,
        }
        model = build_cnn_q_estimator_model_from_config(config)
        # Postprocess pipeline converts to tuple of callables
        assert len(model.postprocess) == 0

    def test_postprocess_can_be_set(self):
        """CNNQEstimatorModel postprocess can be configured."""
        config = {
            "features_list": (32,),
            "kernel_sizes": (4,),
            "strides": (2,),
            "num_q_values": 3,
            "postprocess": ["abs", "softplus"],
        }
        model = build_cnn_q_estimator_model_from_config(config)
        assert len(model.postprocess) == 2

    def test_postprocess_none_is_noop_and_callable(self):
        """'none' postprocess can be configured and applied in forward pass."""
        config = {
            "features_list": (16, 32),
            "kernel_sizes": (4, 3),
            "strides": (2, 2),
            "num_q_values": 3,
            "postprocess": ["none"],
        }
        model = build_cnn_q_estimator_model_from_config(config)

        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 64, 3)) * 255.0
        variables = model.init(key, x)
        y = model.apply(variables, x)  # type: ignore[misc]

        assert y.shape == (2, 3)  # type: ignore[union-attr]

    def test_postprocess_must_be_list_or_tuple(self):
        """CNNQEstimatorModel postprocess must be list or tuple."""
        with pytest.raises(
            ValueError, match="postprocess must be a list or tuple"
        ):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32,),
                    "kernel_sizes": (4,),
                    "strides": (2,),
                    "num_q_values": 3,
                    "postprocess": "abs",
                }
            )

    def test_invalid_postprocess_operation(self):
        """Invalid postprocess operation raises error."""
        with pytest.raises(ValueError, match="Unknown postprocess operation"):
            build_cnn_q_estimator_model_from_config(
                {
                    "features_list": (32,),
                    "kernel_sizes": (4,),
                    "strides": (2,),
                    "num_q_values": 3,
                    "postprocess": ["invalid_op"],
                }
            )

    def test_full_config(self):
        """Build CNNQEstimatorModel with all fields specified."""
        config = {
            "features_list": [16, 32, 64],
            "kernel_sizes": [5, 4, 3],
            "strides": [2, 2, 1],
            "num_q_values": 5,
            "norm_fn": "instance",
            "mlp_hidden_dim": 256,
            "postprocess": ["softplus"],
        }
        model = build_cnn_q_estimator_model_from_config(config)
        assert model.features_list == (16, 32, 64)
        assert model.kernel_sizes == (5, 4, 3)
        assert model.strides == (2, 2, 1)
        assert model.num_q_values == 5
        assert model.norm_fn == "instance"
        assert model.mlp_hidden_dim == 256
        assert len(model.postprocess) == 1

    def test_model_is_callable(self):
        """Built CNNQEstimatorModel can be called and returns Q-values."""
        config = {
            "features_list": (16, 32),
            "kernel_sizes": (4, 3),
            "strides": (2, 2),
            "num_q_values": 3,
        }
        model = build_cnn_q_estimator_model_from_config(config)

        # Initialize with input (B, H, W, C)
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 64, 64, 3)) * 255.0  # Batch of 2, 64x64 RGB images
        variables = model.init(key, x)
        y = model.apply(variables, x)  # type: ignore[misc]

        # Model outputs Q-values
        # Batch=2, num_q_values=3
        assert y.shape == (2, 3)  # type: ignore[union-attr]
        assert isinstance(model, CNNQEstimatorModel)


class TestModelDispatcher:
    """Tests for model dispatcher and registry."""

    def test_registry_contains_expected_models(self):
        """MODEL_REGISTRY contains expected model types."""
        assert "cnn_density" in MODEL_REGISTRY
        assert "cnn_flow_field" in MODEL_REGISTRY
        assert "cnn_q_estimator" in MODEL_REGISTRY

    def test_dispatcher_cnn_density(self):
        """Dispatcher correctly routes to CNNDensityModel builder."""
        config = {"type": "cnn_density", "features_list": [32, 64]}
        model = build_model_from_config(config)
        assert isinstance(model, CNNDensityModel)
        assert model.features_list == [32, 64]

    def test_dispatcher_cnn_q_estimator(self):
        """Dispatcher correctly routes to CNNQEstimatorModel builder."""
        config = {
            "type": "cnn_q_estimator",
            "features_list": (32, 64),
            "kernel_sizes": (4, 3),
            "strides": (2, 2),
            "num_q_values": 3,
        }
        model = build_model_from_config(config)
        assert isinstance(model, CNNQEstimatorModel)

    def test_flow_field_builder_and_dispatcher(self):
        """Flow-field model is buildable directly and via dispatcher."""
        config = {
            "features_list": [32, 64],
            "use_residual": False,
            "norm_fn": "none",
        }
        model = build_cnn_flow_field_model_from_config(config)
        assert isinstance(model, CNNFlowFieldModel)

        dispatched = build_model_from_config(
            {"type": "cnn_flow_field", **config}
        )
        assert isinstance(dispatched, CNNFlowFieldModel)

    @pytest.mark.parametrize(
        "raw_value",
        ["true", "false", "on", "no", "sometimes", 0, 1, 10, -2, [], None],
    )
    def test_flow_field_builder_invalid_use_residual_raises(
        self, raw_value: object
    ):
        """Flow-field builder rejects invalid use_residual values."""
        with pytest.raises(ValueError, match="use_residual"):
            build_cnn_flow_field_model_from_config(
                {"features_list": [32, 64], "use_residual": raw_value}
            )

    def test_flow_field_model_forward_conv_and_residual(self):
        """CNNFlowFieldModel forward supports both branch types."""
        key = jax.random.PRNGKey(101)
        x = jnp.ones((2, 16, 16, 3))

        conv_model = CNNFlowFieldModel(
            features_list=[8, 16], use_residual=False, norm_fn="none"
        )
        conv_vars = conv_model.init(key, x)
        conv_out = conv_model.apply(conv_vars, x)  # type: ignore[misc]
        assert conv_out.shape == (2, 16, 16, 2)

        res_model = CNNFlowFieldModel(
            features_list=[8, 16], use_residual=True, norm_fn="none"
        )
        res_vars = res_model.init(key, x)
        res_out = res_model.apply(res_vars, x)  # type: ignore[misc]
        assert res_out.shape == (2, 16, 16, 2)

    def test_flow_field_builder_validation_errors(self):
        """Flow-field builder validates required and typed fields."""
        with pytest.raises(ValueError, match="must contain 'features_list'"):
            build_cnn_flow_field_model_from_config({})
        with pytest.raises(ValueError, match="features_list must be a list"):
            build_cnn_flow_field_model_from_config({"features_list": (8, 16)})
        with pytest.raises(ValueError, match="features_list must be non-empty"):
            build_cnn_flow_field_model_from_config({"features_list": []})
        with pytest.raises(
            ValueError,
            match="features_list must contain only positive integers",
        ):
            build_cnn_flow_field_model_from_config({"features_list": [8, 0]})

    def test_dispatcher_requires_type(self):
        """Dispatcher requires 'type' field."""
        with pytest.raises(ValueError, match="must contain 'type'"):
            build_model_from_config({"features_list": [32, 64]})

    def test_dispatcher_unknown_type(self):
        """Dispatcher raises error for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type 'unknown'"):
            build_model_from_config(
                {"type": "unknown", "features_list": [32, 64]}
            )

    def test_dispatcher_case_insensitive(self):
        """Dispatcher handles case-insensitive model types."""
        config_upper = {"type": "CNN_DENSITY", "features_list": [32, 64]}
        model_upper = build_model_from_config(config_upper)
        assert isinstance(model_upper, CNNDensityModel)

        config_mixed = {"type": "Cnn_Density", "features_list": [32, 64]}
        model_mixed = build_model_from_config(config_mixed)
        assert isinstance(model_mixed, CNNDensityModel)

    def test_dispatcher_error_message_lists_available(self):
        """Dispatcher error message lists available model types."""
        with pytest.raises(ValueError, match="Available:"):
            build_model_from_config(
                {"type": "invalid", "features_list": [32, 64]}
            )
