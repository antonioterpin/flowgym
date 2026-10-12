"""Tests for router-oriented ART of PIV validation helpers."""

import pytest

from flowgym.flow.art_of_piv import validation as art_of_piv_validation


class TestCNNArchitectureValidation:
    """Tests for normalize_router_config architecture validation."""

    @staticmethod
    def _validate(
        features, kernel_sizes, strides, mlp_hidden_dim=512, norm_fn="none"
    ):
        return art_of_piv_validation.normalize_router_config(
            {
                "features": features,
                "kernel_sizes": kernel_sizes,
                "strides": strides,
                "mlp_hidden_dim": mlp_hidden_dim,
                "norm_fn": norm_fn,
            }
        )

    def test_valid_architecture_lists(self):
        """Test validation with valid CNN parameters as lists."""
        config = self._validate(
            [64, 128, 256], [3, 3, 3], [1, 2, 2], 512, "batch"
        )
        assert config["features_list"] == (64, 128, 256)
        assert config["kernel_sizes"] == (3, 3, 3)
        assert config["strides"] == (1, 2, 2)

    def test_valid_architecture_tuples(self):
        """Test validation with valid CNN parameters as tuples."""
        config = self._validate((64, 128), (3, 3), (1, 2), 512, "group")
        assert config["features_list"] == (64, 128)
        assert config["kernel_sizes"] == (3, 3)
        assert config["strides"] == (1, 2)

    def test_valid_architecture_single_layer(self):
        """Test validation with single layer CNN."""
        config = self._validate([64], [3], [1], 256, "instance")
        assert config["features_list"] == (64,)
        assert config["kernel_sizes"] == (3,)
        assert config["strides"] == (1,)

    def test_empty_features(self):
        """Test ValueError with empty features list."""
        with pytest.raises(ValueError, match="features must be a non-empty"):
            self._validate([], [3], [1], 512, "batch")

    def test_mismatched_kernel_sizes(self):
        """Test ValueError with mismatched kernel sizes length."""
        with pytest.raises(ValueError, match="same length as features"):
            self._validate([64, 128], [3], [1, 1], 512, "batch")

    def test_mismatched_strides(self):
        """Test ValueError with mismatched strides length."""
        with pytest.raises(ValueError, match="same length as features"):
            self._validate([64, 128], [3, 3], [1], 512, "batch")

    def test_mismatched_kernel_strides(self):
        """Test ValueError with mismatched kernel_sizes and strides."""
        with pytest.raises(ValueError, match="same length"):
            self._validate([64], [3, 3], [1], 512, "batch")

    def test_invalid_kernel_size_zero(self):
        """Test TypeError with zero kernel size."""
        with pytest.raises(TypeError, match="positive integers"):
            self._validate([64], [0], [1], 512, "batch")

    def test_invalid_kernel_size_negative(self):
        """Test TypeError with negative kernel size."""
        with pytest.raises(TypeError, match="positive integers"):
            self._validate([64], [-3], [1], 512, "batch")

    def test_invalid_stride_zero(self):
        """Test TypeError with zero stride."""
        with pytest.raises(TypeError, match="positive integers"):
            self._validate([64], [3], [0], 512, "batch")

    def test_invalid_mlp_hidden_dim_zero(self):
        """Test ValueError with zero mlp_hidden_dim."""
        with pytest.raises(
            ValueError, match="mlp_hidden_dim must be a positive"
        ):
            self._validate([64], [3], [1], 0, "batch")

    def test_invalid_mlp_hidden_dim_negative(self):
        """Test ValueError with negative mlp_hidden_dim."""
        with pytest.raises(
            ValueError, match="mlp_hidden_dim must be a positive"
        ):
            self._validate([64], [3], [1], -512, "batch")

    def test_invalid_mlp_hidden_dim_float(self):
        """Test ValueError with float mlp_hidden_dim."""
        with pytest.raises(
            ValueError, match="mlp_hidden_dim must be a positive"
        ):
            self._validate([64], [3], [1], 512.5, "batch")

    def test_invalid_norm_fn_unknown(self):
        """Test ValueError with unknown norm_fn."""
        with pytest.raises(ValueError, match="norm_fn must be one of"):
            self._validate([64], [3], [1], 512, "unknown")

    def test_valid_norm_fn_none(self):
        """Test validation with 'none' norm_fn."""
        config = self._validate([64], [3], [1], 512, "none")
        assert config["features_list"] == (64,)


class TestPostprocessValidation:
    """Tests for validate_and_build_postprocess."""

    def test_empty_postprocess(self):
        """Test validation with empty postprocess sequence."""
        result = art_of_piv_validation.validate_and_build_postprocess(())
        assert result == ()

    def test_single_callable(self):
        """Test validation with single callable function."""

        def fn(x):
            return x

        result = art_of_piv_validation.validate_and_build_postprocess([fn])
        assert result == (fn,)

    def test_multiple_callables(self):
        """Test validation with multiple callable functions."""

        def fn1(x):
            return x

        def fn2(x):
            return x * 2

        result = art_of_piv_validation.validate_and_build_postprocess(
            [fn1, fn2]
        )
        assert result == (fn1, fn2)

    def test_callable_tuple(self):
        """Test validation with tuple of callables."""

        def fn1(x):
            return x

        def fn2(x):
            return x * 2

        result = art_of_piv_validation.validate_and_build_postprocess(
            (fn1, fn2)
        )
        assert result == (fn1, fn2)

    def test_mixed_types_error(self):
        """Test TypeError with mixed callables and strings."""
        with pytest.raises(
            TypeError, match="either all callables or all strings"
        ):
            art_of_piv_validation.validate_and_build_postprocess(
                ["softmax", lambda x: x]
            )

    def test_non_sequence_error(self):
        """Test TypeError when postprocess is not a sequence."""
        with pytest.raises(TypeError, match="must be a sequence"):
            art_of_piv_validation.validate_and_build_postprocess("softmax")

    def test_non_callable_non_string_error(self):
        """Test TypeError with element that's neither callable nor string."""
        with pytest.raises(TypeError, match="callables or strings"):
            art_of_piv_validation.validate_and_build_postprocess([123])


class TestNestedRouterConfigValidation:
    """Tests for normalize_router_config on the nested encoder/head schema."""

    def test_valid_minimal_config(self):
        """Validate a minimal nested router config."""
        config = {
            "encoder": {"features_list": [16, 32]},
            "head": {"hidden_dims": [32, 16]},
        }
        result = art_of_piv_validation.normalize_router_config(config)
        assert result["encoder"]["features_list"] == (16, 32)
        assert result["head"]["hidden_dims"] == (32, 16)

    def test_top_level_must_be_dict(self):
        """Non-dict router_config raises TypeError."""
        with pytest.raises(TypeError, match="router_config must be a dict"):
            art_of_piv_validation.normalize_router_config(
                [1, 2, 3]  # type: ignore[arg-type]
            )

    def test_encoder_required(self):
        """A config with 'head' but no 'encoder' raises ValueError."""
        with pytest.raises(ValueError, match="missing 'encoder'"):
            art_of_piv_validation.normalize_router_config(
                {"head": {"hidden_dims": [32]}}
            )

    def test_head_required(self):
        """A config with 'encoder' but no 'head' raises ValueError."""
        with pytest.raises(ValueError, match="missing 'head'"):
            art_of_piv_validation.normalize_router_config(
                {"encoder": {"features_list": [16]}}
            )

    def test_encoder_must_be_dict(self):
        """Non-dict encoder raises TypeError."""
        with pytest.raises(TypeError, match="encoder config must be a dict"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": [16, 32],
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_head_must_be_dict(self):
        """Non-dict head raises TypeError."""
        with pytest.raises(TypeError, match="head config must be a dict"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": [32, 16],
                }
            )

    def test_encoder_features_list_required(self):
        """Missing encoder features_list raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'features_list'"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"output_dim": 64},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_encoder_features_list_must_be_sequence(self):
        """Non-sequence features_list raises ValueError."""
        with pytest.raises(
            ValueError, match="features_list must be a list or tuple"
        ):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": "16,32"},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_encoder_features_list_must_be_nonempty(self):
        """Empty features_list raises ValueError."""
        with pytest.raises(ValueError, match="features_list must be non-empty"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": []},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_encoder_features_list_positive_integers(self):
        """Non-positive features_list entries raise ValueError."""
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16, 0]},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_encoder_output_dim_default(self):
        """encoder output_dim defaults to 64."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16]},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["encoder"]["output_dim"] == 64

    def test_encoder_output_dim_configurable(self):
        """encoder output_dim is configurable."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16], "output_dim": 128},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["encoder"]["output_dim"] == 128

    def test_encoder_output_dim_validation(self):
        """Non-positive output_dim raises ValueError."""
        with pytest.raises(
            ValueError, match="output_dim must be a positive integer"
        ):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16], "output_dim": 0},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_encoder_use_residual_default(self):
        """encoder use_residual defaults to False."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16]},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["encoder"]["use_residual"] is False

    def test_encoder_use_residual_configurable(self):
        """encoder use_residual is configurable."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16], "use_residual": True},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["encoder"]["use_residual"] is True

    def test_encoder_use_residual_must_be_boolean(self):
        """Non-boolean use_residual raises ValueError."""
        with pytest.raises(ValueError, match="use_residual must be a boolean"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16], "use_residual": "yes"},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_encoder_norm_fn_default(self):
        """encoder norm_fn defaults to 'none'."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16]},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["encoder"]["norm_fn"] == "none"

    def test_encoder_norm_fn_accepts_known_values(self):
        """encoder norm_fn accepts each valid normalization kind."""
        for norm_type in ["batch", "instance", "group", "none"]:
            result = art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {
                        "features_list": [16],
                        "norm_fn": norm_type,
                    },
                    "head": {"hidden_dims": [32]},
                }
            )
            assert result["encoder"]["norm_fn"] == norm_type

    def test_encoder_norm_fn_invalid(self):
        """Unknown encoder norm_fn raises ValueError."""
        with pytest.raises(ValueError, match="norm_fn must be one of"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16], "norm_fn": "layer"},
                    "head": {"hidden_dims": [32]},
                }
            )

    def test_head_hidden_dims_required(self):
        """Missing head hidden_dims raises ValueError."""
        with pytest.raises(ValueError, match="must contain 'hidden_dims'"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {"activation": "relu"},
                }
            )

    def test_head_hidden_dims_must_be_sequence(self):
        """Non-sequence hidden_dims raises ValueError."""
        with pytest.raises(
            ValueError, match="hidden_dims must be a list or tuple"
        ):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {"hidden_dims": 32},
                }
            )

    def test_head_hidden_dims_must_be_nonempty(self):
        """Empty head hidden_dims raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {"hidden_dims": []},
                }
            )

    def test_head_hidden_dims_positive_integers(self):
        """Non-positive head hidden_dims entries raise ValueError."""
        with pytest.raises(
            ValueError, match="must contain only positive integers"
        ):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {"hidden_dims": [32, -1]},
                }
            )

    def test_head_activation_default(self):
        """head activation defaults to 'relu'."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16]},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["head"]["activation"] == "relu"

    def test_head_activation_configurable(self):
        """head activation accepts known activation names."""
        for activation in ["gelu", "tanh", "sigmoid"]:
            result = art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {
                        "hidden_dims": [32],
                        "activation": activation,
                    },
                }
            )
            assert result["head"]["activation"] == activation

    def test_head_activation_invalid(self):
        """Invalid head activation raises ValueError."""
        with pytest.raises(ValueError, match="head activation:"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {
                        "hidden_dims": [32],
                        "activation": "invalid",
                    },
                }
            )

    def test_head_single_arm_default(self):
        """head single_arm defaults to False."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16]},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["head"]["single_arm"] is False

    def test_head_single_arm_must_be_boolean(self):
        """Non-boolean single_arm raises ValueError."""
        with pytest.raises(ValueError, match="single_arm must be a boolean"):
            art_of_piv_validation.normalize_router_config(
                {
                    "encoder": {"features_list": [16]},
                    "head": {"hidden_dims": [32], "single_arm": "yes"},
                }
            )

    def test_full_config_normalization(self):
        """Full nested config returns immutable tuples for sequence fields."""
        config = {
            "encoder": {
                "features_list": [16, 32, 64],
                "output_dim": 128,
                "use_residual": True,
                "norm_fn": "batch",
            },
            "head": {
                "hidden_dims": [64, 32, 16],
                "activation": "gelu",
            },
        }
        result = art_of_piv_validation.normalize_router_config(config)
        assert result["encoder"]["features_list"] == (16, 32, 64)
        assert result["head"]["hidden_dims"] == (64, 32, 16)
        assert result["encoder"]["output_dim"] == 128
        assert result["encoder"]["use_residual"] is True
        assert result["encoder"]["norm_fn"] == "batch"
        assert result["head"]["activation"] == "gelu"


class TestRouterConfigSchemaDispatch:
    """normalize_router_config picks the right schema based on shape."""

    def test_nested_schema_tagged_cnn_router(self):
        """Both encoder and head present → tagged as cnn_router."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "encoder": {"features_list": [16]},
                "head": {"hidden_dims": [32]},
            }
        )
        assert result["type"] == "cnn_router"

    def test_flat_schema_tagged_cnn_q_estimator(self):
        """Flat features/kernel_sizes/strides → tagged as cnn_q_estimator."""
        result = art_of_piv_validation.normalize_router_config(
            {
                "features": [16, 32],
                "kernel_sizes": [3, 3],
                "strides": [1, 2],
            }
        )
        assert result["type"] == "cnn_q_estimator"
