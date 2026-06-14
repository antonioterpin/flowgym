"""Tests for training-related Art of PIV validation helpers."""

import pytest

from flowgym.flow.art_of_piv import validation as art_of_piv_validation


class TestTrainingModeAndLossValidation:
    """Tests for validate_training_mode_and_loss."""

    def test_valid_bandits_mode(self):
        """Test validation with bandits mode."""
        result = art_of_piv_validation.validate_training_mode_and_loss(
            "bandits", {"name": "l2"}
        )
        assert result == "bandits"

    def test_valid_bandits_mode_no_config(self):
        """Test validation with bandits mode and no loss config."""
        result = art_of_piv_validation.validate_training_mode_and_loss(
            "bandits", None
        )
        assert result == "bandits"

    def test_valid_regression_mode(self):
        """Test validation with regression mode."""
        result = art_of_piv_validation.validate_training_mode_and_loss(
            "regression", {"name": "l2"}
        )
        assert result == "regression"

    def test_valid_classification_mode_softmax(self):
        """Test validation with classification mode and softmax loss."""
        result = art_of_piv_validation.validate_training_mode_and_loss(
            "classification",
            {"name": "softmax_cross_entropy_with_integer_labels"},
        )
        assert result == "classification"

    def test_valid_classification_mode_hinge(self):
        """Test validation with classification mode and hinge loss."""
        result = art_of_piv_validation.validate_training_mode_and_loss(
            "classification", {"name": "multiclass_hinge"}
        )
        assert result == "classification"

    def test_valid_classification_mode_perceptron(self):
        """Test validation with classification mode and perceptron loss."""
        result = art_of_piv_validation.validate_training_mode_and_loss(
            "classification", {"name": "multiclass_perceptron"}
        )
        assert result == "classification"

    def test_invalid_classification_loss(self):
        """Test ValueError when classification mode has incompatible loss."""
        with pytest.raises(ValueError, match="Classification mode requires"):
            art_of_piv_validation.validate_training_mode_and_loss(
                "classification", {"name": "l2"}
            )

    def test_invalid_classification_loss_no_config(self):
        """Test ValueError when classification mode has no loss config."""
        with pytest.raises(ValueError, match="Classification mode requires"):
            art_of_piv_validation.validate_training_mode_and_loss(
                "classification", None
            )

    def test_invalid_training_mode(self):
        """Test ValueError with invalid training mode."""
        with pytest.raises(ValueError, match="training_mode must be one of"):
            art_of_piv_validation.validate_training_mode_and_loss(
                "invalid", None
            )


class TestExplorationConfigValidation:
    """Tests for validate_and_set_exploration_defaults."""

    def test_none_exploration_bandits_mode(self):
        """Test default exploration for bandits mode."""
        result = art_of_piv_validation.validate_and_set_exploration_defaults(
            None, "bandits"
        )
        assert result["name"] == "epsilon_greedy"
        assert result["schedule"]["value"] == 0.1

    def test_none_exploration_regression_mode(self):
        """Test default exploration for regression mode (greedy)."""
        result = art_of_piv_validation.validate_and_set_exploration_defaults(
            None, "regression"
        )
        assert result["name"] == "epsilon_greedy"
        assert result["schedule"]["value"] == 0.0

    def test_none_exploration_classification_mode(self):
        """Test default exploration for classification mode (greedy)."""
        result = art_of_piv_validation.validate_and_set_exploration_defaults(
            None, "classification"
        )
        assert result["name"] == "epsilon_greedy"
        assert result["schedule"]["value"] == 0.0

    def test_provided_exploration_config(self):
        """Test that provided config is validated and returned."""
        config = {
            "name": "epsilon_greedy",
            "schedule": {"name": "linear", "start": 1.0, "end": 0.1},
        }
        result = art_of_piv_validation.validate_and_set_exploration_defaults(
            config, "bandits"
        )
        assert result == config

    def test_invalid_schedule_type_string(self):
        """Test TypeError when schedule is a string."""
        with pytest.raises(TypeError, match="schedule' must be a dict"):
            art_of_piv_validation.validate_and_set_exploration_defaults(
                {"name": "epsilon_greedy", "schedule": "constant"}, "bandits"
            )

    def test_invalid_schedule_type_list(self):
        """Test TypeError when schedule is a list."""
        with pytest.raises(TypeError, match="schedule' must be a dict"):
            art_of_piv_validation.validate_and_set_exploration_defaults(
                {"name": "epsilon_greedy", "schedule": []}, "bandits"
            )

    def test_missing_schedule_key(self):
        """Test missing schedule key with the default empty schedule dict."""
        config = {"name": "epsilon_greedy"}
        result = art_of_piv_validation.validate_and_set_exploration_defaults(
            config, "bandits"
        )
        assert result == config


class TestGammaValidation:
    """Tests for validate_gamma."""

    def test_valid_gamma_zero(self):
        """Test validation with gamma = 0."""
        assert art_of_piv_validation.validate_gamma(0) == 0

    def test_valid_gamma_zero_float(self):
        """Test validation with gamma = 0.0."""
        assert art_of_piv_validation.validate_gamma(0.0) == 0.0

    def test_valid_gamma_middle(self):
        """Test validation with gamma in valid range."""
        assert art_of_piv_validation.validate_gamma(0.5) == 0.5

    def test_valid_gamma_near_one(self):
        """Test validation with gamma near 1."""
        assert art_of_piv_validation.validate_gamma(0.99) == 0.99

    def test_valid_gamma_very_small(self):
        """Test validation with very small positive gamma."""
        assert art_of_piv_validation.validate_gamma(0.001) == 0.001

    def test_invalid_gamma_negative_small(self):
        """Test ValueError with small negative gamma."""
        with pytest.raises(ValueError, match="gamma must be in the range"):
            art_of_piv_validation.validate_gamma(-0.1)

    def test_invalid_gamma_negative_large(self):
        """Test ValueError with large negative gamma."""
        with pytest.raises(ValueError, match="gamma must be in the range"):
            art_of_piv_validation.validate_gamma(-1.0)

    def test_invalid_gamma_equal_one(self):
        """Test ValueError with gamma = 1."""
        with pytest.raises(ValueError, match="gamma must be in the range"):
            art_of_piv_validation.validate_gamma(1.0)

    def test_invalid_gamma_greater_one(self):
        """Test ValueError with gamma > 1."""
        with pytest.raises(ValueError, match="gamma must be in the range"):
            art_of_piv_validation.validate_gamma(1.5)

    def test_invalid_gamma_type_string(self):
        """Test TypeError with string gamma."""
        with pytest.raises(TypeError, match="gamma must be a float or int"):
            art_of_piv_validation.validate_gamma("0.5")

    def test_invalid_gamma_type_none(self):
        """Test TypeError with None gamma."""
        with pytest.raises(TypeError, match="gamma must be a float or int"):
            art_of_piv_validation.validate_gamma(None)
