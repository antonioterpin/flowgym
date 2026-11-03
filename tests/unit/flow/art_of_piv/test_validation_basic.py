"""Tests for basic ART of PIV validation helpers."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.flow.art_of_piv import validation as art_of_piv_validation


class TestBooleanFlagsValidation:
    """Tests for validate_boolean_flags."""

    def test_valid_flags(self):
        """Test validation with all valid boolean flags."""
        result = art_of_piv_validation.validate_boolean_flags(True, False, True)
        assert result == (True, False, True)

    def test_all_false_flags(self):
        """Test validation with all False flags."""
        result = art_of_piv_validation.validate_boolean_flags(
            False, False, False
        )
        assert result == (False, False, False)

    def test_invalid_output_full_res_string(self):
        """Test TypeError when output_full_res is a string."""
        with pytest.raises(
            TypeError, match="output_full_res must be a boolean"
        ):
            art_of_piv_validation.validate_boolean_flags("True", False, True)

    def test_invalid_output_full_res_int(self):
        """Test TypeError when output_full_res is an int."""
        with pytest.raises(
            TypeError, match="output_full_res must be a boolean"
        ):
            art_of_piv_validation.validate_boolean_flags(1, False, True)

    def test_invalid_use_temporal_propagation(self):
        """Test TypeError when use_temporal_propagation is not boolean."""
        with pytest.raises(
            TypeError, match="use_temporal_propagation must be a boolean"
        ):
            art_of_piv_validation.validate_boolean_flags(True, 1, True)

    def test_invalid_use_last_estimate(self):
        """Test TypeError when use_last_estimate is not boolean."""
        with pytest.raises(
            TypeError, match="use_last_estimate must be a boolean"
        ):
            art_of_piv_validation.validate_boolean_flags(True, False, None)


class TestTrainTypeValidation:
    """Tests for validate_train_type."""

    def test_valid_supervised(self):
        """Test validation with 'supervised' train type."""
        assert (
            art_of_piv_validation.validate_train_type("supervised")
            == "supervised"
        )

    def test_valid_rl(self):
        """Test validation with 'rl' train type."""
        assert art_of_piv_validation.validate_train_type("rl") == "rl"

    def test_invalid_train_type(self):
        """Test ValueError with invalid train type."""
        with pytest.raises(ValueError, match="train_type must be one of"):
            art_of_piv_validation.validate_train_type("invalid")

    def test_invalid_train_type_capitalized(self):
        """Test ValueError when train type is capitalized."""
        with pytest.raises(ValueError, match="train_type must be one of"):
            art_of_piv_validation.validate_train_type("Supervised")


class TestRNGValidation:
    """Tests for validate_rng."""

    def test_none_rng(self):
        """Test that None passes through."""
        assert art_of_piv_validation.validate_rng(None) is None

    def test_int_conversion(self):
        """Test that int is converted to PRNGKey."""
        result = art_of_piv_validation.validate_rng(42)
        assert result is not None
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (2,)

    def test_zero_int_conversion(self):
        """Test that int 0 is converted to PRNGKey."""
        result = art_of_piv_validation.validate_rng(0)
        assert result is not None
        assert isinstance(result, jnp.ndarray)

    def test_prngkey_passthrough(self):
        """Test that PRNGKey passes through unchanged."""
        key = jax.random.PRNGKey(42)
        result = art_of_piv_validation.validate_rng(key)
        assert jnp.array_equal(result, key)
