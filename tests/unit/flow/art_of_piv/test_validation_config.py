"""Tests for configuration-loading ART of PIV validation helpers."""

import pytest

from flowgym.flow.art_of_piv import validation as art_of_piv_validation


class TestEstimatorsConfigValidation:
    """Tests for validate_and_load_estimators_config."""

    def test_invalid_path_type_int(self):
        """Test TypeError when path is an integer."""
        with pytest.raises(
            TypeError, match="estimators_list_path must be a string"
        ):
            art_of_piv_validation.validate_and_load_estimators_config(123)

    def test_invalid_path_type_none(self):
        """Test TypeError when path is None."""
        with pytest.raises(
            TypeError, match="estimators_list_path must be a string"
        ):
            art_of_piv_validation.validate_and_load_estimators_config(None)

    def test_invalid_extension_json(self):
        """Test ValueError when path ends with .json."""
        with pytest.raises(ValueError, match=r"must be a \.yaml file"):
            art_of_piv_validation.validate_and_load_estimators_config(
                "config.json"
            )

    def test_invalid_extension_no_extension(self):
        """Test ValueError when path has no extension."""
        with pytest.raises(ValueError, match=r"must be a \.yaml file"):
            art_of_piv_validation.validate_and_load_estimators_config("config")

    def test_invalid_extension_yml(self):
        """Test ValueError when path ends with .yml (not .yaml)."""
        with pytest.raises(ValueError, match=r"must be a \.yaml file"):
            art_of_piv_validation.validate_and_load_estimators_config(
                "config.yml"
            )
