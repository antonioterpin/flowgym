"""Tests for flowgym.utils.load_configuration."""

import pytest

from flowgym.utils import load_configuration


def test_extends_merges_over_the_base_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """``extends`` (from the working directory) is merged under the file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base.yaml").write_text(
        "estimator: dis_jax\nconfig: {levels: 3, patch_size: 9}\n"
    )
    (tmp_path / "models").mkdir()
    model = tmp_path / "models" / "m.yaml"
    model.write_text("extends: base.yaml\nconfig: {patch_size: 13}\n")

    assert load_configuration(model) == {
        "estimator": "dis_jax",
        "config": {"levels": 3, "patch_size": 13},
    }
