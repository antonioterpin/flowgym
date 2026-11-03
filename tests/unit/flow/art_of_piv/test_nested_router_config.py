"""Integration tests for nested encoder/head router_config in ArtOfPIV."""

from unittest.mock import MagicMock

import pytest
import yaml

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.nn.cnn import CNNQEstimatorModel, CNNRouterModel


def _write_dummy_estimators_yaml(tmp_path):
    """Create a minimal estimators-list YAML for stubbed initialization."""
    path = tmp_path / "estimators.yaml"
    path.write_text(
        yaml.safe_dump({"estimators": [{"estimator": "dummy", "config": {}}]}),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def stub_create_estimators(monkeypatch):
    """Stub ArtOfPIVEstimator._create_estimators for the test's lifetime only.

    Using ``monkeypatch.setattr`` ensures the original method is restored at
    test teardown, so this stub does not leak into other tests collected
    after this module.
    """
    monkeypatch.setattr(
        ArtOfPIVEstimator,
        "_create_estimators",
        MagicMock(
            return_value=(
                (lambda x: (None, {"epe": 1.0}),),
                (MagicMock(),),
                (MagicMock(),),
            )
        ),
    )


def _make_estimator(router_config: dict, tmp_path):
    """Construct an ArtOfPIVEstimator with sub-estimator creation stubbed."""
    yaml_path = _write_dummy_estimators_yaml(tmp_path)
    return ArtOfPIVEstimator(
        router_config=router_config,
        estimators_list_path=yaml_path,
        training_mode="bandits",
    )


def test_nested_router_config_builds_router_model(
    stub_create_estimators, tmp_path
):
    """Nested encoder/head config builds a CNNRouterModel."""
    estimator = _make_estimator(
        router_config={
            "encoder": {
                "features_list": [32, 64, 64],
                "output_dim": 64,
                "use_residual": False,
                "norm_fn": "none",
            },
            "head": {
                "hidden_dims": [512],
                "activation": "relu",
            },
        },
        tmp_path=tmp_path,
    )
    assert isinstance(estimator.model, CNNRouterModel)
    assert estimator.model.encoder.output_dim == 64
    assert estimator.model.head_hidden_dims == (512,)


def test_flat_router_config_still_builds_q_estimator_model(
    stub_create_estimators, tmp_path
):
    """Legacy flat router config still produces a CNNQEstimatorModel."""
    estimator = _make_estimator(
        router_config={
            "features": [16],
            "kernel_sizes": [3],
            "strides": [1],
        },
        tmp_path=tmp_path,
    )
    assert isinstance(estimator.model, CNNQEstimatorModel)
