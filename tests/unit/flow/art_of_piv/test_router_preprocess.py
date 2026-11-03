"""Tests for the ART of PIV router_preprocess option."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest
import yaml

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator

ROUTER = {
    "encoder": {"features_list": [4]},
    "head": {"hidden_dims": [4]},
}
CROP = [
    {
        "name": "crop_special",
        "target_height": 4,
        "target_width": 4,
        "fraction_v": 0.5,
        "fraction_h": 0.5,
    }
]


@pytest.fixture
def stub_create_estimators(monkeypatch):
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


def _make(tmp_path, **kwargs):
    path = tmp_path / "estimators.yaml"
    path.write_text(
        yaml.safe_dump({"estimators": [{"estimator": "dummy", "config": {}}]}),
        encoding="utf-8",
    )
    return ArtOfPIVEstimator(
        router_config=ROUTER, estimators_list_path=str(path), **kwargs
    )


def _record_router_input(estimator):
    seen = {}

    def apply(variables, x):
        seen["shape"] = x.shape
        return jnp.zeros((x.shape[0], 1))

    estimator.model = SimpleNamespace(apply=apply)
    return seen


@pytest.mark.parametrize(
    ("router_preprocess", "expected"),
    [(None, (2, 8, 8, 2)), (CROP, (2, 4, 4, 2))],
)
def test_router_input_follows_router_preprocess(
    stub_create_estimators, tmp_path, router_preprocess, expected
):
    estimator = _make(tmp_path, router_preprocess=router_preprocess)
    seen = _record_router_input(estimator)
    frames = jnp.zeros((2, 8, 8))

    estimator.q_estimate(frames, frames, SimpleNamespace(params={}))

    assert seen["shape"] == expected


@pytest.mark.parametrize(
    ("router_preprocess", "expected"),
    [
        (None, True),
        (CROP, True),
        ([{"name": "clahe", "clip_limit": 2.0}], False),
    ],
)
def test_supports_jit_excludes_host_side_router_steps(
    stub_create_estimators, tmp_path, router_preprocess, expected
):
    estimator = _make(tmp_path, router_preprocess=router_preprocess)

    assert estimator.supports_jit() is expected


def test_trainable_state_is_built_for_the_router_input(
    stub_create_estimators, tmp_path
):
    estimator = _make(tmp_path, router_preprocess=CROP)
    shapes = []
    estimator.model = SimpleNamespace(
        apply=lambda *a: None,
        init=lambda key, x: shapes.append(x.shape) or {"params": {}},
    )

    estimator.create_trainable_state(jnp.zeros((1, 8, 8)), key=None)

    assert shapes == [(1, 4, 4, 2)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"router_preprocess": [{"crop": 1}]},
        {
            "router_preprocess": [
                {"name": "resize_image", "target_height": -1, "target_width": 4}
            ]
        },
        {"router_preprocess": CROP, "use_last_estimate": True},
    ],
)
def test_invalid_router_preprocess_is_rejected(
    stub_create_estimators, tmp_path, kwargs
):
    with pytest.raises(ValueError):
        _make(tmp_path, **kwargs)
