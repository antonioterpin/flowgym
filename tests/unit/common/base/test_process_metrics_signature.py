"""Every registered estimator accepts the ``process_metrics`` call of eval."""

import inspect

import pytest

from flowgym import ALL_ESTIMATORS


@pytest.mark.parametrize(
    "estimator_cls",
    [c for c in ALL_ESTIMATORS.values() if inspect.isclass(c)],
    ids=lambda c: c.__name__,
)
def test_process_metrics_accepts_eval_kwargs(estimator_cls):
    """``eval_flow`` passes ``flow_field`` and ``flow_field_gt`` by keyword."""
    inspect.signature(estimator_cls.process_metrics).bind(
        None, {}, flow_field=None, flow_field_gt=None
    )
