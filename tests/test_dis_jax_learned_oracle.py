"""Tests for DIS + learned-oracle postprocessing integration hooks."""

import jax.numpy as jnp
from flax.core import FrozenDict

from flowgym.common.base import EstimatorTrainableState
from flowgym.flow.base import OUTLIER_REJECTION_STEPS
from flowgym.flow.dis.dis_jax import DISJAXFlowFieldEstimator


def test_learned_oracle_is_tracked_as_outlier_rejection_step():
    assert "learned_oracle_threshold" in OUTLIER_REJECTION_STEPS


def test_dis_estimator_disables_jit_for_learned_oracle_postprocess():
    estimator = DISJAXFlowFieldEstimator(
        patch_size=11,
        patch_stride=2,
        grad_desc_iters=4,
        start_level=0,
        levels=3,
        use_temporal_propagation=False,
        postprocessing_steps=[
            {
                "name": "learned_oracle_threshold",
                "threshold_value": 0.5,
                "load_from": (
                    "results/training_oracle/tau_0_5/"
                    "learned_oracle_threshold/0"
                ),
                "features": [16, 32],
            }
        ],
    )

    assert estimator.supports_jit() is False


def test_dis_estimator_supports_jit_with_preloaded_learned_oracle_state():
    estimator = DISJAXFlowFieldEstimator(
        patch_size=11,
        patch_stride=2,
        grad_desc_iters=4,
        start_level=0,
        levels=3,
        use_temporal_propagation=False,
        postprocessing_steps=[
            {
                "name": "learned_oracle_threshold",
                "threshold_value": 0.5,
                "oracle_trainable_state": EstimatorTrainableState(
                    apply_fn=lambda variables, x, **kwargs: x[..., 0],
                    params=FrozenDict(
                        {
                            "Conv_0": {
                                "kernel": jnp.zeros(
                                    (3, 3, 3, 8), dtype=jnp.float32
                                )
                            }
                        }
                    ),
                ),
                "features": [16, 32],
            }
        ],
    )

    assert estimator.supports_jit() is True
