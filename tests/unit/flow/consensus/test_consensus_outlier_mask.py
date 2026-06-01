"""Tests for consensus weight masking based on rejected outlier pixels."""

import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.consensus.consensus import ConsensusFlowEstimator


def _new_consensus_with_n_estimators(
    n_estimators: int,
) -> ConsensusFlowEstimator:
    model = ConsensusFlowEstimator.__new__(ConsensusFlowEstimator)
    model.num_estimators = n_estimators
    return model


def test_build_weight_mask_none_when_no_rejection_metrics():
    model = _new_consensus_with_n_estimators(3)
    metrics_per_estimator = [{}, {}, {}]

    keep_mask, rejected_mask = model._build_weight_mask_from_estimator_metrics(
        metrics_per_estimator,
        (2, 3, 4, 5, 2),
    )

    assert keep_mask is None
    assert rejected_mask is None


def test_build_weight_mask_from_rejected_metrics():
    model = _new_consensus_with_n_estimators(3)

    rej0 = jnp.zeros((2, 4, 5), dtype=jnp.bool_).at[:, 1, 2].set(True)
    rej1 = jnp.zeros((2, 4, 5), dtype=jnp.bool_).at[:, 0, 0].set(True)
    metrics_per_estimator = [
        {"postprocess_combined_rejected_mask": rej0},
        {"postprocess_combined_rejected_mask": rej1},
        {},  # Missing mask means "no rejected pixels" for that estimator.
    ]

    keep_mask, rejected_mask = model._build_weight_mask_from_estimator_metrics(
        metrics_per_estimator,
        (2, 3, 4, 5, 2),
    )

    assert keep_mask is not None
    assert rejected_mask is not None
    assert keep_mask.shape == (2, 3, 4, 5)
    assert rejected_mask.shape == (2, 3, 4, 5)
    assert bool(rejected_mask[0, 0, 1, 2])
    assert bool(rejected_mask[0, 1, 0, 0])
    assert not bool(rejected_mask[0, 2, 1, 2])
    assert not bool(keep_mask[0, 0, 1, 2])
    assert bool(keep_mask[0, 0, 1, 1])


def test_build_weight_mask_raises_on_shape_mismatch():
    model = _new_consensus_with_n_estimators(2)
    bad_mask = jnp.zeros((2, 4, 4), dtype=jnp.bool_)
    metrics_per_estimator = [
        {"postprocess_combined_rejected_mask": bad_mask},
        {},
    ]

    with pytest.raises(ValueError, match="postprocess_combined_rejected_mask"):
        model._build_weight_mask_from_estimator_metrics(
            metrics_per_estimator,
            (2, 2, 5, 5, 2),
        )


def test_estimator_rejected_percentage_excludes_per_step_keys():
    """Only the combined per-estimator rejection key feeds the aggregate.

    ``process_metrics`` aggregates the canonical
    ``estimator_{i}_postprocess_rejected_percentage`` key (set in
    ``_estimate`` from the combined reject mask) and must ignore finer-grained
    per-step keys ``estimator_{i}_postprocess_{name}_{j}_rejected_percentage``,
    which also end in ``_rejected_percentage`` but must not be blended into the
    same per-estimator bucket.
    """
    model = _new_consensus_with_n_estimators(1)
    model.experiment_params = {}

    metrics = {
        # Combined per-estimator rejection: mean over the batch is 15.0.
        "estimator_0_postprocess_rejected_percentage": np.array([10.0, 20.0]),
        # Per-step rejection from an inner validation step: must be ignored
        # (blending it in would skew the mean toward 52.5).
        "estimator_0_postprocess_universal_median_test_0_rejected_percentage": (
            np.array([90.0, 90.0])
        ),
    }

    model.process_metrics(dict(metrics))
    aggregated = model._get_estimator_rejected_percentages()

    assert aggregated == {
        "estimator_0_mean_rejected_percentage": pytest.approx(15.0)
    }
