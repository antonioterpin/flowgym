"""Tests for learned oracle threshold estimator and utility."""

import jax
import jax.numpy as jnp
import pytest
from flax.core import FrozenDict

from flowgym import ALL_ESTIMATORS
from flowgym.common.base import EstimatorTrainableState
from flowgym.flow.postprocess import validate_params
from flowgym.flow.postprocess.data_validation import learned_oracle_threshold
from flowgym.flow.postprocess.oracle_threshold import (
    LearnedOracleThresholdEstimator,
)
from flowgym.types import CachePayload, SupervisedExperience


def _all_params_close(params_a, params_b) -> bool:
    close_tree = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y), params_a, params_b
    )
    return all(bool(v) for v in jax.tree_util.tree_leaves(close_tree))


def test_learned_oracle_threshold_uses_sigmoid_threshold():
    flow = jnp.array(
        [
            [
                [[1.0, 0.0], [-1.0, 0.0]],
                [[0.2, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    valid = jnp.array([[[True, True], [False, True]]], dtype=jnp.bool_)

    trainable_state = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: x[..., 0],
        params=FrozenDict(),
    )

    _, mask, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=trainable_state,
        valid=valid,
        threshold_value=0.5,
    )

    assert mask is not None
    expected = valid & (flow[..., 0] > 0.0)
    assert jnp.array_equal(mask, expected)


def test_learned_oracle_train_step_updates_parameters():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
    )
    key = jax.random.PRNGKey(0)
    images = jnp.zeros((2, 8, 8), dtype=jnp.float32)
    trainable_state = estimator.create_trainable_state(images, key)

    flow_gt = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    flow_pred = flow_gt.at[0, 1, 1, 0].set(2.0)
    flow_pred = flow_pred.at[1, 3, 5, 1].set(-2.0)

    state = estimator.create_state(
        images=images,
        estimates=flow_pred,
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    experience = SupervisedExperience(
        state=state,
        obs=(images, images),
        ground_truth=flow_gt,
        cache_payload=CachePayload(estimates=flow_pred),
    )

    train_step = estimator.create_train_step()
    loss, new_state, metrics = train_step(trainable_state, experience)

    assert jnp.isfinite(loss)
    assert int(new_state.step) == int(trainable_state.step) + 1
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert "mask_accuracy" in metrics
    assert 0.0 <= float(metrics["mask_accuracy"]) <= 1.0
    assert not _all_params_close(trainable_state.params, new_state.params)


def test_learned_oracle_estimator_call_uses_cached_flow():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    key = jax.random.PRNGKey(1)
    images = jnp.zeros((2, 6, 6), dtype=jnp.float32)
    trainable_state = estimator.create_trainable_state(images, key)

    initial_estimates = jnp.zeros((2, 6, 6, 2), dtype=jnp.float32)
    cached_flow = initial_estimates.at[:, 2, 2, 0].set(1.0)

    state = estimator.create_state(
        images=images,
        estimates=initial_estimates,
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )

    new_state, metrics = estimator(
        images,
        state,
        trainable_state,
        cache_payload=CachePayload(estimates=cached_flow),
    )

    assert "mask" in metrics
    mask = jnp.asarray(metrics["mask"])
    assert mask.shape == (2, 6, 6)
    assert mask.dtype == jnp.bool_
    assert jnp.allclose(new_state["estimates"][:, -1, ...], cached_flow)


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.0, 1.2])
def test_learned_oracle_estimator_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        LearnedOracleThresholdEstimator(threshold=threshold)


@pytest.mark.parametrize("oracle_epe_threshold", [0.0, -0.1])
def test_learned_oracle_estimator_rejects_invalid_oracle_epe_threshold(
    oracle_epe_threshold: float,
):
    with pytest.raises(ValueError, match="oracle_epe_threshold"):
        LearnedOracleThresholdEstimator(
            threshold=0.5,
            oracle_epe_threshold=oracle_epe_threshold,
        )


def test_learned_oracle_estimator_rejects_invalid_features():
    with pytest.raises(ValueError, match="non-empty"):
        LearnedOracleThresholdEstimator(threshold=0.5, features=[])
    with pytest.raises(ValueError, match="positive integers"):
        LearnedOracleThresholdEstimator(threshold=0.5, features=[8, 0])


def test_learned_oracle_create_trainable_state_rejects_bad_input_shape():
    estimator = LearnedOracleThresholdEstimator(threshold=0.5)
    with pytest.raises(ValueError, match="dummy_input"):
        estimator.create_trainable_state(
            jnp.zeros((2, 8), dtype=jnp.float32),
            jax.random.PRNGKey(0),
        )


def test_learned_oracle_threshold_rejects_missing_apply_fn():
    flow = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    ts = EstimatorTrainableState(
        apply_fn=None,
        params=FrozenDict(),
    )
    with pytest.raises(ValueError, match="apply_fn"):
        learned_oracle_threshold(
            flow_field=flow,
            trainable_state=ts,
            threshold_value=0.5,
        )


def test_learned_oracle_threshold_supports_legacy_apply_signature():
    flow = jnp.array([[[[1.0, 0.0], [-1.0, 0.0]]]], dtype=jnp.float32)

    def legacy_apply(x, params, **kwargs):
        del params, kwargs
        return x[..., 0]

    ts = EstimatorTrainableState(
        apply_fn=legacy_apply,
        params=FrozenDict(),
    )
    _, mask, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=ts,
        threshold_value=0.5,
    )
    assert mask is not None
    assert bool(mask[0, 0, 0])
    assert not bool(mask[0, 0, 1])


def test_learned_oracle_threshold_rejects_bad_logits_shape():
    flow = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    ts = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: jnp.zeros((1, 4)),
        params=FrozenDict(),
    )
    with pytest.raises(ValueError, match="logits"):
        learned_oracle_threshold(
            flow_field=flow,
            trainable_state=ts,
            threshold_value=0.5,
        )


def test_learned_oracle_validate_params_registered():
    validate_params("learned_oracle_threshold", threshold_value=0.5)
    with pytest.raises(ValueError, match="threshold_value"):
        validate_params("learned_oracle_threshold", threshold_value=1.0)


def test_learned_oracle_train_step_rejects_gt_shape_mismatch():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    images = jnp.zeros((2, 8, 8), dtype=jnp.float32)
    trainable_state = estimator.create_trainable_state(
        images, jax.random.PRNGKey(0)
    )
    flow_pred = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    state = estimator.create_state(
        images=images,
        estimates=flow_pred,
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    experience = SupervisedExperience(
        state=state,
        obs=(images, images),
        ground_truth=jnp.zeros((2, 8, 8), dtype=jnp.float32),
        cache_payload=CachePayload(estimates=flow_pred),
    )
    train_step = estimator.create_train_step()
    with pytest.raises(ValueError, match="shape mismatch"):
        train_step(trainable_state, experience)


def test_learned_oracle_estimator_call_without_cache_uses_state_estimate():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    images = jnp.zeros((1, 5, 5), dtype=jnp.float32)
    trainable_state = estimator.create_trainable_state(
        images, jax.random.PRNGKey(0)
    )
    state_flow = jnp.zeros((1, 5, 5, 2), dtype=jnp.float32).at[
        0, 1, 3, 0
    ].set(1.5)
    state = estimator.create_state(
        images=images,
        estimates=state_flow,
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    new_state, _ = estimator(images, state, trainable_state)
    assert jnp.allclose(new_state["estimates"][:, -1, ...], state_flow)


def test_learned_oracle_estimator_honors_valid_mask_from_cache_extras():
    estimator = LearnedOracleThresholdEstimator(threshold=0.5)
    images = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    state = estimator.create_state(
        images=images,
        estimates=jnp.zeros((1, 4, 4, 2), dtype=jnp.float32),
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    keep_all_state = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: jnp.full(
            x.shape[:-1], 10.0, dtype=jnp.float32
        ),
        params=FrozenDict(),
    )
    valid = jnp.ones((1, 4, 4), dtype=jnp.bool_).at[0, 0, 0].set(False)
    payload = CachePayload(
        estimates=jnp.zeros((1, 4, 4, 2), dtype=jnp.float32),
        extras={"valid": valid},
    )
    _, metrics = estimator(images, state, keep_all_state, cache_payload=payload)
    mask = jnp.asarray(metrics["mask"])
    assert mask.dtype == jnp.bool_
    assert not bool(mask[0, 0, 0])
    assert bool(mask[0, 1, 1])


def test_learned_oracle_estimator_registered():
    assert "learned_oracle_threshold" in ALL_ESTIMATORS
