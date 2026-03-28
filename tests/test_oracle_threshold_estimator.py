"""Tests for learned oracle threshold estimator and utility."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict

from eval import evaluate_batches
from flowgym import ALL_ESTIMATORS
from flowgym.common.base import EstimatorTrainableState
from flowgym.flow.postprocess import validate_params
from flowgym.flow.postprocess.data_validation import learned_oracle_threshold
from flowgym.flow.postprocess.oracle_threshold import (
    LearnedOracleThresholdEstimator,
)
from flowgym.types import CachePayload, SupervisedExperience


def _all_params_close(params_a, params_b) -> bool:
    close_tree = jax.tree_util.tree_map(jnp.allclose, params_a, params_b)
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


def test_learned_oracle_threshold_supports_k_subestimators_shape():
    flow = jnp.array(
        [
            [
                [
                    [[1.0, 0.0], [-1.0, 0.0]],
                    [[0.2, 0.0], [0.0, 0.0]],
                ],
                [
                    [[-1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 0.0], [0.3, 0.0]],
                ],
            ]
        ],
        dtype=jnp.float32,
    )  # (B=1, K=2, H=2, W=2, C=2)
    valid = jnp.array(
        [[[[True, True], [False, True]], [[True, True], [True, False]]]],
        dtype=jnp.bool_,
    )

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
    assert mask.shape == (1, 2, 2, 2)
    expected = valid & (flow[..., 0] > 0.0)
    assert jnp.array_equal(mask, expected)


def test_learned_oracle_threshold_uses_estimator_index_channel_for_k():
    flow = jnp.zeros((1, 3, 2, 2, 2), dtype=jnp.float32)
    valid = jnp.ones((1, 3, 2, 2), dtype=jnp.bool_)

    trainable_state = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: x[..., 2] - 0.75,
        params=FrozenDict(),
    )

    _, mask, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=trainable_state,
        valid=valid,
        threshold_value=0.5,
    )

    assert mask is not None
    expected = jnp.array(
        [
            [
                [[False, False], [False, False]],
                [[False, False], [False, False]],
                [[True, True], [True, True]],
            ]
        ],
        dtype=jnp.bool_,
    )
    assert jnp.array_equal(mask, expected)


def test_learned_oracle_threshold_include_image_pair_uses_images():
    flow = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    prev = jnp.ones((1, 2, 2), dtype=jnp.float32)
    curr = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    valid = jnp.ones((1, 2, 2), dtype=jnp.bool_)

    trainable_state = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: x[..., 3] - x[..., 4],
        params=FrozenDict(),
    )

    _, mask, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=trainable_state,
        valid=valid,
        threshold_value=0.5,
        include_image_pair=True,
        previous_image=prev,
        current_image=curr,
    )

    assert mask is not None
    assert jnp.all(mask)


def test_learned_oracle_threshold_include_image_pair_requires_images():
    flow = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    trainable_state = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: x[..., 0],
        params=FrozenDict(),
    )
    with pytest.raises(ValueError, match="previous_image"):
        learned_oracle_threshold(
            flow_field=flow,
            trainable_state=trainable_state,
            threshold_value=0.5,
            include_image_pair=True,
        )


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
    for metric_name in (
        "mask_accuracy",
        "mask_precision",
        "mask_recall",
        "mask_specificity",
        "mask_balanced_accuracy",
        "mask_f1",
        "mask_iou",
        "pred_inlier_fraction",
        "oracle_inlier_fraction",
    ):
        assert metric_name in metrics
        metric_value = float(metrics[metric_name])
        assert np.isfinite(metric_value)
        assert 0.0 <= metric_value <= 1.0
    assert not _all_params_close(trainable_state.params, new_state.params)


def test_learned_oracle_train_step_supports_k_flow_candidates():
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
    flow_pred = jnp.zeros((2, 3, 8, 8, 2), dtype=jnp.float32)
    flow_pred = flow_pred.at[0, 0, 1, 1, 0].set(2.0)
    flow_pred = flow_pred.at[1, 2, 3, 5, 1].set(-2.0)

    state = estimator.create_state(
        images=images,
        estimates=jnp.zeros((2, 8, 8, 2), dtype=jnp.float32),
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
    assert "mask_f1" in metrics
    assert 0.0 <= float(metrics["mask_f1"]) <= 1.0


def test_learned_oracle_train_step_supports_image_pair_input():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
        include_image_pair=True,
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
    )
    key = jax.random.PRNGKey(0)
    images_prev = jnp.zeros((2, 8, 8), dtype=jnp.float32)
    images_curr = jnp.ones((2, 8, 8), dtype=jnp.float32)
    trainable_state = estimator.create_trainable_state(images_prev, key)

    flow_gt = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    flow_pred = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    state = estimator.create_state(
        images=images_prev,
        estimates=flow_pred,
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    experience = SupervisedExperience(
        state=state,
        obs=(images_prev, images_curr),
        ground_truth=flow_gt,
        cache_payload=CachePayload(estimates=flow_pred),
    )

    train_step = estimator.create_train_step()
    loss, new_state, metrics = train_step(trainable_state, experience)

    assert jnp.isfinite(loss)
    assert int(new_state.step) == int(trainable_state.step) + 1
    assert "mask_f1" in metrics
    assert 0.0 <= float(metrics["mask_f1"]) <= 1.0


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


def test_learned_oracle_estimator_empty_cache_payload_uses_k_path():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    estimator.num_sub_estimators = 2

    def _fake_sub_flows(self, images, state):
        b, h, w = images.shape
        del state
        return jnp.ones((b, 2, h, w, 2), dtype=jnp.float32)

    estimator._compute_sub_estimator_flows = types.MethodType(
        _fake_sub_flows, estimator
    )

    key = jax.random.PRNGKey(1)
    images = jnp.zeros((2, 6, 6), dtype=jnp.float32)
    trainable_state = estimator.create_trainable_state(images, key)
    initial_estimates = jnp.zeros((2, 6, 6, 2), dtype=jnp.float32)

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
        cache_payload=CachePayload(),
    )

    assert "mask" in metrics
    assert "mask_flow_fields" in metrics
    mask = jnp.asarray(metrics["mask"])
    assert mask.shape == (2, 2, 6, 6)
    assert jnp.asarray(metrics["mask_flow_fields"]).shape == (2, 2, 6, 6, 2)
    assert jnp.allclose(jnp.asarray(metrics["mask_flow_fields"]), 1.0)
    assert jnp.allclose(new_state["estimates"][:, -1, ...], 1.0)


def test_learned_oracle_prepare_experience_precomputes_non_jittable_flows():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    estimator.num_sub_estimators = 2
    estimator._inner_estimators_support_jit = False

    def _fake_sub_flows(self, images, state):
        b, h, w = images.shape
        del state
        return jnp.ones((b, 2, h, w, 2), dtype=jnp.float32)

    estimator._compute_sub_estimator_flows = types.MethodType(
        _fake_sub_flows, estimator
    )

    images = jnp.zeros((2, 6, 6), dtype=jnp.float32)
    state = estimator.create_state(
        images=images,
        estimates=jnp.zeros((2, 6, 6, 2), dtype=jnp.float32),
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    experience = SupervisedExperience(
        state=state,
        obs=(images, images),
        ground_truth=jnp.zeros((2, 6, 6, 2), dtype=jnp.float32),
        cache_payload=None,
    )

    prepared = estimator.prepare_experience_for_training(
        experience=experience,
        trainable_state=estimator.create_trainable_state(
            images, jax.random.PRNGKey(0)
        ),
    )

    assert prepared.cache_payload is not None
    assert prepared.cache_payload.estimates is not None
    assert prepared.cache_payload.estimates.shape == (2, 2, 6, 6, 2)
    assert jnp.allclose(prepared.cache_payload.estimates, 1.0)


def test_learned_oracle_supports_jitted_train_step_with_non_jittable_inner():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    estimator._inner_estimators_support_jit = False

    assert estimator.supports_jit() is False
    assert estimator.supports_train_step_jit() is True


def test_learned_oracle_train_step_requires_precompute_for_non_jittable_inner():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    estimator.num_sub_estimators = 1
    estimator._inner_estimators_support_jit = False

    images = jnp.zeros((2, 8, 8), dtype=jnp.float32)
    flow_gt = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    state = estimator.create_state(
        images=images,
        estimates=jnp.zeros((2, 8, 8, 2), dtype=jnp.float32),
        image_history_size=1,
        estimate_history_size=1,
        rng=0,
    )
    experience = SupervisedExperience(
        state=state,
        obs=(images, images),
        ground_truth=flow_gt,
        cache_payload=CachePayload(),
    )
    train_step = estimator.create_train_step()
    trainable_state = estimator.create_trainable_state(
        images, jax.random.PRNGKey(0)
    )

    with pytest.raises(ValueError, match="prepare_experience_for_training"):
        train_step(trainable_state, experience)


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
    validate_params(
        "learned_oracle_threshold",
        threshold_value=0.5,
        estimator_index=1,
        estimator_count=3,
    )
    with pytest.raises(ValueError, match="estimator_count"):
        validate_params(
            "learned_oracle_threshold",
            threshold_value=0.5,
            estimator_count=0,
        )


def test_learned_oracle_threshold_uses_estimator_index_and_count():
    flow = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    ts = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: x[..., 2],
        params=FrozenDict(),
    )

    _, mask_idx0, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=ts,
        threshold_value=0.5,
        estimator_index=0,
        estimator_count=3,
    )
    _, mask_idx1, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=ts,
        threshold_value=0.5,
        estimator_index=1,
        estimator_count=3,
    )

    assert mask_idx0 is not None
    assert mask_idx1 is not None
    # idx=0 => normalized channel 0.0 => sigmoid(0)=0.5 => rejected (strict >)
    assert not bool(mask_idx0[0, 0, 0])
    # idx=1 => normalized channel 0.5 => sigmoid(0.5)>0.5 => kept
    assert bool(mask_idx1[0, 0, 0])


def test_learned_oracle_threshold_supports_legacy_flow_only_params():
    flow = jnp.array([[[[1.0, 0.0], [-1.0, 0.0]]]], dtype=jnp.float32)
    seen_input_channels: dict[str, int] = {}

    def apply_fn(variables, x, **kwargs):
        del variables, kwargs
        seen_input_channels["value"] = int(x.shape[-1])
        return x[..., 0]

    ts = EstimatorTrainableState(
        apply_fn=apply_fn,
        params=FrozenDict(
            {
                "Conv_0": {
                    "kernel": jnp.zeros((3, 3, 2, 8), dtype=jnp.float32)
                }
            }
        ),
    )
    _, mask, _ = learned_oracle_threshold(
        flow_field=flow,
        trainable_state=ts,
        threshold_value=0.5,
        estimator_index=2,
        estimator_count=3,
    )

    assert mask is not None
    assert seen_input_channels["value"] == 2
    assert jnp.array_equal(mask, flow[..., 0] > 0.0)


def test_learned_oracle_threshold_rejects_invalid_inferred_channels():
    flow = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    ts = EstimatorTrainableState(
        apply_fn=lambda variables, x, **kwargs: x[..., 0],
        params=FrozenDict(
            {
                "Conv_0": {
                    "kernel": jnp.zeros((3, 3, 4, 8), dtype=jnp.float32)
                }
            }
        ),
    )

    with pytest.raises(ValueError, match="input channels"):
        learned_oracle_threshold(
            flow_field=flow,
            trainable_state=ts,
            threshold_value=0.5,
        )


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


def test_learned_oracle_validation_metrics_include_classification_scores():
    estimator = LearnedOracleThresholdEstimator(
        threshold=0.5,
        oracle_epe_threshold=0.25,
        features=(8, 8),
    )
    images = jnp.zeros((2, 8, 8), dtype=jnp.float32)
    flow_gt = jnp.zeros((2, 8, 8, 2), dtype=jnp.float32)
    flow_pred = flow_gt.at[0, 1, 1, 0].set(2.0)
    flow_pred = flow_pred.at[1, 3, 5, 1].set(-2.0)

    class _OneBatchSampler:
        def __init__(self):
            self.batch = type(
                "Batch",
                (),
                {
                    "images1": images,
                    "images2": images,
                    "flow_fields": flow_gt,
                    "mask": None,
                },
            )()

        def __iter__(self):
            yield self.batch

        def reset(self):
            return None

    trainable_state = estimator.create_trainable_state(
        images, jax.random.PRNGKey(0)
    )

    def create_state_fn(img, key):
        del key
        return estimator.create_state(
            images=img,
            estimates=flow_pred,
            image_history_size=1,
            estimate_history_size=1,
            rng=0,
        )

    def compute_estimate_fn(img2, state, trained_state, cache_payload=None):
        del cache_payload
        return estimator(
            img2,
            state,
            trained_state,
            cache_payload=CachePayload(estimates=flow_pred),
        )

    results = evaluate_batches(
        model=estimator,
        sampler=_OneBatchSampler(),
        create_state_fn=create_state_fn,
        compute_estimate_fn=compute_estimate_fn,
        trainable_state=trainable_state,
        estimate_type="flow",
        num_batches=1,
    )

    for metric_name in (
        "mask_accuracy",
        "mask_precision",
        "mask_recall",
        "mask_specificity",
        "mask_balanced_accuracy",
        "mask_f1",
        "mask_iou",
        "pred_inlier_fraction",
        "oracle_inlier_fraction",
    ):
        assert metric_name in results
        metric_value = float(results[metric_name])
        assert np.isfinite(metric_value)
        assert 0.0 <= metric_value <= 1.0
