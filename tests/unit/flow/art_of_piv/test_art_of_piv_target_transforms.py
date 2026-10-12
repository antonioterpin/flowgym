"""Integration tests for ART of PIV target transformations."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import yaml
from flax.core.frozen_dict import FrozenDict

from flowgym.flow.art_of_piv.estimator import ArtOfPIVEstimator
from flowgym.types import SupervisedExperience


def _build_stubbed_estimator(
    tmp_yaml_path: str,
    *,
    training_mode: str,
    target_transform_config: dict,
):
    ArtOfPIVEstimator._create_estimators = MagicMock(
        return_value=(
            (lambda x: (None, {"epe": 1.0}),),
            (MagicMock(),),
            (MagicMock(),),
        )
    )
    return ArtOfPIVEstimator(
        router_config={
            "features": [16],
            "kernel_sizes": [3],
            "strides": [1],
        },
        estimators_list_path=tmp_yaml_path,
        training_mode=training_mode,
        target_transform_config=target_transform_config,
    )


def _make_trainable_state(batch_size: int):
    import optax

    from flowgym.common.base import NNEstimatorTrainableState

    return NNEstimatorTrainableState.create(
        apply_fn=lambda x: x,
        params=FrozenDict({}),
        tx=optax.identity(),
        extras=FrozenDict(
            {"steps": jnp.zeros((batch_size, 1), dtype=jnp.int32)}
        ),
    )


def _mock_run_router_outputs(batch_size: int, num_estimators: int):
    """Build deterministic router outputs for supervised train-step tests."""
    return (
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size, num_estimators)),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jax.random.split(jax.random.PRNGKey(0), batch_size),
    )


def test_art_of_piv_target_transform_regression(tmp_path):
    temp_yaml = tmp_path / "test_estimators.yaml"
    temp_yaml.write_text(
        yaml.safe_dump({"estimators": [{"estimator": "dummy", "config": {}}]}),
        encoding="utf-8",
    )

    estimator = _build_stubbed_estimator(
        str(temp_yaml),
        training_mode="regression",
        target_transform_config={"name": "scale", "factor": 10.0},
    )

    # Create dummy experience with cached EPEs
    B = 2
    K = 1
    epe_all = jnp.array([[1.0], [2.0]])  # (B, K)

    experience = SupervisedExperience(
        obs=(jnp.zeros((B, 32, 32, 3)), jnp.zeros((B, 32, 32, 3))),
        state={"keys": jax.random.split(jax.random.PRNGKey(0), B)},
        ground_truth=jnp.zeros((B, 32, 32, 2)),
        cache_payload={"epe_all": epe_all},
    )

    # Mock q_estimate
    estimator.q_estimate = MagicMock(return_value=jnp.zeros((B, K)))

    # Keep router outputs deterministic so this test isolates target transform
    # behavior inside the supervised training step.
    estimator._run_router = MagicMock(
        return_value=_mock_run_router_outputs(B, K)
    )

    # Create a supervised train step
    train_step_fn = estimator.create_supervised_train_step()

    trainable_state = _make_trainable_state(B)

    expected_target_mean = jnp.mean(10.0 * epe_all)

    _, _, metrics = train_step_fn(trainable_state, experience)

    assert jnp.allclose(metrics["epe/raw_mean"], jnp.mean(epe_all)), (
        "Regression mode should preserve the raw EPE mean in metrics."
    )
    assert jnp.allclose(metrics["target/mean"], expected_target_mean), (
        "Regression mode should apply target_transform_config to EPE targets."
    )


def test_art_of_piv_target_transform_bandits(tmp_path):
    temp_yaml = tmp_path / "test_estimators_bandits.yaml"
    temp_yaml.write_text(
        yaml.safe_dump({"estimators": [{"estimator": "dummy", "config": {}}]}),
        encoding="utf-8",
    )

    estimator = _build_stubbed_estimator(
        str(temp_yaml),
        training_mode="bandits",
        target_transform_config={"name": "scale", "factor": 5.0},
    )

    # Create dummy experience with cached EPEs
    B = 1
    K = 1
    epe_all = jnp.array([[2.0]])  # (B, K)

    experience = SupervisedExperience(
        obs=(jnp.zeros((B, 32, 32, 3)), jnp.zeros((B, 32, 32, 3))),
        state={"keys": jax.random.split(jax.random.PRNGKey(0), B)},
        ground_truth=jnp.zeros((B, 32, 32, 2)),
        cache_payload={"epe_all": epe_all},
    )

    # Mock choice is always 0
    estimator.explore = MagicMock(return_value=jnp.array([0]))
    estimator.q_estimate = MagicMock(return_value=jnp.zeros((B, K)))

    estimator._run_router = MagicMock(
        return_value=_mock_run_router_outputs(B, K)
    )
    train_step_fn = estimator.create_supervised_train_step()

    trainable_state = _make_trainable_state(B)
    # Capture reward in metrics
    _, _, metrics = train_step_fn(trainable_state, experience)

    # Raw EPE is 2.0. Target EPE is 5.0 * 2.0 = 10.0. Reward is -10.0.
    assert jnp.allclose(metrics["epe/raw_mean"], 2.0), (
        "Bandits mode should report the untransformed raw EPE mean."
    )
    assert jnp.allclose(metrics["target/mean"], 10.0), (
        "Bandits mode should apply the scale transform to target values."
    )
