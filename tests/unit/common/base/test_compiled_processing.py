"""Standalone processing must match the estimator's configured pipeline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym import make
from flowgym.common import preprocess as preprocess_module
from flowgym.density.simple import SimpleDensityEstimator
from flowgym.flow import postprocess as postprocess_module
from flowgym.flow.dummy import DummyEstimator

RESIZE_IMAGE = {"name": "resize_image", "target_height": 4, "target_width": 6}
SMOOTHING = {"name": "temporal_smoothing_ema", "alpha": 0.5}


def _is_jitted(fn) -> bool:
    return hasattr(fn, "lower")


@pytest.mark.parametrize("jit", [False, True])
def test_compiled_preprocess_matches_initialization_and_estimation(jit):
    """Export preprocessing without running the estimator or storing extras."""
    model = DummyEstimator(preprocessing_steps=[RESIZE_IMAGE])
    compiled = make.compile_model(model, jnp.zeros((2, 4, 6, 2)), jit=jit)
    images = jnp.arange(128, dtype=jnp.float32).reshape(2, 8, 8)
    assert compiled.create_state is not None, "Estimates were provided"
    state = compiled.create_state(images, jax.random.PRNGKey(0))
    processed = compiled.preprocess(images)
    np.testing.assert_allclose(processed, state["images"][:, -1])
    trained = model.create_trainable_state(images, jax.random.PRNGKey(0))
    updated, _ = compiled.compute_estimate(images, state, trained)
    np.testing.assert_allclose(processed, updated["images"][:, -1])


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "steps",
    [
        [],
        [SMOOTHING],
        [{"name": "constant_threshold_filter", "vel_min": 1.0, "vel_max": 2.0}],
        [
            {
                "name": "resize_flow",
                "target_height": 4,
                "target_width": 6,
            }
        ],
    ],
)
def test_compiled_postprocess_matches_full_estimator(jit, steps):
    """Standalone postprocessing preserves the flow and diagnostic metrics."""
    model = DummyEstimator(postprocessing_steps=steps)
    compiled = make.compile_model(model, None, jit=jit)
    assert compiled.postprocess is not None, "Flow estimators postprocess"
    images = jnp.ones((2, 8, 8))
    resize = any(step["name"] == "resize_flow" for step in steps)
    shape = (2, 4, 6, 2) if resize else (2, 8, 8, 2)
    state = model.create_state(
        images,
        jnp.ones(shape),
        image_history_size=1,
        rng=jax.random.PRNGKey(0),
    )
    trained = model.create_trainable_state(images, jax.random.PRNGKey(0))
    processed, metrics = compiled.postprocess(
        jnp.zeros((2, 8, 8, 2)), images, state, trained
    )
    updated, expected_metrics = compiled.compute_estimate(
        images, state, trained
    )
    np.testing.assert_allclose(processed, updated["estimates"][:, -1])
    assert metrics.keys() == expected_metrics.keys(), "Metrics must match"
    for key, value in metrics.items():
        np.testing.assert_allclose(value, expected_metrics[key])


def test_density_estimators_have_no_postprocess():
    """Only flow estimators expose a flow postprocessing pipeline."""
    compiled = make.compile_model(SimpleDensityEstimator(threshold=0.5), None)
    assert compiled.create_state is None, "No estimates were provided"
    assert compiled.postprocess is None, "Density has no postprocessing"
    assert _is_jitted(compiled.preprocess), "Preprocessing is pure JAX"


def test_jit_false_compiles_nothing():
    """Disabling JIT returns every function eagerly."""
    model = DummyEstimator(preprocessing_steps=[RESIZE_IMAGE])
    compiled = make.compile_model(model, jnp.zeros((2, 4, 6, 2)), jit=False)
    for fn in (
        compiled.create_state,
        compiled.compute_estimate,
        compiled.preprocess,
        compiled.postprocess,
    ):
        assert not _is_jitted(fn), "jit=False must not compile"


def test_non_jittable_estimator_keeps_processing_jitted(monkeypatch):
    """A non-JAX estimator does not stop its pure-JAX pipelines compiling."""
    model = DummyEstimator(
        preprocessing_steps=[RESIZE_IMAGE], postprocessing_steps=[SMOOTHING]
    )
    monkeypatch.setattr(model, "supports_jit", lambda: False)
    compiled = make.compile_model(model, jnp.zeros((2, 4, 6, 2)))
    assert not _is_jitted(compiled.create_state), "Runs the estimator"
    assert not _is_jitted(compiled.compute_estimate), "Runs the estimator"
    assert _is_jitted(compiled.preprocess), "Steps are pure JAX"
    assert _is_jitted(compiled.postprocess), "Steps are pure JAX"


def test_non_jittable_preprocessing_step_runs_eagerly(monkeypatch):
    """A step declaring jittable=False disables JIT wherever it runs."""
    monkeypatch.setattr(
        preprocess_module.resize_image, "jittable", False, raising=False
    )
    model = DummyEstimator(
        preprocessing_steps=[RESIZE_IMAGE], postprocessing_steps=[SMOOTHING]
    )
    compiled = make.compile_model(model, jnp.zeros((2, 4, 6, 2)))
    assert not _is_jitted(compiled.preprocess), "Step is not jittable"
    assert not _is_jitted(compiled.create_state), "State preprocesses"
    assert not _is_jitted(compiled.compute_estimate), "Estimate preprocesses"
    assert _is_jitted(compiled.postprocess), "Postprocessing is unaffected"


def test_non_jittable_postprocessing_step_runs_eagerly(monkeypatch):
    """A non-jittable flow filter only disables JIT where it runs."""
    monkeypatch.setattr(
        postprocess_module.temporal_smoothing_ema,
        "jittable",
        False,
        raising=False,
    )
    model = DummyEstimator(
        preprocessing_steps=[RESIZE_IMAGE], postprocessing_steps=[SMOOTHING]
    )
    compiled = make.compile_model(model, jnp.zeros((2, 4, 6, 2)))
    assert not _is_jitted(compiled.postprocess), "Step is not jittable"
    assert not _is_jitted(compiled.compute_estimate), "Estimate postprocesses"
    assert _is_jitted(compiled.preprocess), "Preprocessing is unaffected"
    assert _is_jitted(compiled.create_state), "State does not postprocess"
