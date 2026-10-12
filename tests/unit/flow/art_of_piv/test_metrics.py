"""Tests for the ART of PIV metrics processing module."""

import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.art_of_piv.metrics import (
    ChoiceTracker,
    build_inference_metrics,
    process_art_of_piv_metrics,
)

# ---------------------------------------------------------------------------
# ChoiceTracker tests
# ---------------------------------------------------------------------------


class TestChoiceTracker:
    """Tests for the ChoiceTracker helper."""

    def test_batch_rates_sum_to_one(self):
        """Batch rates across all estimators should sum to 1."""
        tracker = ChoiceTracker(num_estimators=3, window_size=10)
        choices = np.array([0, 1, 1, 2, 0])
        result = tracker.update(choices)

        total = sum(
            float(result[f"choices/rate_estimator_{i}"]) for i in range(3)
        )
        assert abs(total - 1.0) < 1e-6, (
            f"Batch rates sum to {total}, expected 1.0"
        )

    def test_running_rates_accumulate(self):
        """Running rates should reflect the full history of updates."""
        tracker = ChoiceTracker(num_estimators=2, window_size=100)

        # All choices are estimator 0
        tracker.update(np.array([0, 0, 0, 0]))
        # All choices are estimator 1
        result = tracker.update(np.array([1, 1, 1, 1]))

        assert float(
            result["choices/running_rate_estimator_0"]
        ) == pytest.approx(0.5), "Expected running rate of 0.5 for estimator 0"
        assert float(
            result["choices/running_rate_estimator_1"]
        ) == pytest.approx(0.5), "Expected running rate of 0.5 for estimator 1"

    def test_window_eviction(self):
        """Old batches should be evicted after `window_size` updates."""
        tracker = ChoiceTracker(num_estimators=2, window_size=2)

        # Batch 1: all estimator 0
        tracker.update(np.array([0, 0]))
        # Batch 2: all estimator 1
        tracker.update(np.array([1, 1]))
        # Batch 3: all estimator 1 — batch 1 should be evicted
        result = tracker.update(np.array([1, 1]))

        # Window now holds batches 2 and 3, both estimator 1
        assert float(
            result["choices/window_rate_estimator_0"]
        ) == pytest.approx(0.0), (
            "Estimator 0 should have been evicted from window"
        )
        assert float(
            result["choices/window_rate_estimator_1"]
        ) == pytest.approx(1.0), "Estimator 1 should dominate after eviction"

    def test_window_rates_are_sample_weighted(self):
        """Window rates should be weighted by batch size, not update count."""
        tracker = ChoiceTracker(num_estimators=2, window_size=2)

        # Window contains two updates with different batch sizes.
        tracker.update(np.array([0, 0, 0, 0]))  # 4 samples, all estimator 0
        result = tracker.update(np.array([1]))  # 1 sample, estimator 1

        assert float(
            result["choices/window_rate_estimator_0"]
        ) == pytest.approx(0.8), "Expected 4/5 for estimator 0"
        assert float(
            result["choices/window_rate_estimator_1"]
        ) == pytest.approx(0.2), "Expected 1/5 for estimator 1"

    def test_single_estimator(self):
        """Edge case: only one estimator available."""
        tracker = ChoiceTracker(num_estimators=1, window_size=5)
        result = tracker.update(np.array([0, 0, 0]))

        assert float(result["choices/rate_estimator_0"]) == pytest.approx(
            1.0
        ), "Single estimator should always have rate 1.0"

    def test_jax_array_input(self):
        """Tracker should accept JAX arrays as input."""
        tracker = ChoiceTracker(num_estimators=3, window_size=10)
        choices = jnp.array([0, 2, 2, 1])
        result = tracker.update(choices)

        assert float(result["choices/rate_estimator_2"]) == pytest.approx(
            0.5
        ), "Expected 50% rate for estimator 2"


# ---------------------------------------------------------------------------
# process_art_of_piv_metrics tests
# ---------------------------------------------------------------------------


class TestProcessMetrics:
    """Tests for the process_art_of_piv_metrics function."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh ChoiceTracker for each test."""
        return ChoiceTracker(num_estimators=4, window_size=10)

    def test_q_value_stats(self, tracker):
        """Q-value summary statistics should be present and correct."""
        q = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        raw = {"q_values": q, "choices": jnp.array([0, 1])}
        result = process_art_of_piv_metrics(raw, tracker)

        assert "q_values/mean" in result, "q_values/mean should be present"
        assert float(result["q_values/mean"]) == pytest.approx(4.5), (
            f"Expected mean 4.5, got {result['q_values/mean']}"
        )
        assert float(result["q_values/max"]) == pytest.approx(8.0), (
            f"Expected max 8.0, got {result['q_values/max']}"
        )
        assert float(result["q_values/min"]) == pytest.approx(1.0), (
            f"Expected min 1.0, got {result['q_values/min']}"
        )

    def test_choice_tracker_integration(self, tracker):
        """process_art_of_piv_metrics should feed choices to the tracker."""
        raw = {"choices": jnp.array([0, 0, 1, 3])}
        result = process_art_of_piv_metrics(raw, tracker)

        assert "choices/rate_estimator_0" in result, (
            "Per-batch rate should be present"
        )
        assert float(result["choices/rate_estimator_0"]) == pytest.approx(
            0.5
        ), "Expected 50% rate for estimator 0"

    def test_scalar_passthrough_keys(self, tracker):
        """Scalar metrics should be passed through verbatim."""
        raw = {
            "exploration_rate": jnp.array([0.1]),
            "estimated_epe": jnp.array(2.5),
            "choices/entropy": jnp.array(1.2),
            "choices/accuracy": jnp.array(0.75),
            "oracle_epe": jnp.array(0.5),
            "regret": jnp.array(0.3),
        }
        result = process_art_of_piv_metrics(raw, tracker)

        assert float(result["exploration/rate"]) == pytest.approx(0.1), (
            "exploration/rate should be 0.1"
        )
        assert float(result["oracle_epe"]) == pytest.approx(0.5), (
            "oracle_epe should be 0.5"
        )

    def test_epsilon_accuracy_keys(self, tracker):
        """Epsilon-accuracy keys should be forwarded."""
        raw = {
            "choices/accuracy_eps_0.005": jnp.array(0.9),
            "choices/accuracy_eps_0.001": jnp.array(0.8),
        }
        result = process_art_of_piv_metrics(raw, tracker)

        assert "choices/accuracy_eps_0.005" in result, (
            "accuracy_eps_0.005 should be present"
        )

    def test_error_keys(self, tracker):
        """Error and relative_errors should be forwarded."""
        raw = {
            "errors": jnp.array([0.1, 0.2]),
            "relative_errors": jnp.array([0.01, 0.02]),
        }
        result = process_art_of_piv_metrics(raw, tracker)

        assert "errors" in result, "errors should be present"
        assert "relative_errors" in result, "relative_errors should be present"

    def test_training_step_metrics(self, tracker):
        """Training-step metrics should be forwarded."""
        raw = {
            "epe/raw_mean": jnp.array(1.5),
            "target/mean": jnp.array(-0.5),
            "regret/raw_mean": jnp.array(0.2),
            "regret/raw_p90": jnp.array(0.4),
            "epe/raw_best_mean": jnp.array(0.8),
        }
        result = process_art_of_piv_metrics(raw, tracker)

        for key in raw:
            assert key in result, (
                f"{key} should be present in processed metrics"
            )

    def test_improvement_metrics(self, tracker):
        """Improvement metrics should be computed when both inputs exist."""
        raw = {"mean_error": 2.0, "oracle_epe/mean": 1.0}
        result = process_art_of_piv_metrics(raw, tracker)

        assert float(result["improvement_margin"]) == pytest.approx(1.0), (
            "Improvement margin should be 1.0"
        )
        assert float(result["improvement_pct"]) == pytest.approx(50.0), (
            "Improvement pct should be 50%"
        )

    def test_empty_metrics(self, tracker):
        """Empty raw metrics should produce an empty Metrics object."""
        result = process_art_of_piv_metrics({}, tracker)
        assert len(result) == 0, f"Expected empty metrics, got {dict(result)}"


class TestBuildInferenceMetrics:
    """Tests for the shared inference metric assembly helper."""

    def test_without_cache_payload(self):
        """Base metrics should be emitted without cache-dependent fields."""
        qs = jnp.array([[1.0, 0.0], [0.2, 0.8]], dtype=jnp.float32)
        choices = jnp.array([0, 1], dtype=jnp.int32)
        exploration_rate = jnp.array([0.1, 0.2], dtype=jnp.float32)
        steps = jnp.array([3, 4], dtype=jnp.int32)
        estimated_epe = jnp.array([0.5, 0.6], dtype=jnp.float32)

        metrics = build_inference_metrics(
            qs=qs,
            choices=choices,
            exploration_rate=exploration_rate,
            steps=steps,
            estimated_epe=estimated_epe,
            cache_payload=None,
        )

        assert "q_values" in metrics
        assert "choices/entropy" in metrics
        assert "choices/accuracy" not in metrics
        np.testing.assert_array_equal(metrics["choices"], np.array([0, 1]))
        np.testing.assert_allclose(metrics["estimated_epe"], estimated_epe)

    def test_with_cache_payload_and_errors(self):
        """Cache payload should add oracle/accuracy/regret and keep errors."""
        qs = jnp.array([[0.9, 0.1], [0.2, 0.8]], dtype=jnp.float32)
        choices = jnp.array([0, 1], dtype=jnp.int32)
        exploration_rate = jnp.array([0.0, 0.0], dtype=jnp.float32)
        steps = jnp.array([5, 5], dtype=jnp.int32)

        class CachePayloadMock:
            epe_all = np.array([[0.1, 0.3], [0.4, 0.2]], dtype=np.float32)
            epe_mask = np.array([[True, True], [True, True]])

        errors = jnp.array([0.1, 0.2], dtype=jnp.float32)
        metrics = build_inference_metrics(
            qs=qs,
            choices=choices,
            exploration_rate=exploration_rate,
            steps=steps,
            estimated_epe=-1.0,
            cache_payload=CachePayloadMock(),
            errors=errors,
        )

        np.testing.assert_allclose(metrics["choices/accuracy"], [1.0, 1.0])
        np.testing.assert_allclose(metrics["oracle_epe"], [0.1, 0.2])
        np.testing.assert_allclose(metrics["regret"], [0.0, 0.0])
        np.testing.assert_allclose(metrics["errors"], errors)
        assert "choices/accuracy_eps_0.005" in metrics
        assert "choices/accuracy_eps_0.001" in metrics
