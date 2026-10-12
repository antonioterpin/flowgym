"""Test for ART of PIV cache short-circuit.

This test ensures that when cached EPEs are available via CachePayload,
the ART of PIV estimator skips expensive sub-estimator computation during
both training and evaluation.
"""

import jax.numpy as jnp
import numpy as np

from flowgym.types import CachePayload


def test_art_of_piv_short_circuits_with_cached_epe():
    """Test that ART of PIV uses cached EPEs and skips sub-estimator inference.

    This is a regression test for the bug where evaluation time didn't improve
    on the second run despite having cached EPEs available.

    The root cause: inference.py checks `"epe_all" in extras` but epe_all is in
    `cache_payload.epe_all`, not directly in extras.
    """
    from flowgym.flow.art_of_piv import inference as art_of_piv_inference

    B, num_estimators = 2, 3
    H, W = 64, 64

    # Create a cache payload with epe_all and epe_mask
    epe_all = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]], dtype=np.float32)
    epe_mask = np.ones((B, num_estimators), dtype=bool)

    cache_payload = CachePayload(
        has_precomputed_errors=True,
        epe_all=epe_all,
        epe_mask=epe_mask,
    )

    # Create mock state and trainable state
    state = {
        "images": jnp.zeros((B, 1, H, W)),
        "estimates": jnp.zeros((B, 1, H, W, 2)),
    }

    class MockTrainableState:
        def __init__(self):
            self.extras = {"steps": jnp.zeros((1, 1))}

    trainable_state = MockTrainableState()

    # Create extras dict with cache_payload (as done by estimator.py)
    extras = {
        "cache_payload": cache_payload,
        "train": True,  # Enable exploration in router
    }

    # Also flatten cache_payload fields into extras (current behavior)
    # This is what should happen, but doesn't in the current implementation
    extras.update(cache_payload.extras)

    # Mock router function that returns choices and Q-values
    def mock_run_router_fn(*args, **kwargs):
        choices = jnp.array([0, 1])  # Select different estimators
        qs = jnp.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]])
        exploration_rate = jnp.array([0.1, 0.1])
        estimate_keys = jnp.zeros((B, 2), dtype=jnp.uint32)
        return choices, qs, exploration_rate, estimate_keys

    # Outer-scope sentinel used by the nested mock to record invocation.
    # The final assertion verifies that cache short-circuiting kept this False.
    sub_estimators_called = False

    def mock_run_subestimators_fn(*args, **kwargs):
        # `nonlocal` is required to mutate the outer sentinel flag.
        nonlocal sub_estimators_called
        sub_estimators_called = True
        # Return dummy values
        flow = jnp.zeros((B, H, W, 2))
        out_extras = {"choices": jnp.array([0, 1])}
        metrics = {"errors": jnp.zeros(B)}
        return flow, out_extras, metrics

    # Call the estimate function
    images = jnp.zeros((B, H, W))

    _flow, _out_extras, metrics = art_of_piv_inference.estimate(
        images=images,
        state=state,
        trainable_state=trainable_state,
        extras=extras,
        use_temporal_propagation=False,
        use_last_estimate=False,
        run_router_fn=mock_run_router_fn,
        run_subestimators_fn=mock_run_subestimators_fn,
    )

    # ASSERTIONS
    # 1. Sub-estimators should NOT have been called (short-circuit should work)
    assert not sub_estimators_called, (
        "Sub-estimators were called even though cached EPEs were available. "
        "The short-circuit logic should have prevented this."
    )

    # 2. The returned metrics should contain the cached EPEs
    assert "errors" in metrics, (
        "Cached short-circuit metrics must expose errors."
    )
    assert "estimated_epe" in metrics, (
        "Cached short-circuit metrics must expose estimated_epe."
    )
    # Choice 0 selected estimator 0 -> epe_all[0, 0] = 0.1
    # Choice 1 selected estimator 1 -> epe_all[1, 1] = 0.25
    expected_errors = jnp.array([0.1, 0.25])
    np.testing.assert_allclose(
        metrics["errors"],
        expected_errors,
        rtol=1e-5,
        err_msg=(
            "Cached short-circuit should emit per-sample errors selected "
            "from cache_payload.epe_all by chosen estimator."
        ),
    )
    expected_estimated_epe = jnp.array([0.9, 0.9])
    np.testing.assert_allclose(
        metrics["estimated_epe"],
        expected_estimated_epe,
        rtol=1e-5,
        err_msg=(
            "Cached short-circuit should preserve router Q-value for each "
            "chosen action in estimated_epe."
        ),
    )

    # 3. Oracle EPE should be the minimum across all estimators
    expected_oracle = jnp.array([0.1, 0.15])  # Min of each row
    assert "oracle_epe" in metrics, (
        "Cached short-circuit metrics must include oracle_epe."
    )
    np.testing.assert_allclose(
        metrics["oracle_epe"],
        expected_oracle,
        rtol=1e-5,
        err_msg=(
            "oracle_epe should be the per-sample minimum across all cached "
            "sub-estimator EPE values."
        ),
    )

    # 4. Accuracy should be computed (choice 0 picked best, choice 1 didn't)
    assert "choices/accuracy" in metrics, (
        "Cached short-circuit metrics must expose choices/accuracy."
    )
    # Sample 0: chose 0, best is 0 -> correct
    # Sample 1: chose 1, best is 0 -> incorrect
    expected_accuracy = jnp.array([1.0, 0.0])
    np.testing.assert_allclose(
        metrics["choices/accuracy"],
        expected_accuracy,
        rtol=1e-5,
        err_msg=(
            "choices/accuracy should compare each selected estimator against "
            "the oracle choice from cached EPEs."
        ),
    )


def test_art_of_piv_short_circuits_during_evaluation():
    """Test that ART of PIV uses cached EPEs during evaluation (train=False).

    This test verifies that the cache optimization works during evaluation,
    not just training, which provides significant speedup benefits.
    """
    from flowgym.flow.art_of_piv import inference as art_of_piv_inference

    B, num_estimators = 2, 3
    H, W = 64, 64

    # Create a cache payload with epe_all and epe_mask
    epe_all = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]], dtype=np.float32)
    epe_mask = np.ones((B, num_estimators), dtype=bool)

    cache_payload = CachePayload(
        has_precomputed_errors=True,
        epe_all=epe_all,
        epe_mask=epe_mask,
    )

    # Create mock state and trainable state
    state = {
        "images": jnp.zeros((B, 1, H, W)),
        "estimates": jnp.zeros((B, 1, H, W, 2)),
    }

    class MockTrainableState:
        def __init__(self):
            self.extras = {"steps": jnp.zeros((1, 1))}

    trainable_state = MockTrainableState()

    # Create extras dict with cache_payload (evaluation mode)
    extras = {
        "cache_payload": cache_payload,
        "train": False,  # Evaluation mode - no exploration
    }
    extras.update(cache_payload.extras)

    # Mock router function that returns greedy choices
    def mock_run_router_fn(*args, **kwargs):
        # During evaluation, router should pick greedily (best Q-values)
        choices = jnp.array([0, 0])  # Both pick estimator 0 (best)
        qs = jnp.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0]])
        exploration_rate = jnp.array([0.0, 0.0])  # No exploration
        estimate_keys = jnp.zeros((B, 2), dtype=jnp.uint32)
        return choices, qs, exploration_rate, estimate_keys

    # Outer-scope sentinel used by the nested mock to record invocation.
    # The final assertion verifies that cache short-circuiting kept this False.
    sub_estimators_called = False

    def mock_run_subestimators_fn(*args, **kwargs):
        # `nonlocal` is required to mutate the outer sentinel flag.
        nonlocal sub_estimators_called
        sub_estimators_called = True
        # Return dummy values
        flow = jnp.zeros((B, H, W, 2))
        out_extras = {"choices": jnp.array([0, 0])}
        metrics = {"errors": jnp.zeros(B)}
        return flow, out_extras, metrics

    # Call the estimate function
    images = jnp.zeros((B, H, W))

    _flow, _out_extras, metrics = art_of_piv_inference.estimate(
        images=images,
        state=state,
        trainable_state=trainable_state,
        extras=extras,
        use_temporal_propagation=False,
        use_last_estimate=False,
        run_router_fn=mock_run_router_fn,
        run_subestimators_fn=mock_run_subestimators_fn,
    )

    # ASSERTIONS
    # 1. Sub-estimators should NOT have been called (cache short-circuit works)
    assert not sub_estimators_called, (
        "Sub-estimators were called during evaluation even though cached EPEs "
        "were available. The cache optimization should work for both training "
        "and evaluation."
    )

    # 2. The returned metrics should contain the cached EPEs
    assert "errors" in metrics, "Evaluation short-circuit metrics need errors."
    # Both samples chose estimator 0:
    # epe_all[0, 0] = 0.1, epe_all[1, 0] = 0.15
    expected_errors = jnp.array([0.1, 0.15])
    np.testing.assert_allclose(
        metrics["errors"],
        expected_errors,
        rtol=1e-5,
        err_msg=(
            "Evaluation short-circuit should emit cached per-sample errors "
            "for greedy estimator choices."
        ),
    )

    # 3. Exploration rate should be zero during evaluation
    np.testing.assert_allclose(
        metrics["exploration_rate"],
        [0.0, 0.0],
        err_msg=(
            "Evaluation mode should force zero exploration rate in "
            "short-circuit path."
        ),
    )

    # 4. Both samples should have perfect accuracy (both chose best)
    assert "choices/accuracy" in metrics, (
        "Evaluation short-circuit metrics need choices/accuracy."
    )
    expected_accuracy = jnp.array([1.0, 1.0])
    np.testing.assert_allclose(
        metrics["choices/accuracy"],
        expected_accuracy,
        rtol=1e-5,
        err_msg=(
            "Evaluation short-circuit should report perfect accuracy when "
            "both greedy choices match oracle selections."
        ),
    )
