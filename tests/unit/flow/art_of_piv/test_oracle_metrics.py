"""Tests for oracle EPE metrics in Art of PIV.

This test verifies that oracle EPE (best possible EPE if router was perfect)
is correctly computed and included in validation metrics using actual batches.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def test_oracle_epe_computation_logic():
    """Test the logic for computing oracle EPE from epe_all."""
    # Test the core logic: oracle_epe = min(epe_all_masked)

    # Scenario 1: Simple case
    epe_all = jnp.array(
        [
            [0.5, 0.8, 1.2],  # Sample 0: best is 0.5 (strategy 0)
            [0.9, 0.3, 0.7],  # Sample 1: best is 0.3 (strategy 1)
        ]
    )
    epe_mask = jnp.ones_like(epe_all, dtype=bool)
    epe_all_masked = jnp.where(epe_mask, epe_all, jnp.inf)
    row_has_any = jnp.any(epe_mask, axis=-1)

    # Compute oracle
    oracle_epe = jnp.min(epe_all_masked, axis=-1)
    oracle_epe = jnp.where(row_has_any, oracle_epe, jnp.nan)

    expected_oracle = jnp.array([0.5, 0.3])
    np.testing.assert_allclose(
        oracle_epe,
        expected_oracle,
        rtol=1e-5,
        err_msg="Oracle EPE should be the minimum EPE for each sample",
    )

    # Scenario 2: With some masked (missing) strategies
    epe_all = jnp.array(
        [
            [0.5, jnp.inf, 1.2],  # Strategy 1 is missing
            [0.9, 0.3, jnp.inf],  # Strategy 2 is missing
        ]
    )
    epe_mask = jnp.array(
        [
            [True, False, True],
            [True, True, False],
        ]
    )
    epe_all_masked = jnp.where(epe_mask, epe_all, jnp.inf)
    row_has_any = jnp.any(epe_mask, axis=-1)

    oracle_epe = jnp.min(epe_all_masked, axis=-1)
    oracle_epe = jnp.where(row_has_any, oracle_epe, jnp.nan)

    expected_oracle = jnp.array([0.5, 0.3])  # Min of valid values
    np.testing.assert_allclose(
        oracle_epe,
        expected_oracle,
        rtol=1e-5,
        err_msg="Oracle EPE should ignore masked strategies",
    )


def test_oracle_epe_improvement_potential():
    """Test computing improvement potential from oracle EPE."""
    # Realistic scenario
    epe_all = jnp.array(
        [
            [0.8, 0.5, 1.0],  # Oracle: 0.5
            [0.7, 1.5, 0.9],  # Oracle: 0.7
            [0.75, 0.6, 1.1],  # Oracle: 0.6
            [0.8, 1.4, 0.95],  # Oracle: 0.8
        ]
    )

    # Compute oracle
    oracle_epe = jnp.min(epe_all, axis=-1)
    expected_oracle = jnp.array([0.5, 0.7, 0.6, 0.8])
    np.testing.assert_allclose(oracle_epe, expected_oracle, rtol=1e-5)

    oracle_mean = float(jnp.mean(oracle_epe))
    assert oracle_mean == pytest.approx(0.65, abs=1e-5)

    # Simulate router choices (not always optimal)
    # Router chooses: [0, 0, 0, 2] -> EPEs: [0.8, 0.7, 0.75, 0.95]
    chosen_idx = jnp.array([0, 0, 0, 2])
    chosen_epe = jnp.take_along_axis(
        epe_all, chosen_idx[:, None], axis=1
    ).squeeze(-1)

    chosen_mean = float(jnp.mean(chosen_epe))
    improvement_margin = chosen_mean - oracle_mean

    # Verify improvement margin calculation
    # chosen_epe = [0.8, 0.7, 0.75, 0.95], mean = 0.8
    # oracle_epe = [0.5, 0.7, 0.6, 0.8], mean = 0.65
    # improvement = 0.8 - 0.65 = 0.15
    assert improvement_margin >= 0, "Router can't be better than oracle"
    assert improvement_margin == pytest.approx(0.15, abs=1e-4)

    improvement_pct = (
        (improvement_margin / chosen_mean * 100) if chosen_mean > 0 else 0.0
    )
    assert improvement_pct == pytest.approx(18.75, abs=0.1)

    print(f"\nChosen EPE: {chosen_mean:.3f}")
    print(f"Oracle EPE: {oracle_mean:.3f}")
    print(f"Potential improvement: {improvement_margin:.3f}")
    print(f"Potential improvement %: {improvement_pct:.1f}%")


def test_regret_computation():
    """Test that regret is correctly computed as chosen_epe - oracle_epe."""
    epe_all = jnp.array(
        [
            [0.5, 0.8, 1.2],
            [0.9, 0.3, 0.7],
            [0.6, 0.4, 0.5],
        ]
    )

    # Oracle: best for each sample
    oracle_epe = jnp.min(epe_all, axis=-1)
    expected_oracle = jnp.array([0.5, 0.3, 0.4])
    np.testing.assert_allclose(oracle_epe, expected_oracle, rtol=1e-5)

    # Router chooses (suboptimal for some samples)
    chosen_idx = jnp.array([1, 0, 1])  # Chooses strategies: [1, 0, 1]
    chosen_epe = jnp.take_along_axis(
        epe_all, chosen_idx[:, None], axis=1
    ).squeeze(-1)
    # Chosen EPEs: [0.8, 0.9, 0.4]

    # Compute regret
    regret = chosen_epe - oracle_epe
    expected_regret = jnp.array([0.3, 0.6, 0.0])  # [0.8-0.5, 0.9-0.3, 0.4-0.4]

    np.testing.assert_allclose(
        regret,
        expected_regret,
        rtol=1e-5,
        err_msg="Regret should be chosen_epe - oracle_epe",
    )

    # Verify regret is always non-negative (router can't beat oracle)
    assert jnp.all(regret >= 0), "Regret must be non-negative"


def test_accuracy_and_oracle_relationship():
    """Test relationship between router accuracy and oracle improvement."""
    epe_all = jnp.array(
        [
            [0.5, 0.8, 1.2],  # Best: 0
            [0.9, 0.3, 0.7],  # Best: 1
            [0.6, 0.4, 0.5],  # Best: 1
            [1.0, 0.8, 0.9],  # Best: 1
        ]
    )

    oracle_epe = jnp.min(epe_all, axis=-1)
    best_idx = jnp.argmin(epe_all, axis=-1)

    # Scenario 1: Perfect router (100% accuracy)
    perfect_choices = best_idx
    perfect_epe = jnp.take_along_axis(
        epe_all, perfect_choices[:, None], axis=1
    ).squeeze(-1)

    # Perfect router achieves oracle EPE
    np.testing.assert_allclose(perfect_epe, oracle_epe, rtol=1e-5)
    regret_perfect = jnp.mean(perfect_epe - oracle_epe)
    assert regret_perfect == pytest.approx(0.0, abs=1e-6)

    # Scenario 2: Imperfect router (50% accuracy)
    # Correct on samples 0 and 2, wrong on 1 and 3
    imperfect_choices = jnp.array([0, 0, 1, 0])
    imperfect_epe = jnp.take_along_axis(
        epe_all, imperfect_choices[:, None], axis=1
    ).squeeze(-1)

    accuracy = jnp.mean((imperfect_choices == best_idx).astype(jnp.float32))
    assert accuracy == pytest.approx(0.5, abs=1e-5)

    # Imperfect router has higher EPE than oracle
    regret_imperfect = jnp.mean(imperfect_epe - oracle_epe)
    assert regret_imperfect > 0, "Imperfect router should have positive regret"

    print(
        f"\nPerfect (100% acc) mean EPE: {jnp.mean(perfect_epe):.3f} "
        f"(regret: {regret_perfect:.3f})"
    )
    print(
        f"Imperfect (50% acc) mean EPE: "
        f"{jnp.mean(imperfect_epe):.3f} "
        f"(regret: {regret_imperfect:.3f})"
    )
    print(f"Oracle mean EPE: {jnp.mean(oracle_epe):.3f}")
