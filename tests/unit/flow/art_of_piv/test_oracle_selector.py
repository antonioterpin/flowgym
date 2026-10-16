"""Tests for the ART of PIV oracle selector."""

import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.art_of_piv import ArtOfPIVOracleEstimator
from flowgym.types import CachePayload


@pytest.fixture
def oracle(tmp_path):
    """Oracle over a 3-member library, sub-estimators not built."""
    library = tmp_path / "library.yaml"
    library.write_text("estimators: [{name: a}, {name: b}, {name: c}]\n")
    return ArtOfPIVOracleEstimator(
        estimators_list_path=str(library), initialize_estimators=False
    )


def test_oracle_from_router_config(tmp_path):
    """A router's config builds the oracle; router_config, oracle dropped."""
    library = tmp_path / "library.yaml"
    library.write_text("estimators: [{name: a}, {name: b}]\n")
    oracle = ArtOfPIVOracleEstimator.from_config(
        {
            "estimators_list_path": str(library),
            "initialize_estimators": False,
            "router_config": {"hidden_dims": [8]},
            "oracle": False,
        }
    )
    assert oracle.num_estimators == 2
    assert oracle.is_oracle()


def test_oracle_rejects_empty_library(tmp_path):
    """An empty library fails at construction, not at inference."""
    library = tmp_path / "library.yaml"
    library.write_text("estimators: []\n")
    with pytest.raises(ValueError, match="No estimators"):
        ArtOfPIVOracleEstimator(
            estimators_list_path=str(library), initialize_estimators=False
        )


def test_oracle_picks_lowest_valid_cached_epe(oracle):
    """The choice is the lowest cached EPE, skipping masked entries."""
    # Row 1's lowest EPE (0.05) is masked out, so its oracle is 0.25
    payload = CachePayload(
        has_precomputed_errors=True,
        epe_all=np.array([[0.3, 0.1, 0.2], [0.05, 0.25, 0.35]], np.float32),
        epe_mask=np.array([[True, True, True], [False, True, True]]),
    )
    _, extras, metrics = oracle._estimate(
        jnp.zeros((2, 8, 8)), {}, None, {"cache_payload": payload}
    )
    np.testing.assert_array_equal(extras["choices"], [1, 1])
    np.testing.assert_allclose(metrics["errors"], [0.1, 0.25])


def test_oracle_runs_library_live_without_cache(oracle):
    """Without cached errors, every member runs and the GT picks the best."""
    # Member k outputs a constant flow k
    oracle.estimator_fns = tuple(
        lambda images, state, ts, k=k: (
            {"estimates": jnp.full((2, 1, 8, 8, 2), float(k))},
            {},
        )
        for k in range(3)
    )
    oracle.sub_model_states = (None, None, None)
    gt = jnp.stack([jnp.full((8, 8, 2), 0.9), jnp.full((8, 8, 2), 2.2)])
    state = {"estimates": gt[:, jnp.newaxis], "keys": jnp.zeros((2, 2))}

    flow, extras, _ = oracle._estimate(jnp.zeros((2, 8, 8)), state, None, {})

    np.testing.assert_array_equal(extras["choices"], [1, 2])
    np.testing.assert_allclose(flow[:, 0, 0, 0], [1.0, 2.0])
