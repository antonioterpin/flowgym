import pytest
import jax.numpy as jnp
from jax import random

from flowgym.flow.consensus.consensus_algorithms import z_objective, flows_objective
from flowgym.flow.consensus.objectives import make_weights


@pytest.mark.parametrize(
    "N, H, W, C",
    [
        (1, 2, 2, 1),
        (3, 4, 5, 2),
    ],
)
@pytest.mark.parametrize("rho", [0.1, 1.0, 2.0])
def test_z_objective_no_regularization(N, H, W, C, rho):
    """Test consensus term with no regularizer."""
    flows = jnp.ones((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    # The consensus term: each residual is 1, squared = 1, sum over all = N*H*W*C
    expected = 0.5 * rho * (N * H * W * C)
    result = z_objective(consensus_flow, flows, consensus_dual, rho)
    assert jnp.isscalar(result) or result.shape == ()
    assert jnp.isclose(result, expected)


@pytest.mark.parametrize("N, H, W, C", [(1, 2, 2, 1)])
@pytest.mark.parametrize("regularizer_loss", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("rho", [1.0])
def test_z_objective_with_regularizer(N, H, W, C, rho, monkeypatch, regularizer_loss):
    """Test consensus term plus regularization."""
    flows = jnp.ones((N, H, W, C)) + 1
    consensus_flow = jnp.ones((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))

    def dummy_total_regularization_loss(z, reg_names, reg_weights_dict):
        # Simulate a single term, always returning 15.0
        return regularizer_loss

    # Define a regularizer that always returns 5, weight=3
    reg_list = ["smoothness"]  # matches code expectations
    reg_weights = {"smoothness": 3.0}

    # Patch total_regularization_loss to our dummy version
    monkeypatch.setattr(
        "flowgym.flow.consensus.objectives.total_regularization_loss",
        dummy_total_regularization_loss,
    )

    # Now consensus_term = 0, reg_term = 15
    result = z_objective(
        consensus_flow, flows, consensus_dual, rho, reg_list, reg_weights
    )
    assert jnp.isscalar(result)
    assert jnp.isclose(result, regularizer_loss + 0.5 * rho * (N * H * W * C))


@pytest.mark.parametrize("N, H, W, C", [(1, 2, 2, 1)])
@pytest.mark.parametrize("rho", [0.1, 2.0])
def test_z_objective_consensus_term_scales_with_rho(N, H, W, C, rho):
    """Test that consensus term scales linearly with rho."""
    flows = jnp.ones((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    result = z_objective(consensus_flow, flows, consensus_dual, rho)
    expected = 0.5 * rho * (N * H * W * C)
    assert jnp.isclose(result, expected)


@pytest.mark.parametrize(
    "N, H, W, C",
    [
        (1, 2, 2, 1),
        (3, 4, 5, 2),
        (2, 3, 4, 1),
        (2, 2, 2, 1),
        (1, 1, 1, 1),
        (2, 2, 2, 2),
    ],
)
def test_z_objective_output_shape(N, H, W, C):
    """Test output is scalar for any reasonable input shape."""
    flows = jnp.zeros((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    out = z_objective(consensus_flow, flows, consensus_dual)
    assert jnp.isscalar(out)


@pytest.mark.parametrize(
    "N, H, W, C",
    [
        (1, 2, 2, 1),
        (3, 4, 5, 2),
    ],
)
@pytest.mark.parametrize("rho", [0.1, 1.0, 2.0])
def test_flows_objective_l2_no_weights(N, H, W, C, rho):
    """Test l2 data term with no weights, flows all ones, initial all zeros."""
    flows = jnp.ones((N, H, W, C))
    initial_flows = jnp.zeros((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    # data_term: all (1-0)**2 = 1, summed over all
    data_term = N * H * W * C
    consensus_term = 0.5 * rho * (N * H * W * C)  # all residuals = 1
    expected = data_term + consensus_term
    result = flows_objective(
        flows,
        consensus_flow,
        consensus_dual,
        initial_flows=initial_flows,
        weights=None,
        objective_type="l2",
        rho=rho,
    )
    assert jnp.isscalar(result)
    assert jnp.isclose(result, expected)


@pytest.mark.parametrize("N, H, W, C", [(2, 2, 2, 1)])
@pytest.mark.parametrize("rho", [1.0])
@pytest.mark.parametrize("weight_value", [0.5, 2.0])
def test_flows_objective_l2_with_weights(N, H, W, C, rho, weight_value):
    """Test l2 with weights applied to the data term."""
    flows = jnp.ones((N, H, W, C))
    initial_flows = jnp.zeros((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    weights = jnp.ones((N,)) * weight_value
    # Each anchor^2 = 1, each gets scaled by weight_value
    data_term = N * H * W * C * weight_value
    consensus_term = 0.5 * rho * (N * H * W * C)
    expected = data_term + consensus_term
    result = flows_objective(
        flows,
        consensus_flow,
        consensus_dual,
        initial_flows=initial_flows,
        weights=weights,
        objective_type="l2",
        rho=rho,
    )
    assert jnp.isclose(result, expected)


@pytest.mark.parametrize("N, H, W, C", [(1, 2, 2, 1)])
@pytest.mark.parametrize("rho", [0.1, 1.0, 2.0])
def test_flows_objective_l1_no_weights(N, H, W, C, rho):
    """Test l1 data term with no weights."""
    flows = -jnp.ones((N, H, W, C))
    initial_flows = jnp.zeros((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    # data_term: sum(abs(-1-0)) = num_elements
    data_term = N * H * W * C
    consensus_term = 0.5 * rho * (N * H * W * C)
    expected = data_term + consensus_term
    result = flows_objective(
        flows,
        consensus_flow,
        consensus_dual,
        initial_flows=initial_flows,
        weights=None,
        objective_type="l1",
        rho=rho,
    )
    assert jnp.isclose(result, expected)


@pytest.mark.parametrize("N, H, W, C", [(2, 2, 2, 1)])
@pytest.mark.parametrize("rho", [1.0])
@pytest.mark.parametrize("weight_value", [0.5, 2.0])
def test_flows_objective_l1_with_weights(N, H, W, C, rho, weight_value):
    """Test l1 with weights applied to the data term."""
    flows = -jnp.ones((N, H, W, C))
    initial_flows = jnp.zeros((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    weights = jnp.ones((N,)) * weight_value
    # Each |anchor| = 1, scaled by weight_value
    data_term = N * H * W * C * weight_value
    consensus_term = 0.5 * rho * (N * H * W * C)
    expected = data_term + consensus_term
    result = flows_objective(
        flows,
        consensus_flow,
        consensus_dual,
        initial_flows=initial_flows,
        weights=weights,
        objective_type="l1",
        rho=rho,
    )
    assert jnp.isclose(result, expected)


@pytest.mark.parametrize(
    "N, H, W, C",
    [
        (1, 2, 2, 1),
        (3, 4, 5, 2),
        (1, 2, 3, 4),
        (3, 2, 2, 2),
        (1, 1, 1, 1),
        (2, 2, 2, 2),
    ],
)
def test_flows_objective_output_shape(N, H, W, C):
    """Test output is scalar for any reasonable input shape."""
    flows = jnp.zeros((N, H, W, C))
    initial_flows = jnp.zeros((N, H, W, C))
    consensus_flow = jnp.zeros((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    out = flows_objective(
        flows, consensus_flow, consensus_dual, initial_flows=initial_flows
    )
    assert jnp.isscalar(out) or out.shape == ()


@pytest.mark.parametrize("N,H,W,C", [(1, 2, 2, 1)])
@pytest.mark.parametrize("objective_type", ["l2", "l1"])
def test_flows_objective_consensus_term_zero(N, H, W, C, objective_type):
    """Test that if flows == consensus_flow and dual==0, consensus_term==0."""
    flows = jnp.ones((N, H, W, C))
    initial_flows = jnp.ones((N, H, W, C))
    consensus_flow = jnp.ones((H, W, C))
    consensus_dual = jnp.zeros((N, H, W, C))
    result = flows_objective(
        flows,
        consensus_flow,
        consensus_dual,
        initial_flows=initial_flows,
        objective_type=objective_type,
    )
    # Data term is zero, consensus term is zero
    assert jnp.isclose(result, 0.0)


@pytest.fixture
def config():
    return {
        "weights_type": "photometric",
        "patch_size": 3,
        "patch_stride": 1,
        "normalization": "per_pixel",
    }


@pytest.mark.parametrize("B, N, H, W", [(1, 2, 64, 64), (2, 3, 256, 256)])
def test_photometric_weights_shape_and_sum(config, B, N, H, W):
    """Test that weights have the correct shape and sum to 1."""
    config["normalization"] = "per_batch"
    key = random.PRNGKey(0)
    flows = random.normal(key, (B, N, H, W, 2))
    prevs = random.normal(key, (B, H, W))
    currs = random.normal(key, (B, H, W))

    weights = make_weights(flows, prevs, currs, config)
    assert weights.shape == (B, N, H, W)
    assert jnp.allclose(jnp.sum(weights, axis=(1, 2, 3)), 1.0, atol=1e-5)


@pytest.mark.parametrize("B, N, H, W", [(1, 4, 64, 64), (2, 3, 256, 256)])
def test_photometric_weights_favor_best_flow(config, B, N, H, W):
    """Test that the perfect flow gets the highest weight."""
    key = random.PRNGKey(42)
    key_prev, key_curr, key_no_flow = random.split(key, 3)

    flows = random.normal(key, (B, N, H, W, 2)) * 10
    prevs = random.uniform(key_prev, (B, H, W)) * 255
    currs = random.uniform(key_curr, (B, H, W)) * 255

    perfect_flow = jnp.zeros((H, W, 2))  # Perfect flow for agent 0
    no_flow_image = (
        random.uniform(key_no_flow, (H, W)) * 255
    )  # No photometric error for perfect flow
    # Set agent 0 as perfect (all zeros)
    flows = flows.at[0, 0].set(perfect_flow)
    prevs = prevs.at[0].set(no_flow_image)
    currs = currs.at[0].set(no_flow_image)

    weights = make_weights(flows, prevs, currs, config)

    # Agent 0 should be highly favored
    max_weight = weights[0, 0].mean()

    # TODO: this threshold doesn't make sense if the normalization isn't per pixel
    assert max_weight > 0.85, f"Perfect agent's mean weight: {max_weight}"
    assert jnp.all(
        weights[0, 1:] < max_weight
    ), "Other agents should have lower weights"


@pytest.mark.parametrize("B, N, H, W", [(2, 4, 64, 64), (1, 3, 256, 256)])
def test_photometric_weights_uniform_when_equal(config, B, N, H, W):
    """Test that weights are uniform when all flows are identical."""
    key = random.PRNGKey(7)
    identical_flow = random.normal(key, (B, H, W, 2))
    flows = jnp.stack([identical_flow] * N, axis=1)
    prevs = random.normal(key, (B, H, W)) * 255
    currs = random.normal(key, (B, H, W)) * 255

    weights = make_weights(flows, prevs, currs, config)
    interior = weights[:, :, 1:-1, 1:-1]
    expected = 1.0 / N
    assert jnp.allclose(interior, expected, atol=1e-3)
    assert jnp.allclose(interior.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("N, H, W", [(2, 4, 4), (3, 8, 8)])
def test_photometric_weights_handles_4d_flows(config, N, H, W):
    key = random.PRNGKey(8)
    flows = random.normal(key, (N, H, W, 2))  # 4D input
    prevs = random.normal(key, (H, W))
    currs = random.normal(key, (H, W))

    weights = make_weights(flows, prevs, currs, config)
    assert weights.shape == (1, N, H, W)


@pytest.mark.parametrize("B, N, H, W", [(1, 2, 64, 64)])
def test_photometric_weights_zero_error_stability(config, B, N, H, W):
    flows = jnp.zeros((B, N, H, W, 2))
    prevs = jnp.zeros((B, H, W))
    currs = jnp.zeros((B, H, W))

    weights = make_weights(flows, prevs, currs, config)
    interior = weights[:, :, 1:-1, 1:-1]
    assert jnp.all(jnp.isfinite(weights)), "Weights contain inf or nan"
    expected = 1.0 / N
    assert jnp.allclose(interior, expected, atol=1e-3)
    assert jnp.allclose(interior.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("B, N, H, W", [(10, 2, 64, 64)])  # B must be > 3 for this test
def test_make_weights_batch_consistency(config, B, N, H, W):
    """Test that make_weights works consistently for full batch and sub-batch."""
    flows = random.uniform(random.PRNGKey(0), (B, N, H, W, 2)) * 5
    prevs = random.uniform(random.PRNGKey(1), (B, H, W)) * 255
    currs = random.uniform(random.PRNGKey(2), (B, H, W)) * 255

    # Get weights for full batch
    out_full = make_weights(flows, prevs, currs, config)
    # Get weights for first 3 elements
    out_sub1 = make_weights(flows[:3], prevs[:3], currs[:3], config)
    out_sub2 = make_weights(flows[0], prevs[0], currs[0], config)

    # Assert output shape
    assert out_full.shape[0] == B
    assert out_sub1.shape[0] == 3
    assert out_sub2.shape[0] == 1

    # The first 3 in full batch must be identical to out_sub
    assert jnp.allclose(
        out_full[:3], out_sub1, atol=1e-6
    ), "First 3 elements in full batch do not match sub-batch"

    # The first element in full batch must be identical to out_sub2
    assert jnp.allclose(
        out_full[0], out_sub2, atol=1e-6
    ), "First element in full batch does not match sub-batch"
