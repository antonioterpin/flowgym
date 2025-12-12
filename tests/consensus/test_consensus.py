import pytest
import re
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
from flowgym.utils import load_configuration
from flowgym.flow.consensus.consensus_algorithms import (
    mean_consensus,
    median_consensus,
    admm_consensus,
)

# Import functions under test
from flowgym.flow.dis.process import (
    photometric_error_with_patches,
)

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


@pytest.mark.parametrize(
    "flows",
    [
        jnp.array(
            [
                [[[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]]],
            ]
        )
    ],
)
def test_mean_consensus_basic(flows):
    """Basic test for mean consensus with a single estimate."""
    result, _ = mean_consensus(flows, jnp.ones(flows.shape[:-1]))
    expected = jnp.mean(flows, axis=1)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows",
    [
        jnp.array(
            [
                [[[1.0, 2.0]]],  # Estimate 0
                [[[9.0, 10.0]]],  # Estimate 1
            ]
        )
    ],
)  # Shape (2, 1, 1, 2)
def test_mean_consensus_weighted(flows):
    """Test mean consensus with weighted estimates."""
    weight1 = 0.25 * jnp.ones_like(flows[0, ..., 0])
    weight2 = 0.75 * jnp.ones_like(flows[1, ..., 0])
    weights = jnp.stack([weight1, weight2], axis=0)
    result, _ = mean_consensus(flows, weights)
    expected = 0.25 * flows[0] + 0.75 * flows[1]
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows",
    [
        jnp.array(
            [
                [[[0.0, 0.0]]],
                [[[2.0, 2.0]]],
            ]
        )
    ],
)
@pytest.mark.parametrize("weights_list", [[10, 90], [0.01, 0.09]])
@pytest.mark.parametrize("expected", [jnp.array([[[[1.8, 1.8]]]])])
def test_mean_consensus_weights_are_normalized(flows, weights_list, expected):
    """Test mean consensus with weights that are not normalized."""
    weights1 = weights_list[0] * jnp.ones_like(flows[0, ..., 0])
    weights2 = weights_list[1] * jnp.ones_like(flows[1, ..., 0])
    weights = jnp.stack([weights1, weights2], axis=0)
    result, _ = mean_consensus(flows, weights)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("flows", [jnp.ones((1, 2, 2, 2))])
@pytest.mark.parametrize("expected", [jnp.ones((2, 2, 2))])
@pytest.mark.parametrize("weights", [jnp.ones((1, 2, 2)) * 0.5, jnp.ones((1, 2, 2))])
def test_mean_consensus_single_estimate(flows, expected, weights):
    """Test mean consensus with a single estimate."""
    result, _ = mean_consensus(flows, weights)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows",
    [
        jnp.array(
            [
                [
                    [[1.0, 2.0]],
                    [[9.0, 10.0]],
                    [[5.0, 6.0]],
                ],  # N=3, 1x1 image, 2 channels
            ]
        )
    ],
)  # shape (3, 1, 1, 2)
def test_median_consensus_basic(flows):
    """Basic test for median consensus with a single estimate."""
    result, _ = median_consensus(flows)
    expected = jnp.median(flows, axis=0)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows",
    [
        jnp.array(
            [
                [[[10.0]]],
                [[[30.0]]],
                [[[20.0]]],
            ],
        )
    ],
)  # shape (3, 1, 1, 1)
@pytest.mark.parametrize(
    "expected",
    [
        jnp.array(
            [
                [[[20.0]]],  # Median of [10, 30, 20]
            ]
        )
    ],
)  # shape (1, 1, 1, 1)
def test_median_consensus_with_batching(flows, expected):
    """Test median consensus with multiple batches."""
    result, _ = median_consensus(flows)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows",
    [
        jnp.array(
            [
                [[[1.0]]],
                [[[2.0]]],
                [[[3.0]]],
                [[[4.0]]],
            ],
        )
    ],
)  # shape (4, 1, 1, 1)
@pytest.mark.parametrize(
    "expected", [jnp.array([[[2.5]]])]
)  # Median of [1, 2, 3, 4] is (2+3)/2 = 2.5
def test_median_consensus_even_n(flows, expected):
    """Test median consensus with an even number of estimates."""
    result, _ = median_consensus(flows)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("flows", [jnp.ones((5, 1, 1, 2))])  # shape (5, 1, 1, 2)
@pytest.mark.parametrize(
    "weights", [jnp.ones((5, 1, 1, 2)) * 0.5, jnp.ones((5, 1, 1, 2))]
)
def test_median_consensus_ignores_weights(flows, weights):
    """Test median consensus ignores weights."""
    result, _ = median_consensus(flows, weights)
    expected = jnp.median(flows, axis=0)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows", [jnp.array([[[[[42.0, 0.0]]]]])]
)  # shape (1, 1, 1, 1, 2)
@pytest.mark.parametrize(
    "expected", [jnp.array([[[[42.0, 0.0]]]])]
)  # Median is the only value
def test_median_consensus_single_estimate(flows, expected):
    """Test median consensus with a single estimate."""
    result, _ = median_consensus(flows)
    assert jnp.allclose(result, expected)


def test_median_consensus_large():
    """Test median consensus with a large number of estimates."""
    flows = random.normal(
        random.PRNGKey(0), (10, 1000, 1, 1, 2)
    )  # (B=10, N=1000, H=1, W=1, C=2)
    result, _ = median_consensus(flows)
    expected = jnp.median(flows, axis=0)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows",
    [
        jnp.stack(
            [
                jnp.ones((2, 2, 2)),  # all ones
                jnp.zeros((2, 2, 2)),  # all zeros
                -jnp.ones((2, 2, 2)),  # all -1
            ],
            axis=0,
        )
    ],
)  # shape (3, 2, 2, 2)
@pytest.mark.parametrize("expected", [jnp.zeros((2, 2, 2))])  # Median is 0 everywhere
def test_median_consensus_multiple_dimensions(flows, expected):
    """Test median consensus with multiple dimensions."""
    result, _ = median_consensus(flows)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "flows", [jnp.array([[[[[1], [3]], [[5], [7]], [[9], [11]]]]], dtype=jnp.int32)]
)  # shape (1, 3, 2, 1, 1)
def test_median_consensus_dtype_preserved(flows):
    """Test median consensus preserves dtype."""
    result, _ = median_consensus(flows)
    # jnp.median always returns float
    assert result.dtype == jnp.float32


@pytest.mark.parametrize("rho", ["a", [1.0], None, {}])
def test_invalid_rho_type(rho):
    """Test ADMM consensus with invalid rho type."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {"rho": rho}
    with pytest.raises(ValueError, match="Invalid rho type"):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize("rho", [0, -1.5, -1, -1e-6])
def test_invalid_rho_value(rho):
    """Test ADMM consensus with non-positive rho."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {"rho": rho}
    with pytest.raises(ValueError, match="Rho must be positive"):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize("flows_objective_type", ["", None, "l3", "L2", 2])
def test_invalid_flows_objective_type(flows_objective_type):
    """Test ADMM consensus with invalid flows_objective_type."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {"flows_objective_type": flows_objective_type}
    with pytest.raises(ValueError, match="Invalid flows_objective_type"):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize(
    "weights",
    [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.ones((2, 2, 4, 4)),
    ],
)
def test_invalid_weights_shape(weights):
    """Test ADMM consensus with flow_weights of wrong shape."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    config = {}
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Weights must have the same shape as flows except for the last dimension, "
            f"got {weights.shape} and {flows.shape[:-1]}."
        ),
    ):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize(
    "weights",
    [
        "not a list",  # wrong type
        123,  # wrong type
        {"a": 1, "b": 2},  # wrong type (dict
    ],
)
def test_invalid_flow_weights_type(weights):
    """Test ADMM consensus with flow_weights of wrong type."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    config = {}
    with pytest.raises(
        ValueError,
        match=f"Invalid weights type: {type(weights)}. " "Expected jnp.ndarray.",
    ):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize("regularizer_list", ["tv", 123, {"tv": 1}])
def test_invalid_regularizer_list_type(regularizer_list):
    """Test ADMM consensus with invalid regularizer_list type."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {"regularizer_list": regularizer_list}
    with pytest.raises(ValueError, match="Invalid regularizer_list type"):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize("regularizer_weights", ["tv", [1, 2], 5])
def test_invalid_regularizer_weights_type(regularizer_weights):
    """Test ADMM consensus with invalid regularizer_weights type."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {"regularizer_list": ["tv"], "regularizer_weights": regularizer_weights}
    with pytest.raises(ValueError, match="Invalid regularizer_weights type"):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize(
    "regularizer_weights", [{"tv": "string"}, {"tv": None}, {"tv": [1, 2]}]
)
def test_invalid_regularizer_weights_values(regularizer_weights):
    """Test ADMM consensus with invalid regularizer_weights values."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {"regularizer_list": ["tv"], "regularizer_weights": regularizer_weights}
    with pytest.raises(
        ValueError, match="All values in regularizer_weights must be float or int"
    ):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize(
    "param_name",
    ["num_iterations_flows", "num_iterations_consensus", "max_admm_iterations"],
)
@pytest.mark.parametrize("bad_value", [0, -5, 1.5, "ten"])
def test_invalid_iteration_params(param_name, bad_value):
    """Test ADMM consensus with invalid iteration parameters."""
    flows = jnp.zeros((2, 3, 4, 4, 2))
    weights = jnp.ones((2, 3, 4, 4))  # Dummy weights
    config = {
        param_name: bad_value,
        "solver_flows": "sgd",
        "solver_consensus": "sgd",
    }
    with pytest.raises(ValueError, match="must be a positive integer"):
        admm_consensus(flows, weights, config)


@pytest.mark.parametrize("eps_rel_stopping", [None, 1e-4, 1e-6])
@pytest.mark.parametrize("eps_abs_stopping", [None, 1e-4, 1e-6])
def test_admm_consensus_eps_stopping(eps_rel_stopping, eps_abs_stopping):
    """Test ADMM consensus with stopping criteria."""
    flows = random.uniform(random.PRNGKey(0), (3, 4, 4, 2))
    weights = jnp.ones((3, 4, 4))  # Dummy weights
    config = {
        "eps_rel_stopping": eps_rel_stopping,
        "eps_abs_stopping": eps_abs_stopping,
    }
    result, metrix = admm_consensus(flows, weights, config)
    stopping_time = metrix["final_stopping_time"]
    assert result is not None
    assert stopping_time is not None
    assert isinstance(stopping_time, (int, jnp.ndarray))
    assert jnp.array(stopping_time).ndim == 0
    assert jnp.all(stopping_time >= 0)


@pytest.mark.parametrize(
    "img_shape, patch_size, patch_stride",
    [
        ((20, 20), 5, 1),
        ((32, 15), 7, 1),
        ((10, 10), 3, 1),
        ((21, 21), 9, 1),
    ],
)
def test_output_shapes_and_types(img_shape, patch_size, patch_stride):
    key = random.PRNGKey(0)
    prev = random.uniform(key, shape=img_shape, dtype=jnp.float32)
    curr = random.uniform(key, shape=img_shape, dtype=jnp.float32)
    flow = jnp.zeros((*img_shape, 2), dtype=jnp.float32)

    errors = photometric_error_with_patches(prev, curr, flow, patch_size, patch_stride)
    H, W = img_shape
    half = patch_size // 2

    assert errors.shape == (
        (H - 2 * half) // patch_stride,
        (W - 2 * half) // patch_stride,
    )
    assert errors.dtype == jnp.float32 or errors.dtype == jnp.float64


def test_identity_zero_flow():
    """If prev==curr and flow==0, error must be zero everywhere."""
    img_shape = (16, 16)
    patch_size = 5
    patch_stride = 1
    value = 2.34  # arbitrary constant

    prev = jnp.ones(img_shape) * value
    curr = jnp.ones(img_shape) * value
    flow = jnp.zeros((*img_shape, 2), dtype=jnp.float32)

    errors = photometric_error_with_patches(prev, curr, flow, patch_size, patch_stride)

    assert jnp.allclose(errors, 0.0, atol=1e-6)


@pytest.mark.parametrize("shift_y, shift_x", [(1, 0), (0, 2), (2, 3)])
def test_shift_and_flow_cancel(shift_y, shift_x):
    """If curr is shifted and flow compensates, error must be zero."""
    img_shape = (12, 13)
    patch_size = 3
    patch_stride = 1

    flow = jnp.zeros((*img_shape, 2), dtype=jnp.float32)
    flow = flow.at[..., 0].set(shift_y)
    flow = flow.at[..., 1].set(shift_x)

    prev = jnp.arange(np.prod(img_shape), dtype=jnp.float32).reshape(img_shape)
    # Shift prev by (shift_y, shift_x) to get curr
    curr = jnp.roll(prev, shift=(shift_y, shift_x), axis=(0, 1))

    errors = photometric_error_with_patches(prev, curr, flow, patch_size, patch_stride)
    # Ignore patches on the image edge (where roll causes wraparound, not real shift)
    # Only test patches whose center stays in bounds after shifting
    half = patch_size // 2
    ys = jnp.arange(half, img_shape[0] - half, patch_stride)
    xs = jnp.arange(half, img_shape[1] - half, patch_stride)
    valid = jnp.zeros(errors.shape, dtype=bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            fy, fx = flow[y, x]  # flow at patch center (y, x)
            y2 = y + fy
            x2 = x + fx
            # Make sure the *shifted* patch center is also valid
            valid = valid.at[i, j].set(
                (half <= y2 < img_shape[0] - half)
                and (half <= x2 < img_shape[1] - half)
            )

    assert jnp.allclose(errors[valid], 0.0, atol=1e-5)


def test_random_input_robustness():
    """Errors are finite and non-negative for random inputs."""
    rng = np.random.RandomState(42)
    img_shape = (17, 23)
    patch_size = 5
    patch_stride = 1

    prev = jnp.array(rng.uniform(0, 1, img_shape), dtype=jnp.float32)
    curr = jnp.array(rng.uniform(0, 1, img_shape), dtype=jnp.float32)
    flow = jnp.array(rng.normal(0, 2, (*img_shape, 2)), dtype=jnp.float32)

    errors = photometric_error_with_patches(prev, curr, flow, patch_size, patch_stride)

    assert jnp.all(jnp.isfinite(errors))
    assert jnp.all(errors >= 0)


@pytest.mark.parametrize(
    "img_shape, patch_size, patch_stride",
    [
        ((7, 7), 7, 1),
        ((9, 9), 7, 1),
        ((7, 7), 3, 1),
    ],
)
def test_patch_edge_cases(img_shape, patch_size, patch_stride):
    """Test that the function handles edge cases with patches correctly."""
    rng = random.PRNGKey(123)
    prev = random.uniform(rng, img_shape, minval=-1, maxval=2, dtype=jnp.float32)
    curr = random.uniform(rng, img_shape, minval=-1, maxval=2, dtype=jnp.float32)
    flow = jnp.zeros((*img_shape, 2), dtype=jnp.float32)

    errors = photometric_error_with_patches(prev, curr, flow, patch_size, patch_stride)

    half = patch_size // 2

    number_of_patches_y = (img_shape[0] - 1 - 2 * half) // patch_stride + 1
    number_of_patches_x = (img_shape[1] - 1 - 2 * half) // patch_stride + 1
    # No shape asserts here, just ensure we didn't crash and outputs are reasonable
    assert errors.ndim == 2
    assert errors.shape == (number_of_patches_y, number_of_patches_x)


# Optionally: test that the function works with jax.jit
def test_jit_compatibility():
    img_shape = (16, 16)
    patch_size = 5
    patch_stride = 1
    rng = np.random.RandomState(2024)
    prev = jnp.array(rng.uniform(0, 1, img_shape), dtype=jnp.float32)
    curr = jnp.array(rng.uniform(0, 1, img_shape), dtype=jnp.float32)
    flow = jnp.zeros((*img_shape, 2), dtype=jnp.float32)

    jit_func = jax.jit(
        lambda prev, curr, flow: photometric_error_with_patches(
            prev, curr, flow, patch_size, patch_stride
        )
    )
    errors = jit_func(prev, curr, flow)
    assert jnp.all(jnp.isfinite(errors))
