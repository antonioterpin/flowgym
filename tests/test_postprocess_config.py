import pytest
import timeit
import jax
from jax import random as jrandom, numpy as jnp
from flowgym.flow.base import FlowFieldEstimator
from flowgym.make import compile_model
from flowgym.flow.postprocess import (
    tile_average_interpolation,
    laplace_interpolation,
    median_smoothing,
    average_smoothing,
    gaussian_smoothing,
    constant_threshold_filter,
    adaptive_global_filter,
    adaptive_local_filter,
    universal_median_test,
    quantize,
    resize_flow,
    resize_flow_validate_params,
    temporal_smoothing_ema_validate_params,
    temporal_smoothing_ema,
)

from flowgym.utils import load_configuration

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


class DummyEstimator(FlowFieldEstimator):
    def _estimate(
        self,
        image: jnp.ndarray,
        _,
        __,
        ___,
    ) -> tuple[jnp.ndarray, dict, dict]:
        return jnp.tile(image[..., None], (1, 1, 1, 2)), {}, {}


all_post = [
    (tile_average_interpolation, {"radius": 1}),
    (tile_average_interpolation, {"radius": 2}),
    (laplace_interpolation, {"num_iter": 2}),
    (laplace_interpolation, {"num_iter": 4}),
    (average_smoothing, {"radius": 1}),
    (average_smoothing, {"radius": 2}),
    (median_smoothing, {"radius": 1}),
    (median_smoothing, {"radius": 2}),
    (gaussian_smoothing, {"sigma": 1.0}),
    (gaussian_smoothing, {"sigma": 2.0, "truncate": 6.0}),
    (gaussian_smoothing, {"sigma": 3.0, "truncate": 8.0}),
    (constant_threshold_filter, {"vel_min": 0.0, "vel_max": 2.0}),
    (constant_threshold_filter, {"vel_min": 1.0, "vel_max": 3.0}),
    (adaptive_global_filter, {"n_sigma": 1.0}),
    (adaptive_global_filter, {"n_sigma": 2.0}),
    (adaptive_local_filter, {"n_sigma": 1.0, "radius": 2}),
    (adaptive_local_filter, {"n_sigma": 2.0, "radius": 3}),
    (universal_median_test, {"r_threshold": 1.0, "epsilon": 0.01, "radius": 2}),
    (universal_median_test, {"r_threshold": 2.0, "epsilon": 0.001, "radius": 3}),
]

bad_params = [
    (tile_average_interpolation, {"radius": -1}),
    (tile_average_interpolation, {"radius": 0}),
    (laplace_interpolation, {"num_iter": 0}),
    (laplace_interpolation, {"num_iter": -1}),
    (average_smoothing, {"radius": -1}),
    (average_smoothing, {"radius": 0}),
    (median_smoothing, {"radius": -1}),
    (median_smoothing, {"radius": 0}),
    (gaussian_smoothing, {"sigma": -1.0}),
    (gaussian_smoothing, {"sigma": 0.0}),
    (gaussian_smoothing, {"sigma": 0.5, "truncate": -1.0}),
    (gaussian_smoothing, {"sigma": 0.5, "truncate": 0.0}),
    (constant_threshold_filter, {"vel_min": 2.0, "vel_max": 1.0}),
    (constant_threshold_filter, {"vel_min": -1.0, "vel_max": -2.0}),
    (adaptive_global_filter, {"n_sigma": -1.0}),
    (adaptive_global_filter, {"n_sigma": 0.0}),
    (adaptive_local_filter, {"n_sigma": -1.0, "radius": 2}),
    (adaptive_local_filter, {"n_sigma": 0.0, "radius": 2}),
    (adaptive_local_filter, {"n_sigma": 1.0, "radius": -1}),
    (adaptive_local_filter, {"n_sigma": 1.0, "radius": 0}),
    (universal_median_test, {"r_threshold": -1.0, "epsilon": 0.01, "radius": 2}),
    (universal_median_test, {"r_threshold": 0.0, "epsilon": 0.01, "radius": 2}),
    (universal_median_test, {"r_threshold": 1.0, "epsilon": -0.01, "radius": 2}),
    (universal_median_test, {"r_threshold": 1.0, "epsilon": 0.01, "radius": -1}),
    (universal_median_test, {"r_threshold": 1.0, "epsilon": 0.01, "radius": 0}),
    (
        quantize,
        {
            "min_val": jnp.array([-1.0, -1.0]),
            "max_val": jnp.array([1.0, 1.0]),
            "dtype": jnp.float16,
        },
    ),
    (
        quantize,
        {
            "min_val": jnp.array([-1.0, -1.0]),
            "max_val": jnp.array([1.0, 1.0]),
            "dtype": jnp.uint64,
        },
    ),
    (
        quantize,
        {
            "min_val": jnp.array([-1.0, -1.0]),
            "max_val": jnp.array([1.0, 1.0]),
            "dtype": jnp.float32,
        },
    ),
    (quantize, {"min_val": jnp.array([-1.0, -1.0]), "max_val": jnp.array([1.0, 1.0])}),
]


def apply_from_idxs(idxs, flow):
    """Apply postprocessing functions from given indices."""
    valid = jnp.ones_like(flow[..., 0], dtype=bool)
    for idx in idxs:
        flow, valid, _ = all_post[idx][0](
            flow, valid=valid, state=None, **all_post[idx][1]
        )
    return flow


@pytest.mark.parametrize("B", [1, 16])
@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
@pytest.mark.parametrize("seed", [42, 123, 456, 789, 1, 2, 3, 43, 44, 45])
def test_postprocess_config(B, N, seed):
    # Sample idxs
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    idxs = jrandom.choice(key, jnp.arange(len(all_post)), shape=(N,), replace=True)

    # Create a random flow
    image = jrandom.uniform(
        subkey, (B, 64, 64), minval=0, maxval=255, dtype=jnp.float32
    )

    # Apply postprocessing functions
    flow_raw = jnp.tile(image[..., None], (1, 1, 1, 2))
    flow_reference = apply_from_idxs(idxs, flow_raw)

    # Create a post_processing configuration
    post_process_config = {
        "postprocessing_steps": [
            {"name": all_post[idx][0].__name__} | all_post[idx][1] for idx in idxs
        ]
    }
    # Instantiate the dummy estimator
    model = DummyEstimator.from_config(post_process_config)
    trainable_state = model.create_trainable_state(image, key)
    create_state_fn, compute_estimate_fn = compile_model(model, flow_reference, False)
    # Create the state
    state = create_state_fn(image, key)
    state, _ = compute_estimate_fn(image, state, trainable_state)
    flow = state["estimates"][:, -1, ...]

    # Check if the processed image matches the reference
    assert jnp.allclose(
        flow, flow_reference, atol=1e-5
    ), "Processed image does not match reference after applying preprocessing functions"


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
@pytest.mark.parametrize("seed", [42, 123, 456, 789, 1, 2, 3, 43, 44, 45])
def test_preprocess_config_invalid_params(N, seed):
    # Sample idxs
    key = jrandom.PRNGKey(seed)
    idxs = jrandom.choice(key, jnp.arange(len(bad_params)), shape=(N,), replace=True)

    # Create a post_processing configuration
    post_process_config = {
        "postprocessing_steps": [
            {"name": bad_params[idx][0].__name__} | bad_params[idx][1] for idx in idxs
        ]
    }
    with pytest.raises((ValueError, TypeError)):
        # Instantiate the dummy estimator
        DummyEstimator.from_config(post_process_config)


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "B, H, time_limit",
    [
        (1, 64, 9.2e-4),
        (64, 64, 1.6e-3),
        (1, 128, 9.5e-4),
        (64, 128, 5e-3),
        (1, 256, 1.2e-3),
        (64, 256, 2e-2),
        (1, 1024, 6e-3),
    ],
)
@pytest.mark.parametrize("seed", [42])
def test_postprocess_jit(B, H, time_limit, seed):
    # Sample idxs
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)

    # Create a random image
    image = jrandom.uniform(subkey, (B, H, H), minval=0, maxval=255, dtype=jnp.float32)

    # Create a post_processing configuration
    post_process_config = {
        "postprocessing_steps": [
            {
                "name": "universal_median_test",
                "r_threshold": 2.0,
                "epsilon": 0.01,
                "radius": 2,
            },
            {"name": "laplace_interpolation", "num_iter": 512},
            {"name": "median_smoothing", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
            {"name": "median_smoothing", "radius": 1},
            {"name": "average_smoothing", "radius": 1},
            {"name": "temporal_smoothing_ema", "alpha": 0.5},
        ]
    }
    # Instantiate the dummy estimator
    model = DummyEstimator.from_config(post_process_config)
    trainable_state = model.create_trainable_state(image, key=key)
    create_state_fn, compute_estimate_fn = compile_model(
        model, jnp.zeros(image.shape + (2,)), True
    )
    # Warm up
    state = create_state_fn(image, key)
    state, _ = compute_estimate_fn(image, state, trainable_state)

    def fn():
        state = create_state_fn(image, key)
        state, _ = compute_estimate_fn(image, state, trainable_state)
        flow = state["estimates"][:, -1, ...]
        flow.block_until_ready()

    runtime = timeit.repeat(
        fn,
        repeat=REPETITIONS,
        number=NUMBER_OF_EXECUTIONS,
    )
    runtime = min(runtime) / NUMBER_OF_EXECUTIONS

    assert (
        runtime < time_limit
    ), f"JIT preprocessing took too long: {runtime:.6f} seconds per execution"


def test_quantize_basic_range():
    """Test quantization of flow field within expected range."""
    flow_field = jnp.array([[[-1.0, 1.0], [-1.0, -1.0]]])
    quantized, _, _ = quantize(
        flow_field, min_val=jnp.array([-2.0]), max_val=jnp.array([2.0]), dtype=jnp.uint8
    )

    expected = jnp.array([[[63, 191], [63, 63]]], dtype=jnp.uint8)

    assert jnp.allclose(
        quantized, expected, atol=1
    ), "Quantization values are incorrect"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_below_min():
    """Test quantization clips values below -max_speed."""
    flow_field = jnp.array([[[-15.0, -10.0], [-20.0, -12.0]]])
    quantized, _, _ = quantize(
        flow_field, min_val=jnp.array([-2.0]), max_val=jnp.array([2.0]), dtype=jnp.uint8
    )

    # All values below -2 should be clipped to 0
    expected = jnp.array([[[0, 0], [0, 0]]], dtype=jnp.uint8)

    assert jnp.all(quantized == expected), "Values below min_speed should clip to 0"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_above_max():
    """Test quantization clips values above max_speed."""
    flow_field = jnp.array([[[0.0, 20.0], [0.0, 25.0]]])
    quantized, _, _ = quantize(
        flow_field, min_val=jnp.array([-2.0]), max_val=jnp.array([2.0]), dtype=jnp.uint8
    )

    # All values above 10 should be clipped to 255
    expected = jnp.array([[[127, 255], [127, 255]]], dtype=jnp.uint8)

    assert jnp.all(quantized == expected), "Values above max_speed should clip to max"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_zero_flow():
    """Test quantization of zero flow field."""
    flow_field = jnp.array([[[0.0, 0.0], [0.0, 0.0]]])
    quantized, _, _ = quantize(
        flow_field, min_val=jnp.array([-2.0]), max_val=jnp.array([2.0]), dtype=jnp.uint8
    )

    expected = 127 * jnp.ones(flow_field.shape, dtype=jnp.uint8)

    assert jnp.all(quantized == expected), "Zero flow should map to middle range"
    assert quantized.dtype == jnp.uint8, "Output should be uint8"


def test_quantize_resolution_levels_effect():
    """Test different resolution levels affect quantization steps."""
    flow_field = jnp.array([5.0])
    quantized_low, _, _ = quantize(
        flow_field,
        min_val=jnp.array([-10.0]),
        max_val=jnp.array([10.0]),
        dtype=jnp.uint8,
    )
    quantized_high, _, _ = quantize(
        flow_field,
        min_val=jnp.array([-10.0]),
        max_val=jnp.array([10.0]),
        dtype=jnp.uint16,
    )

    # For uint8, range is 0-255, so 5 maps to 191
    assert quantized_low[0] == 191, "Low resolution quantization incorrect"
    # For uint16, range is 0-65535, so 5 maps to 49151
    assert quantized_high[0] == 49151, "High resolution quantization incorrect"
    assert quantized_low.dtype == jnp.uint8, "Low resolution output should be uint8"
    assert quantized_high.dtype == jnp.uint16, "High resolution output should be int16"


def test_quantize_max_speed_effect():
    """Test different max_speed values affect quantization range."""
    flow_field = jnp.array([5.0])
    quantized_low, _, _ = quantize(
        flow_field, min_val=jnp.array([-5.0]), max_val=jnp.array([5.0]), dtype=jnp.uint8
    )
    quantized_high, _, _ = quantize(
        flow_field,
        min_val=jnp.array([-20.0]),
        max_val=jnp.array([20.0]),
        dtype=jnp.uint8,
    )

    assert quantized_low[0] == 255, "Low max_speed quantization incorrect"
    assert quantized_high[0] == 159, "High max_speed quantization incorrect"
    assert quantized_low.dtype == jnp.uint8, "Output should be uint8"
    assert quantized_high.dtype == jnp.uint8, "Output should be uint8"


@pytest.mark.parametrize("B", [1, 16, 256])
@pytest.mark.parametrize("H", [64, 128, 256, 1024])
@pytest.mark.parametrize("W", [64, 128, 256, 1024])
@pytest.mark.parametrize("seed", [42])
def test_resize_flow(B, H, W, seed):
    """Test resizing flow fields."""
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)

    # Create a random flow field
    flow = jrandom.uniform(subkey, (B, H, W, 2), minval=-1.0, maxval=1.0)

    # Resize to a smaller shape
    target_shape = (H // 2, W // 4)
    resized_flow, _, _ = resize_flow(
        flow=flow,
        target_height=target_shape[0],
        target_width=target_shape[1],
        valid=None,
    )

    assert resized_flow.shape == (
        B,
        target_shape[0],
        target_shape[1],
        2,
    ), (
        "Resized flow shape mismatch: "
        f"{resized_flow.shape} != {(B, target_shape[0], target_shape[1], 2)}"
    )


@pytest.mark.parametrize(
    "target_shape",
    [
        (0, 1),
        (-1, 2),
        (1.5, 2),
    ],
)
def test_resize_flow_validate_params_invalid(target_shape):
    with pytest.raises(ValueError):
        resize_flow_validate_params(target_shape[0], target_shape[1])


@pytest.mark.parametrize(
    "target_shape",
    [
        (1, 1),
        (10, 20),
    ],
)
def test_resize_flow_validate_params_valid(target_shape):
    # should not raise
    resize_flow_validate_params(target_shape[0], target_shape[1])


class DummyState(dict):
    def __init__(self, history_estimates):
        # history_estimates: jnp.ndarray of shape (batch, time, ...)
        self["estimates"] = history_estimates


class TestValidateParams:
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_valid_alphas_do_not_raise(self, alpha):
        # should not raise for floats in [0,1]
        temporal_smoothing_ema_validate_params(float(alpha))

    @pytest.mark.parametrize("alpha", [1, "0.5", None, [], {}])
    def test_non_float_alpha_raises(self, alpha):
        with pytest.raises(ValueError) as exc:
            temporal_smoothing_ema_validate_params(alpha)
        assert "alpha must be a float" in str(exc.value)

    @pytest.mark.parametrize("alpha", [-0.1, -1.0, 1.1, 2.0])
    def test_out_of_range_alpha_raises(self, alpha):
        with pytest.raises(ValueError) as exc:
            temporal_smoothing_ema_validate_params(float(alpha))
        assert "alpha must be in the range [0, 1]" in str(exc.value)


class TestTemporalSmoothingEma:
    @pytest.fixture
    def dummy_batch(self):
        # batch size = 2, image size = 1×1 for simplicity
        flow = jnp.array([[[[1.0]]], [[[2.0]]]])  # shape (2, 1, 1, 1)
        # history_estimates for last time step
        hist = jnp.array([[[[10.0]]], [[[20.0]]]])  # shape (2, 1, 1, 1)
        state = DummyState(history_estimates=jnp.concatenate([hist, hist], axis=1))
        valid_mask = jnp.array([[[1]], [[0]]])  # shape (2, 1, 1)
        return flow, state, valid_mask

    def test_ema_formula_without_valid(self, dummy_batch):
        flow, state, valid = dummy_batch
        alpha = 0.25

        # Call with no valid mask
        smoothed, returned_valid, returned_state = temporal_smoothing_ema(
            flow, alpha, state
        )

        # expected = alpha * flow + (1-alpha)*prev_estimate
        prev = state["estimates"][:, -1]
        expected = alpha * flow + (1 - alpha) * prev

        assert jnp.allclose(smoothed, expected)
        # when no valid is passed, valid should be None
        assert returned_valid is None
        # state is returned unchanged
        assert returned_state is state

    def test_ema_formula_with_valid(self, dummy_batch):
        flow, state, valid = dummy_batch
        alpha = 0.6

        smoothed, returned_valid, returned_state = temporal_smoothing_ema(
            flow, alpha, state, valid=valid
        )

        prev = state["estimates"][:, -1]
        expected = alpha * flow + (1 - alpha) * prev

        assert jnp.allclose(smoothed, expected)
        # valid mask should be passed through
        assert returned_valid is not None
        assert jnp.array_equal(returned_valid, valid)
        assert returned_state is state

    @pytest.mark.parametrize("alpha", [0.0, 1.0])
    def test_edge_alpha_values(self, dummy_batch, alpha):
        # alpha=0 => smoothed == prev_estimate; alpha=1 => smoothed == flow
        flow, state, valid = dummy_batch
        smoothed, _, _ = temporal_smoothing_ema(flow, alpha, state, valid=None)
        prev = state["estimates"][:, -1]

        if alpha == 0.0:
            assert jnp.allclose(smoothed, prev)
        else:
            assert jnp.allclose(smoothed, flow)
