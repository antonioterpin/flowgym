import numpy as np
import pytest
import jax.numpy as jnp
import jax.random as jrandom
import jax
from jax import jit
import timeit
from functools import partial

from flowgym.flow.postprocess import average_smoothing, median_smoothing
from flowgym.utils import load_configuration

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


def _numpy_average_smoothing(flow: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    pad = radius
    padded = np.pad(flow, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(flow)
    B, H, W, C = flow.shape

    for b in range(B):
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    window = padded[b, y : y + k, x : x + k, c]
                    out[b, y, x, c] = window.mean()
    return out


def _numpy_median_smoothing(flow: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    pad = radius
    padded = np.pad(
        flow,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    out = np.empty_like(flow)
    B, H, W, C = flow.shape

    for b in range(B):
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    window = padded[b, y : y + k, x : x + k, c]
                    out[b, y, x, c] = np.median(window)
    return out


@pytest.mark.parametrize(
    "method, method_naive, kwargs",
    [
        (
            average_smoothing,
            _numpy_average_smoothing,
            {"radius": 1},
        ),
        (
            average_smoothing,
            _numpy_average_smoothing,
            {"radius": 3},
        ),
        (
            median_smoothing,
            _numpy_median_smoothing,
            {"radius": 1},
        ),
        (
            median_smoothing,
            _numpy_median_smoothing,
            {"radius": 3},
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(1, 5, 5, 2), (2, 7, 8, 2), (3, 64, 64, 2), (4, 128, 128, 2), (5, 256, 256, 2)],
)
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_methods_match_naive_version(seed, shape, method, method_naive, kwargs):
    """Average filter should match naïve NumPy reference within tolerance."""
    rng = np.random.default_rng(seed)
    flow_np = rng.standard_normal(size=shape).astype(np.float32)
    expected = method_naive(flow_np, **kwargs)
    result, _, _ = method(jnp.asarray(flow_np), **kwargs)
    result = np.array(result)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "B, H, W, radius, limit_time",
    [
        (1, 64, 64, 1, 4e-5),
        (512, 64, 64, 2, 8.5e-5),
        (1, 256, 256, 1, 4e-5),
        (512, 256, 256, 2, 1.5e-3),
        (1, 1024, 1024, 1, 6e-5),
        (1, 1024, 1024, 2, 6e-5),
        (32, 1024, 1024, 1, 1.3e-3),
        (32, 1024, 1024, 2, 1.3e-3),
    ],
)
def test_average_time(B, H, W, radius, limit_time):
    """Just makes sure compilation/shape handling hold for >1M voxels."""
    rng = jrandom.PRNGKey(0)
    flow_field = jrandom.normal(rng, (B, H, W, 2))

    fn = jit(partial(average_smoothing, radius=radius))
    flow, _, _ = fn(flow_field)  # should compile & run without raising
    assert flow.shape == (B, H, W, 2)

    def run_jit():
        flow, _, _ = fn(flow_field)
        flow.block_until_ready()

    # Warm up the JIT compilation
    run_jit()

    # Measure the time taken for multiple executions
    total_time_jit = timeit.repeat(
        stmt=run_jit,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "B, H, W, radius, limit_time",
    [
        (1, 64, 64, 1, 5.2e-5),
        (128, 64, 64, 2, 3.2e-4),
        (1, 256, 256, 1, 5.5e-5),
        (128, 256, 256, 2, 4.5e-3),
        (1, 1024, 1024, 1, 2.5e-4),
        (1, 1024, 1024, 2, 6.5e-4),
        (32, 1024, 1024, 1, 7.9e-3),
        (16, 1024, 1024, 2, 9e-3),
    ],
)
def test_median_time(B, H, W, radius, limit_time):
    """Just makes sure compilation/shape handling hold for >1M voxels."""
    rng = jrandom.PRNGKey(0)
    flow_field = jrandom.normal(rng, (B, H, W, 2))

    fn = jit(partial(median_smoothing, radius=radius))
    flow, _, _ = fn(flow_field)  # should compile & run without raising
    assert flow.shape == (B, H, W, 2)

    def run_jit():
        flow, _, _ = fn(flow_field)
        flow.block_until_ready()

    # Warm up the JIT compilation
    run_jit()

    # Measure the time taken for multiple executions
    total_time_jit = timeit.repeat(
        stmt=run_jit,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"
