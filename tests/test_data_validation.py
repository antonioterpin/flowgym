import pytest
import timeit
import jax
from jax import jit
from functools import partial
import jax.random as jrandom
import jax.numpy as jnp
import numpy as np
from flowgym.flow.postprocess import (
    constant_threshold_filter,
    adaptive_global_filter,
    adaptive_local_filter,
    universal_median_test,
)
from flowgym.utils import load_configuration

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


@pytest.mark.parametrize(
    "vel_min, vel_max",
    [
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 2.0),
    ],
)
@pytest.mark.parametrize("B", [1, 16])
@pytest.mark.parametrize(
    "H, W",
    [
        (64, 64),
        (32, 64),
        (1024, 2048),
    ],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_constant_threshold_filter_batch(
    vel_min: float,
    vel_max: float,
    B: int,
    H: int,
    W: int,
    seed: int,
):
    """Test constant threshold filter with batched input."""
    # random flow field
    rng = jrandom.PRNGKey(seed)
    flow_field = jrandom.normal(rng, (B, H, W, 2))

    # apply the filter
    flow_field, mask, _ = constant_threshold_filter(
        flow_field, vel_min, vel_max, valid=None, state=None
    )

    # check the return without batching
    for b in range(B):
        assert mask is not None, f"Mask is None for batch {b}"
        # check the shape of the mask
        assert mask[b].shape == (
            H,
            W,
        ), f"Mask shape mismatch for batch {b}: {mask[b].shape}"

        # check that the mask is boolean
        assert (
            mask[b].dtype == jnp.bool_
        ), f"Mask dtype mismatch for batch {b}: {mask[b].dtype}"

        # check that the mask has no NaNs
        assert not jnp.isnan(mask[b]).any(), f"Mask contains NaNs for batch {b}"

        # check that all the True values in the mask correspond to outliers
        mag = jnp.linalg.norm(flow_field[b], axis=-1)
        assert jnp.all(
            (mag[mask[b]] < vel_min) | (mag[mask[b]] > vel_max)
        ), f"Mask does not correctly identify outliers for batch {b}"

        # check that all the False values in the mask correspond to inliers
        assert jnp.all(
            (mag[~mask[b]] >= vel_min) & (mag[~mask[b]] <= vel_max)
        ), f"Mask does not correctly identify inliers for batch {b}"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "B, H, W, limit_time",
    [
        (512, 64, 64, 5.5e-5),
        (512, 256, 256, 1.3e-3),
        (1, 1024, 1024, 5.1e-5),
        (64, 1024, 1024, 2.6e-3),
        (128, 1024, 1024, 0.005),
        (256, 1024, 1024, 0.0096),
        (512, 1024, 1024, 0.019),
    ],
)
def test_constant_threshold_filter_time(B, H, W, limit_time):
    """Just makes sure compilation/shape handling hold for >1M voxels."""
    rng = jrandom.PRNGKey(0)
    flow_field = jrandom.normal(rng, (B, H, W, 2))

    fn = jit(partial(constant_threshold_filter, vel_min=0.3, vel_max=0.7))
    flow_field, mask, _ = fn(flow_field)  # should compile & run without raising
    assert mask.shape == (B, H, W)

    def run_jit():
        _, mask, _ = fn(flow_field)
        mask.block_until_ready()

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


def test_adaptive_global_filter_zero_variance():
    # all the same magnitude, sigma=0 threshold=mean
    u = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    u, mask, _ = adaptive_global_filter(u, n_sigma=1.0, valid=None, state=None)
    # all have magnitude 1 == mean; none outside
    assert mask is not None
    assert not mask.any()


def _naive_local_filter(
    flow_field: jnp.ndarray | np.ndarray, n_sigma: float, radius: int = 1
) -> jnp.ndarray:
    magnitudes = jnp.linalg.norm(flow_field, axis=-1)
    B, H, W = magnitudes.shape
    wsize = 2 * radius + 1
    mask = np.ones((wsize, wsize), dtype=bool)
    mask[radius, radius] = False

    outliers = np.zeros((B, H, W), dtype=bool)

    for b in range(B):
        for y in range(H):
            for x in range(W):
                patch = np.zeros((wsize, wsize), dtype=flow_field.dtype)
                for dy in range(wsize):
                    for dx in range(wsize):
                        yy, xx = y + dy - radius, x + dx - radius
                        if 0 <= yy < H and 0 <= xx < W:
                            patch[dy, dx] = magnitudes[b, yy, xx]
                mu = patch.mean()
                sigma = patch.std()
                thr_low, thr_high = mu - n_sigma * sigma, mu + n_sigma * sigma
                centre_val = magnitudes[b, y, x]
                outliers[b, y, x] = centre_val < thr_low or centre_val > thr_high

    return jnp.asarray(outliers)


@pytest.mark.parametrize("shape", [(1, 8, 8), (2, 8, 8), (3, 5, 7)])
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("n_sigma", [1.0, 2.5])
@pytest.mark.parametrize("seed", [0, 42])
def test_local_std_vs_naive(
    shape: tuple[int, int, int], radius: int, n_sigma: float, seed: int
):
    """Adaptive filter should match the naive reference implementation."""
    B, H, W = shape
    key = jrandom.PRNGKey(seed)
    flow = jrandom.normal(key, (B, H, W, 2))

    expected = _naive_local_filter(flow, n_sigma, radius)
    flow, got, _ = adaptive_local_filter(flow, n_sigma, radius, valid=None, state=None)

    assert got is not None, "Mask is None"
    assert got.shape == expected.shape, "Output shape mismatch"
    assert jnp.array_equal(got, expected), "Naive and JAX results differ"


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "B, H, W, radius, limit_time",
    [
        (512, 64, 64, 1, 4e-4),
        (512, 64, 64, 2, 9e-4),
        (512, 256, 256, 1, 6.3e-3),
        (512, 256, 256, 2, 1.4e-2),
        (1, 1024, 1024, 1, 1.5e-4),
        (1, 1024, 1024, 2, 5e-4),
        (32, 1024, 1024, 1, 6.3e-3),
        (32, 1024, 1024, 2, 1.4e-2),
    ],
)
def test_adaptive_threshold_local_filter_time(B, H, W, radius, limit_time):
    """Just makes sure compilation/shape handling hold for >1M voxels."""
    rng = jrandom.PRNGKey(0)
    flow_field = jrandom.normal(rng, (B, H, W, 2))

    fn = jit(
        partial(
            adaptive_local_filter, n_sigma=2.0, radius=radius, valid=None, state=None
        )
    )
    flow_field, mask, _ = fn(flow_field)  # should compile & run without raising
    assert mask.shape == (B, H, W)

    def run_jit():
        _, mask, _ = fn(flow_field)
        mask.block_until_ready()

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


def test_universal_median_test_center_outlier():
    # zero field, but put a spike of magnitude 10 at center
    u = jnp.zeros((5, 5, 2))
    u = u.at[2, 2].set(jnp.array([10.0, 0.0]))
    _, mask, _ = universal_median_test(
        u[None, ...], r_threshold=2.0, epsilon=0.1, valid=None, state=None
    )
    assert mask is not None
    mask = mask[0, ...]
    # only the center should be flagged
    assert mask.shape == (5, 5)
    assert mask[2, 2]
    # no other pixel
    mask = mask.at[2, 2].set(False)
    assert not mask.any()


def _naive_median_test(
    flow_field: np.ndarray,
    r_threshold: float = 2.0,
    epsilon: float = 1e-1,
    radius: int = 1,
) -> np.ndarray:
    B, H, W, C = flow_field.shape
    wsize = 2 * radius + 1
    mask = np.ones((wsize, wsize), dtype=bool)
    mask[radius, radius] = False

    outlier = np.zeros((B, H, W), dtype=bool)

    for b in range(B):
        for y in range(H):
            for x in range(W):
                patch = np.zeros((wsize, wsize, C), dtype=flow_field.dtype)
                for dy in range(wsize):
                    for dx in range(wsize):
                        yy, xx = y + dy - radius, x + dx - radius
                        if 0 <= yy < H and 0 <= xx < W:
                            patch[dy, dx, :] = flow_field[b, yy, xx, :]

                neigh = patch[mask].reshape(-1, C)
                median = np.median(neigh, axis=0)
                rm = np.median(np.abs(neigh - median), axis=0)
                r0 = np.abs(patch[radius, radius, :] - median) / (rm + epsilon)
                outlier[b, y, x] = np.any(r0 > r_threshold)

    return outlier


@pytest.mark.parametrize(
    "batch,height,width,radius",
    [
        (1, 7, 8, 1),
        (2, 6, 5, 2),
        (3, 4, 4, 1),
        (2, 16, 16, 1),
        (2, 32, 32, 2),
    ],
)
@pytest.mark.parametrize("r_threshold", [1.5, 2.0])
def test_universal_vs_naive(batch, height, width, radius, r_threshold):
    rng = jrandom.PRNGKey(batch * height * width)  # deterministic seed
    ff_jax = jrandom.normal(rng, (batch, height, width, 2))

    # JAX → NumPy for the reference
    ff_np = np.asarray(ff_jax)

    expected = _naive_median_test(ff_np, r_threshold=r_threshold, radius=radius)
    ff_jax, actual, _ = universal_median_test(
        ff_jax, r_threshold=r_threshold, radius=radius, valid=None, state=None
    )

    # Move JAX result to host memory for comparison
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "B, H, W, limit_time, radius",
    [
        (1, 64, 64, 6.7e-5, 1),
        (512, 64, 64, 7.5e-4, 1),
        (1, 256, 256, 8.8e-5, 1),
        (128, 256, 256, 3e-3, 1),
        (1, 1024, 1024, 4e-4, 1),
        (32, 1024, 1024, 1.02e-2, 1),
        (1, 64, 64, 6.9e-5, 2),
        (512, 64, 64, 1.7e-3, 2),
        (1, 256, 256, 7.7e-5, 2),
        (128, 256, 256, 7e-3, 2),
        (1, 1024, 1024, 9e-4, 2),
        (16, 1024, 1024, 1.5e-2, 2),
        (1, 64, 64, 7.5e-5, 3),
        (512, 64, 64, 3.4e-3, 3),
        (1, 256, 256, 1.2e-4, 3),
        (128, 256, 256, 1.25e-2, 3),
        (1, 1024, 1024, 2e-3, 3),
        (8, 1024, 1024, 1.5e-2, 3),
    ],
)
def test_universal_median_time(B, H, W, limit_time, radius):
    """Just makes sure compilation/shape handling hold for >1M voxels."""
    rng = jrandom.PRNGKey(0)
    flow_field = jrandom.normal(rng, (B, H, W, 2))

    fn = jit(
        partial(universal_median_test, epsilon=0.1, r_threshold=2.0, radius=radius)
    )
    flow_field, mask, _ = fn(flow_field)  # should compile & run without raising
    assert mask.shape == (B, H, W)

    def run_jit():
        _, mask, _ = fn(flow_field)
        mask.block_until_ready()

    # Warm up the JIT compilation
    run_jit()

    # Measure the time taken for multiple executions
    total_time_jit = timeit.repeat(
        stmt=run_jit,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = max(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"
