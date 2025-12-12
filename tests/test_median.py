import jax
import jax.numpy as jnp
import timeit
from flowgym.common.median import (
    median,
    median8,
    median9,
    median24,
    median25,
    median32,
    median48,
    median49,
    median64,
)
import pytest
from functools import partial

from flowgym.utils import load_configuration

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]
NUMBER_OF_EXECUTIONS = config["EXECUTIONS_POST_PROCESS"]


@pytest.mark.parametrize("dim", [1, 2, 3, 8, 9, 24, 25, 32, 48, 49, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
def test_median(dim, batch_size, seed):
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (batch_size, 16, 16, dim))
    res = median(data)
    expected = jnp.median(data, axis=-1)

    assert res.shape == expected.shape, f"Median{dim} {res.shape=}, {expected.shape=}."
    assert jnp.allclose(
        res, expected
    ), f"Median{dim} function did not return expected results."


@pytest.mark.parametrize(
    "fn, dim",
    [
        (median8, 8),
        (median9, 9),
        (median24, 24),
        (median25, 25),
        (median32, 32),
        (median48, 48),
        (median49, 49),
        (median64, 64),
    ],
)
@pytest.mark.parametrize(
    "seed, shape",
    [
        (0, (16, 32)),
        (1, (4,)),
        (2, (32, 1)),
        (42, (1, 8, 1, 2)),
        (100, (10, 1, 3)),
    ],
)
def test_median_correctness(fn, dim, seed, shape):
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, shape + (dim,))
    res = fn(data)
    expected = jnp.median(data, axis=-1)

    assert res.shape == expected.shape, f"Median{dim} {res.shape=}, {expected.shape=}."
    assert jnp.allclose(
        res, expected
    ), f"Median{dim} function did not return expected results."


@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize(
    "fn, dim, limit_time, shape, cmp_std",
    [
        (median8, 8, 3.6e-5, (1, 64, 64), False),
        (median8, 8, 5.3e-5, (256, 64, 64), False),
        (median8, 8, 5.5e-5, (16, 256, 256), False),
        (median8, 8, 5.5e-5, (1, 1024, 1024), False),
        (median8, 8, 8e-4, (16, 1024, 1024), False),
        (median9, 9, 3.5e-5, (1, 64, 64), False),
        (median9, 9, 5.3e-5, (256, 64, 64), False),
        (median9, 9, 5.5e-5, (16, 256, 256), False),
        (median9, 9, 5.5e-5, (1, 1024, 1024), False),
        (median9, 9, 1e-3, (16, 1024, 1024), False),
        (median24, 24, 3.5e-5, (1, 64, 64), False),
        (median24, 24, 3e-4, (256, 64, 64), False),
        (median24, 24, 3e-4, (16, 256, 256), False),
        (median24, 24, 3e-4, (1, 1024, 1024), False),
        (median24, 24, 5e-3, (16, 1024, 1024), False),
        (median25, 25, 3.5e-5, (1, 64, 64), False),
        (median25, 25, 3.4e-4, (256, 64, 64), False),
        (median25, 25, 3.4e-4, (16, 256, 256), False),
        (median25, 25, 3.4e-4, (1, 1024, 1024), False),
        (median25, 25, 5e-3, (16, 1024, 1024), False),
        (median48, 48, 3.5e-5, (1, 64, 64), False),
        (median48, 48, 7.2e-4, (256, 64, 64), False),
        (median48, 48, 7.2e-4, (16, 256, 256), False),
        (median48, 48, 7.2e-4, (1, 1024, 1024), False),
        (median48, 48, 1.2e-2, (16, 1024, 1024), False),
        (median49, 49, 3.5e-5, (1, 64, 64), False),
        (median49, 49, 7.3e-4, (256, 64, 64), False),
        (median49, 49, 7.3e-4, (16, 256, 256), False),
        (median49, 49, 7.3e-4, (1, 1024, 1024), False),
        (median49, 49, 1.2e-2, (16, 1024, 1024), False),
    ],
)
@pytest.mark.parametrize("seed", [0])
def test_median_time(fn, dim, limit_time, seed, shape, cmp_std):
    COMPARE_TO_STANDARD = cmp_std
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, shape + (dim,))

    fn_jit = jax.jit(fn)

    def measure_jit():
        d = fn_jit(data)
        d.block_until_ready()

    # Measure execution time
    total_time_jit = timeit.repeat(
        stmt=measure_jit,
        number=NUMBER_OF_EXECUTIONS,
        repeat=REPETITIONS,
    )
    average_time_jit = min(total_time_jit) / NUMBER_OF_EXECUTIONS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"

    if COMPARE_TO_STANDARD:
        # unbearably SLOW!
        # measure time for standard median
        median_jit = jax.jit(partial(jnp.median, axis=-1))

        def measure_standard():
            d = median_jit(data)
            d.block_until_ready()

        total_time_standard = timeit.repeat(
            stmt=measure_standard,
            number=NUMBER_OF_EXECUTIONS,
            repeat=REPETITIONS,
        )
        average_time_standard = min(total_time_standard) / NUMBER_OF_EXECUTIONS
        assert average_time_jit < average_time_standard / 10, (
            f"Median{dim} is slower than 10x standard median: "
            f"{average_time_jit} vs {average_time_standard}"
        )
