import pytest
import jax.numpy as jnp
from jax import random
from flowgym.common.filters import (
    gaussian_kernel,
    gaussian_smoothing,
    uniform_kernel,
)
from flowgym.flow.open_piv.openpiv_jax import (
    replace_invalid_single,
    replace_outliers,
)


@pytest.fixture
def small_field():
    key = random.PRNGKey(0)
    return random.normal(key, (16, 16, 2))


@pytest.fixture
def small_batch_field():
    key = random.PRNGKey(1)
    return random.normal(key, (256, 16, 16, 2))


@pytest.fixture
def small_flags():
    flags = jnp.zeros((16, 16), dtype=bool)
    flags = flags.at[5:7, 5:7].set(True)
    return flags


@pytest.fixture
def small_batch_flags():
    flags = jnp.zeros((256, 16, 16), dtype=bool)
    flags = flags.at[5:7, 5:7, :].set(True)
    return flags


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("truncate", [2.0, 3.0])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_gaussian_kernel(sigma, truncate, n_channels):
    size = 2 * int(truncate * sigma + 0.5) + 1
    kernel = gaussian_kernel(sigma, truncate, n_channels=n_channels)
    assert kernel.shape == (size, size, n_channels)
    assert jnp.isclose(kernel.sum(axis=(0, 1)), jnp.ones((n_channels,))).all()


def test_gaussian_smoothing(small_batch_field):
    sigma = 1.0
    smoothed = gaussian_smoothing(small_batch_field, sigma)
    assert smoothed.shape == small_batch_field.shape
    # Test smoothing reduces standard deviation
    assert smoothed.std() < small_batch_field.std()


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_uniform_kernel(kernel_size, n_channels):
    kernel = uniform_kernel(kernel_size, n_channels=n_channels)
    assert kernel.shape == (kernel_size * 2 + 1, kernel_size * 2 + 1, n_channels)
    assert jnp.isclose(
        kernel[kernel_size, kernel_size, :], jnp.zeros((n_channels,))
    ).all()
    assert jnp.isclose(kernel.sum(axis=(0, 1)), jnp.ones((n_channels,))).all()


def test_replace_invalid_single(small_field, small_flags):
    kernel = uniform_kernel(kernel_size=1, n_channels=2)
    max_iter = 5
    flow_field = small_field
    flow_field = flow_field.at[small_flags].set(jnp.nan)
    replaced_field = replace_invalid_single(flow_field, small_flags, kernel, max_iter)
    assert not jnp.isnan(replaced_field).any()
    assert replaced_field.shape == small_field.shape
    # Check if invalid positions have been replaced
    replaced_values = replaced_field[small_flags, :]
    original_values = small_field[small_flags, :]
    assert not jnp.array_equal(replaced_values, original_values)
    # Check if valid positions remain unchanged
    valid_positions = ~small_flags
    assert jnp.array_equal(
        replaced_field[valid_positions, :], small_field[valid_positions, :]
    )


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("n_iter", [5])
def test_replace_outliers(small_batch_field, small_batch_flags, kernel_size, n_iter):
    updated_flow = replace_outliers(
        small_batch_field, small_batch_flags, n_iter=n_iter, kernel_size=kernel_size
    )

    assert updated_flow.shape == small_batch_field.shape

    # Verify replacement happened
    assert jnp.all(
        updated_flow[small_batch_flags, :] != small_batch_field[small_batch_flags, :]
    )

    # Verify that the valid values remain unchanged
    valid_positions = ~small_batch_flags
    assert jnp.array_equal(
        updated_flow[valid_positions, :], small_batch_field[valid_positions, :]
    )
