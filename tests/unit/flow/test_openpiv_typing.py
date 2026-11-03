"""Selector parsing and boolean mask guarantees for OpenPIV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.open_piv.process import sig2noise_val
from flowgym.flow.open_piv.types import (
    parse_correlation_method,
    parse_sig2noise_method,
    parse_subpixel_method,
)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
@pytest.mark.parametrize("jit", [False, True])
def test_sig2noise_val_has_boolean_elements(dtype, jit):
    """Threshold masks remain boolean for integer and floating inputs."""
    fn = jax.jit(sig2noise_val) if jit else sig2noise_val
    result = fn(jnp.asarray([0, 1, 2], dtype=dtype))
    assert result.dtype == jnp.bool_, "Outlier mask must have boolean elements"
    np.testing.assert_array_equal(result, [True, False, False])


@pytest.mark.parametrize(
    "parse, method",
    [
        (parse_subpixel_method, "gaussian"),
        (parse_subpixel_method, "parabolic"),
        (parse_subpixel_method, "centroid"),
        (parse_correlation_method, "circular"),
        (parse_correlation_method, "linear"),
        (parse_sig2noise_method, "peak2peak"),
        (parse_sig2noise_method, "peak2mean"),
    ],
)
def test_parsers_accept_valid_and_reject_unknown_selectors(parse, method):
    """Config parsers return valid selectors unchanged and reject typos."""
    assert parse(method) == method, "Parsing must not change a valid selector"
    with pytest.raises(ValueError, match=r"typo.*expected"):
        parse("typo")
