"""Static selector contracts and JAX boolean mask guarantees for OpenPIV."""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.open_piv.process import sig2noise_val


def test_method_selectors_are_checked_statically(tmp_path):
    """Reject unknown selectors while preserving overload return inference."""
    sample = tmp_path / "selectors.py"
    sample.write_text("""from typing import assert_type
from jax import Array
from flowgym.flow.open_piv.process import extended_search_area_piv
from flowgym.flow.open_piv.openpiv_jax import OpenPIVJAXEstimator

def check(image: Array) -> None:
    assert_type(extended_search_area_piv(image, image, 8, 8, 4), Array)
    assert_type(extended_search_area_piv(
        image, image, 8, 8, 4, sig2noise_method="peak2peak"
    ), tuple[Array, Array])
    extended_search_area_piv(image, image, 8, 8, 4, subpixel_method="typo")
    extended_search_area_piv(image, image, 8, 8, 4, correlation_method="typo")
    extended_search_area_piv(image, image, 8, 8, 4, sig2noise_method="typo")
    OpenPIVJAXEstimator(subpixel_method="typo")
    OpenPIVJAXEstimator(correlation_method="typo")
""")
    result = subprocess.run(
        ["basedpyright", "--outputjson", str(sample)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = json.loads(result.stdout)["generalDiagnostics"]
    error_lines = {
        d["range"]["start"]["line"] + 1
        for d in diagnostics
        if d["severity"] == "error"
    }
    assert error_lines == {11, 12, 13, 14, 15}, diagnostics


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
@pytest.mark.parametrize("jit", [False, True])
def test_sig2noise_val_has_boolean_elements(dtype, jit):
    """Threshold masks remain boolean for integer and floating inputs."""
    fn = jax.jit(sig2noise_val) if jit else sig2noise_val
    result = fn(jnp.asarray([0, 1, 2], dtype=dtype))
    assert result.dtype == jnp.bool_, "Outlier mask must have boolean elements"
    np.testing.assert_array_equal(result, [True, False, False])


@pytest.mark.parametrize(
    "parser, method",
    [
        ("parse_subpixel_method", "gaussian"),
        ("parse_subpixel_method", "parabolic"),
        ("parse_subpixel_method", "centroid"),
        ("parse_correlation_method", "circular"),
        ("parse_correlation_method", "linear"),
        ("parse_sig2noise_method", "peak2peak"),
        ("parse_sig2noise_method", "peak2mean"),
    ],
)
def test_config_parsers_preserve_valid_selectors_and_reject_typos(
    parser, method
):
    """Untyped config input keeps a useful runtime validation backstop."""
    from flowgym.flow.open_piv import types

    parse = getattr(types, parser)
    assert parse(method) == method, "Parsing must not change a valid selector"
    with pytest.raises(ValueError, match=r"typo.*expected"):
        parse("typo")
