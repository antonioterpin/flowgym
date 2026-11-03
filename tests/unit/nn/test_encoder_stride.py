"""Configured input downsampling survives router validation and construction."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.flow.art_of_piv.validation.router import normalize_router_config
from flowgym.nn.cnn import build_cnn_encoder_from_config


def test_router_encoder_stride_changes_spatial_resolution() -> None:
    """Only the first convolution applies the requested input stride."""
    config = normalize_router_config(
        {
            "encoder": {"features_list": [4], "first_conv_stride": 4},
            "head": {"hidden_dims": [4]},
        }
    )
    encoder = build_cnn_encoder_from_config(config["encoder"])
    image = jnp.ones((1, 32, 32, 2))
    variables = encoder.init(jax.random.PRNGKey(0), image)
    _, state = encoder.apply(
        variables, image, capture_intermediates=True, mutable=["intermediates"]
    )
    intermediate = state["intermediates"]["ConvBlock_0"]["__call__"][0]
    assert intermediate.shape[1:3] == (8, 8), (
        "Input stride must reach the actual convolution"
    )


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_invalid_encoder_stride(stride: object) -> None:
    """Invalid stride values must not be ignored or truncated."""
    with pytest.raises(ValueError, match="first_conv_stride"):
        normalize_router_config(
            {
                "encoder": {"features_list": [4], "first_conv_stride": stride},
                "head": {"hidden_dims": [4]},
            }
        )
