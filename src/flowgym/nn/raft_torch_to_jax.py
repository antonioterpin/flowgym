"""Convert a PyTorch RAFT32-PIV checkpoint into Flax/JAX parameters.

The PyTorch model is defined in ``flowgym.nn.raft_torch_nn`` and the
Flax counterpart in ``flowgym.nn.raft_model``. Both implement the same
RAFT architecture but with different parameter naming. This module
builds an explicit mapping between PyTorch state-dict keys and the
nested Flax parameter tree produced by ``RaftEstimatorModel.init``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

# PyTorch conv weights: (out_channels, in_channels, kH, kW)
# Flax Conv kernels:    (kH, kW, in_channels, out_channels)
_CONV_PERM = (2, 3, 1, 0)

# Six ResidualBlocks per EncoderBlock, in PyTorch addressing.
_ENCODER_RESIDUALS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
)


def _torch_conv_to_jax(
    weight: np.ndarray, bias: np.ndarray
) -> dict[str, jnp.ndarray]:
    """Convert a single PyTorch conv layer to Flax ``Conv`` parameters.

    Args:
        weight: PyTorch conv weight of shape ``(out, in, kH, kW)``.
        bias: PyTorch conv bias of shape ``(out,)``.

    Returns:
        Dict with ``kernel`` and ``bias`` arrays in Flax convention.
    """
    return {
        "kernel": jnp.asarray(weight.transpose(_CONV_PERM)),
        "bias": jnp.asarray(bias),
    }


def _encoder_params(
    sd: Mapping[str, np.ndarray], prefix: str
) -> dict[str, dict]:
    """Build Flax params for a single ``EncoderBlock``.

    Args:
        sd: Source PyTorch state dict (numpy arrays).
        prefix: Either ``"fnet"`` or ``"cnet"``.

    Returns:
        Nested params dict matching ``EncoderBlock`` keys.
    """
    out: dict[str, dict] = {}

    # First 7x7 conv: PyTorch conv1 -> Flax ConvBlock_0/Conv_0
    out["ConvBlock_0"] = {
        "Conv_0": _torch_conv_to_jax(
            sd[f"{prefix}.conv1.weight"], sd[f"{prefix}.conv1.bias"]
        )
    }

    # Six residual blocks, in declaration order.
    for idx, (layer, sub) in enumerate(_ENCODER_RESIDUALS):
        base = f"{prefix}.layer{layer}.{sub}"
        out[f"ResidualBlock_{idx}"] = {
            "ConvBlock_0": {
                "Conv_0": _torch_conv_to_jax(
                    sd[f"{base}.conv1.weight"], sd[f"{base}.conv1.bias"]
                )
            },
            "ConvBlock_1": {
                "Conv_0": _torch_conv_to_jax(
                    sd[f"{base}.conv2.weight"], sd[f"{base}.conv2.bias"]
                )
            },
            # 1x1 skip/projection. The torch ResidualBlock always carries
            # a ``downsample.0`` 1x1 conv (even when stride == 1), which
            # the Flax ResidualBlock mirrors via its top-level ``Conv_0``.
            "Conv_0": _torch_conv_to_jax(
                sd[f"{base}.downsample.0.weight"],
                sd[f"{base}.downsample.0.bias"],
            ),
        }

    # Output 1x1 conv: PyTorch conv2 -> Flax ConvBlock_1/Conv_0
    out["ConvBlock_1"] = {
        "Conv_0": _torch_conv_to_jax(
            sd[f"{prefix}.conv2.weight"], sd[f"{prefix}.conv2.bias"]
        )
    }

    return out


def _update_block_params(sd: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """Build Flax params for the ``UpdateBlock``.

    Args:
        sd: Source PyTorch state dict (numpy arrays).

    Returns:
        Nested params dict matching ``UpdateBlock`` keys.
    """
    p: dict[str, dict] = {}

    # Motion encoder.
    enc = "update_block.encoder"
    p["MotionEncoderBlock_0"] = {
        "ConvBlock_0": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{enc}.convc1.weight"], sd[f"{enc}.convc1.bias"]
            )
        },
        "ConvBlock_1": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{enc}.convc2.weight"], sd[f"{enc}.convc2.bias"]
            )
        },
        "ConvBlock_2": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{enc}.convf1.weight"], sd[f"{enc}.convf1.bias"]
            )
        },
        "ConvBlock_3": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{enc}.convf2.weight"], sd[f"{enc}.convf2.bias"]
            )
        },
        "ConvBlock_4": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{enc}.conv.weight"], sd[f"{enc}.conv.bias"]
            )
        },
    }

    # Separable conv GRU: horizontal (z1, r1, q1), then vertical (z2, r2, q2).
    gru = "update_block.gru"
    gru_torch_names = (
        "convz1",
        "convr1",
        "convq1",
        "convz2",
        "convr2",
        "convq2",
    )
    p["SepConvGRUBlock_0"] = {
        f"ConvBlock_{i}": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{gru}.{name}.weight"], sd[f"{gru}.{name}.bias"]
            )
        }
        for i, name in enumerate(gru_torch_names)
    }

    # Flow head: conv1 then conv2.
    fh = "update_block.flow_head"
    p["FlowHeadBlock_0"] = {
        "ConvBlock_0": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{fh}.conv1.weight"], sd[f"{fh}.conv1.bias"]
            )
        },
        "ConvBlock_1": {
            "Conv_0": _torch_conv_to_jax(
                sd[f"{fh}.conv2.weight"], sd[f"{fh}.conv2.bias"]
            )
        },
    }

    # Upsampling mask head: nn.Sequential indices 0 and 2 (index 1 is ReLU).
    p["ConvBlock_0"] = {
        "Conv_0": _torch_conv_to_jax(
            sd["update_block.mask.0.weight"], sd["update_block.mask.0.bias"]
        )
    }
    p["ConvBlock_1"] = {
        "Conv_0": _torch_conv_to_jax(
            sd["update_block.mask.2.weight"], sd["update_block.mask.2.bias"]
        )
    }

    return p


def convert_raft_torch_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, dict]:
    """Convert a PyTorch RAFT32 state dict to a Flax params pytree.

    Missing keys propagate as ``KeyError`` from the indexing below.

    Args:
        state_dict: PyTorch ``model_state_dict`` for
            :class:`flowgym.nn.raft_torch_nn.flowNetsRAFT.RAFT`. Values
            may be either ``numpy`` arrays or ``torch.Tensor``; in the
            latter case they are converted to numpy via ``.cpu().numpy()``.

    Returns:
        Nested params dict consumable as ``{"params": <returned>}`` by
        :class:`flowgym.nn.raft_model.RaftEstimatorModel`.
    """
    # Normalize values to numpy (handle torch.Tensor without importing torch).
    sd: dict[str, np.ndarray] = {}
    for k, v in state_dict.items():
        if hasattr(v, "detach"):
            sd[k] = cast(Any, v).detach().cpu().numpy()
        else:
            sd[k] = np.asarray(v)

    # In ``RaftEstimatorModel``, ``EncoderBlock_0`` is fnet and
    # ``EncoderBlock_1`` is cnet (creation order in ``__call__``).
    return {
        "EncoderBlock_0": _encoder_params(sd, "fnet"),
        "EncoderBlock_1": _encoder_params(sd, "cnet"),
        "UpdateBlock_0": _update_block_params(sd),
    }
