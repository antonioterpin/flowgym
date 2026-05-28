"""Tests for the PyTorch -> JAX RAFT32-PIV checkpoint converter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.nn.raft_model import RaftEstimatorModel
from flowgym.nn.raft_torch_to_jax import convert_raft_torch_state_dict

PATCH_SIZE = 32

# Per-ResidualBlock (in_channels, out_channels), in declaration order.
_ENCODER_RESIDUAL_PROGRESSION: tuple[tuple[int, int], ...] = (
    (64, 64),
    (64, 64),
    (64, 96),
    (96, 96),
    (96, 128),
    (128, 128),
)

_ENCODER_LAYER_IDS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
)

# (weight_shape, bias_shape) for every update_block conv.
_UPDATE_BLOCK_SHAPES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "update_block.encoder.convc1": ((256, 324, 1, 1), (256,)),
    "update_block.encoder.convc2": ((192, 256, 3, 3), (192,)),
    "update_block.encoder.convf1": ((128, 2, 7, 7), (128,)),
    "update_block.encoder.convf2": ((64, 128, 3, 3), (64,)),
    "update_block.encoder.conv": ((126, 256, 3, 3), (126,)),
    "update_block.gru.convz1": ((128, 384, 1, 5), (128,)),
    "update_block.gru.convr1": ((128, 384, 1, 5), (128,)),
    "update_block.gru.convq1": ((128, 384, 1, 5), (128,)),
    "update_block.gru.convz2": ((128, 384, 5, 1), (128,)),
    "update_block.gru.convr2": ((128, 384, 5, 1), (128,)),
    "update_block.gru.convq2": ((128, 384, 5, 1), (128,)),
    "update_block.flow_head.conv1": ((256, 128, 3, 3), (256,)),
    "update_block.flow_head.conv2": ((2, 256, 3, 3), (2,)),
    "update_block.mask.0": ((256, 128, 3, 3), (256,)),
    "update_block.mask.2": ((576, 256, 1, 1), (576,)),
}


def _make_numpy_state_dict(seed: int = 0) -> dict[str, np.ndarray]:
    """Build a torch-shaped RAFT32 state dict using only numpy.

    The values are deterministic random arrays of the exact shapes the
    PyTorch ``RAFT`` module would produce, so the converter can be
    exercised end-to-end without a torch dependency.

    Args:
        seed: Seed for the deterministic RNG.

    Returns:
        A dict mapping torch parameter names to numpy arrays.
    """
    rng = np.random.default_rng(seed)
    sd: dict[str, np.ndarray] = {}

    def _add_encoder(prefix: str, output_dim: int) -> None:
        sd[f"{prefix}.conv1.weight"] = rng.standard_normal(
            (64, 1, 7, 7), dtype=np.float32
        )
        sd[f"{prefix}.conv1.bias"] = rng.standard_normal(
            (64,), dtype=np.float32
        )
        for (in_p, p), (layer_idx, sub) in zip(
            _ENCODER_RESIDUAL_PROGRESSION, _ENCODER_LAYER_IDS, strict=True
        ):
            base = f"{prefix}.layer{layer_idx}.{sub}"
            sd[f"{base}.conv1.weight"] = rng.standard_normal(
                (p, in_p, 3, 3), dtype=np.float32
            )
            sd[f"{base}.conv1.bias"] = rng.standard_normal(
                (p,), dtype=np.float32
            )
            sd[f"{base}.conv2.weight"] = rng.standard_normal(
                (p, p, 3, 3), dtype=np.float32
            )
            sd[f"{base}.conv2.bias"] = rng.standard_normal(
                (p,), dtype=np.float32
            )
            sd[f"{base}.downsample.0.weight"] = rng.standard_normal(
                (p, in_p, 1, 1), dtype=np.float32
            )
            sd[f"{base}.downsample.0.bias"] = rng.standard_normal(
                (p,), dtype=np.float32
            )
        sd[f"{prefix}.conv2.weight"] = rng.standard_normal(
            (output_dim, 128, 1, 1), dtype=np.float32
        )
        sd[f"{prefix}.conv2.bias"] = rng.standard_normal(
            (output_dim,), dtype=np.float32
        )

    _add_encoder("fnet", output_dim=256)
    _add_encoder("cnet", output_dim=256)

    for name, (w_shape, b_shape) in _UPDATE_BLOCK_SHAPES.items():
        sd[f"{name}.weight"] = rng.standard_normal(w_shape, dtype=np.float32)
        sd[f"{name}.bias"] = rng.standard_normal(b_shape, dtype=np.float32)

    return sd


def _build_jax_model() -> RaftEstimatorModel:
    """Construct the Flax counterpart with the production RAFT32 config.

    Returns:
        The Flax ``RaftEstimatorModel`` in inference mode.
    """
    return RaftEstimatorModel(
        hidden_dim=128,
        context_dim=128,
        corr_levels=4,
        corr_radius=4,
        iters=12,
        norm_fn="instance",
        dropout=0.0,
        train=False,
    )


def _jax_param_template() -> dict:
    """Return the params pytree produced by ``RaftEstimatorModel.init``.

    Returns:
        A nested params dict with the canonical RAFT32 shapes.
    """
    model = _build_jax_model()
    dummy = jnp.zeros((1, PATCH_SIZE, PATCH_SIZE, 2), dtype=jnp.float32)
    return model.init(jax.random.PRNGKey(0), dummy, dummy)["params"]


def test_converted_params_match_jax_param_tree():
    """Converted params must have the exact tree/shape as ``model.init``.

    Runs in the default test environment: depends only on numpy + JAX.
    """
    state_dict = _make_numpy_state_dict()
    converted = convert_raft_torch_state_dict(state_dict)

    template = _jax_param_template()
    flat_template = jax.tree_util.tree_map(lambda x: x.shape, template)
    flat_converted = jax.tree_util.tree_map(lambda x: x.shape, converted)
    assert flat_template == flat_converted


def test_converted_values_round_trip_through_param_template():
    """Converted leaves match the source state dict, byte-for-byte.

    Each Flax leaf is the corresponding torch tensor transposed into Flax
    convention (``(out, in, kH, kW) -> (kH, kW, in, out)`` for kernels).
    Validate this against the source ``state_dict`` without any torch
    dependency, so the core converter is exercised in the default env.
    """
    state_dict = _make_numpy_state_dict(seed=7)
    converted = convert_raft_torch_state_dict(state_dict)

    # Spot-check both an encoder kernel and an update-block kernel.
    fnet_conv1 = converted["EncoderBlock_0"]["ConvBlock_0"]["Conv_0"]
    expected = state_dict["fnet.conv1.weight"].transpose(2, 3, 1, 0)
    np.testing.assert_array_equal(np.asarray(fnet_conv1["kernel"]), expected)
    np.testing.assert_array_equal(
        np.asarray(fnet_conv1["bias"]), state_dict["fnet.conv1.bias"]
    )

    mask_head = converted["UpdateBlock_0"]["ConvBlock_1"]["Conv_0"]
    expected_mask = state_dict["update_block.mask.2.weight"].transpose(
        2, 3, 1, 0
    )
    np.testing.assert_array_equal(
        np.asarray(mask_head["kernel"]), expected_mask
    )
    np.testing.assert_array_equal(
        np.asarray(mask_head["bias"]), state_dict["update_block.mask.2.bias"]
    )


def test_converter_accepts_objects_with_detach_method():
    """The converter normalises tensor-like inputs via ``.detach().cpu()``.

    Use a tiny duck-typed stand-in (no torch import) to exercise the
    branch that handles tensor objects.
    """

    class _FakeTensor:
        def __init__(self, array: np.ndarray):
            self._array = array

        def detach(self) -> _FakeTensor:
            return self

        def cpu(self) -> _FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self._array

    numpy_sd = _make_numpy_state_dict(seed=11)
    fake_sd = {k: _FakeTensor(v) for k, v in numpy_sd.items()}

    from_numpy = convert_raft_torch_state_dict(numpy_sd)
    from_fake = convert_raft_torch_state_dict(fake_sd)

    leaves_a = jax.tree_util.tree_leaves(from_numpy)
    leaves_b = jax.tree_util.tree_leaves(from_fake)
    for a, b in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ──────────────────────────────────────────────────────────────────────────
# PyTorch parity tests — slow and require the optional ``torch`` dependency.
# ──────────────────────────────────────────────────────────────────────────


class _Args:
    """Minimal argparse-like holder consumed by ``RAFT.forward``."""

    amp = False
    iters = 12


def _torch_to_jax_inputs(torch_module, img1: np.ndarray, img2: np.ndarray):
    """Build matching torch and JAX inputs from two raw image arrays.

    Args:
        torch_module: The imported ``torch`` module (passed in so this
            helper does not import torch at module scope).
        img1: First-frame batch, shape ``(B, 1, H, W)``.
        img2: Second-frame batch, same shape as ``img1``.

    Returns:
        Tuple ``(torch_input, jax_input)``. ``torch_input`` is already
        normalised to ``[0, 1]`` to match how the upstream pipeline
        invokes RAFT; ``jax_input`` is the raw ``(B, H, W, 2)`` stack
        consumed by ``RaftEstimatorModel`` (which normalises internally).
    """
    torch_input = torch_module.from_numpy(
        np.concatenate([img1, img2], axis=1) / 256.0
    )
    jax_input = jnp.asarray(
        np.concatenate([img1, img2], axis=1).transpose(0, 2, 3, 1)
    )
    return torch_input, jax_input


def _build_torch_raft_with_state(state_dict):
    """Instantiate the PyTorch RAFT model and load ``state_dict`` into it.

    Args:
        state_dict: PyTorch state dict for the RAFT model. Any ``warp.*``
            buffer keys are dropped (they are not part of ``RAFT``).

    Returns:
        The loaded RAFT model in ``eval`` mode.
    """
    from flowgym.nn.raft_torch_nn.flowNetsRAFT import RAFT

    sd = {k: v for k, v in state_dict.items() if not k.startswith("warp.")}
    model = RAFT()
    model.load_state_dict(sd)
    model.eval()
    return model


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_converted_model_matches_pytorch_output():
    """Outputs of converted JAX model must match PyTorch RAFT closely."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    from flowgym.nn.raft_torch_nn.flowNetsRAFT import RAFT

    state_dict = RAFT().state_dict()
    torch_model = _build_torch_raft_with_state(state_dict)
    params = convert_raft_torch_state_dict(state_dict)
    jax_model = _build_jax_model()

    rng = np.random.default_rng(42)
    img1 = rng.uniform(0, 255, size=(1, 1, PATCH_SIZE, PATCH_SIZE)).astype(
        np.float32
    )
    img2 = rng.uniform(0, 255, size=(1, 1, PATCH_SIZE, PATCH_SIZE)).astype(
        np.float32
    )

    torch_input, jax_input = _torch_to_jax_inputs(torch, img1, img2)
    with torch.no_grad():
        flow_preds, _ = torch_model(
            torch_input, torch.zeros_like(torch_input), args=_Args()
        )
    torch_out = flow_preds[-1].numpy()  # (B, 2, H, W)

    flow_init = jnp.zeros((1, PATCH_SIZE, PATCH_SIZE, 2), dtype=jnp.float32)
    jax_iters = jax_model.apply({"params": params}, jax_input, flow_init)
    jax_out = np.asarray(jax_iters[-1]).transpose(0, 3, 1, 2)  # (B, 2, H, W)

    # The Flax ``ConvBlock`` runs its inner conv in float16, while the
    # PyTorch reference is driven in float32 here, so an exact match is not
    # achievable. The tolerance below comfortably absorbs the resulting
    # mixed-precision error on a 32x32 patch with [0, 1)-scaled inputs.
    np.testing.assert_allclose(jax_out, torch_out, atol=2e-2, rtol=5e-2)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_all_intermediate_flow_iterations_match():
    """All 12 refinement iterations should agree, not just the last one."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    from flowgym.nn.raft_torch_nn.flowNetsRAFT import RAFT

    state_dict = RAFT().state_dict()
    torch_model = _build_torch_raft_with_state(state_dict)
    params = convert_raft_torch_state_dict(state_dict)
    jax_model = _build_jax_model()

    rng = np.random.default_rng(123)
    img1 = rng.uniform(0, 255, size=(2, 1, PATCH_SIZE, PATCH_SIZE)).astype(
        np.float32
    )
    img2 = rng.uniform(0, 255, size=(2, 1, PATCH_SIZE, PATCH_SIZE)).astype(
        np.float32
    )

    torch_input, jax_input = _torch_to_jax_inputs(torch, img1, img2)
    with torch.no_grad():
        flow_preds, _ = torch_model(
            torch_input, torch.zeros_like(torch_input), args=_Args()
        )

    flow_init = jnp.zeros((2, PATCH_SIZE, PATCH_SIZE, 2), dtype=jnp.float32)
    jax_iters = jax_model.apply({"params": params}, jax_input, flow_init)
    jax_iters_np = np.asarray(jax_iters).transpose(0, 1, 4, 2, 3)

    assert len(flow_preds) == jax_iters_np.shape[0] == 12
    for i, torch_flow in enumerate(flow_preds):
        np.testing.assert_allclose(
            jax_iters_np[i], torch_flow.numpy(), atol=2e-2, rtol=5e-2
        )
