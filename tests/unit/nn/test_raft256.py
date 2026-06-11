"""Tests for the RAFT256-PIV Flax model and its PyTorch parity.

RAFT256-PIV reuses the RAFT32-PIV parameter tree (the PyTorch->Flax converter
in :mod:`flowgym.nn.raft_torch_to_jax` is shared), so the converter's tree and
value mappings are already exercised by ``test_raft_torch_to_jax``. Here we
check the RAFT256-specific pieces:

* the convex upsampling head matches an explicit reference implementation;
* the RAFT256 Flax parameter tree is structurally identical to RAFT32's, so
  the shared converter applies; and
* the converted Flax model reproduces the PyTorch ``RAFT256`` output (within
  the same mixed-precision tolerance as RAFT32 — the Flax ``ConvBlock`` runs
  its inner convolutions in float16 while the PyTorch reference runs in
  float32).

The port is mathematically exact: running the Flax model with float32 inner
convolutions and ``jax_default_matmul_precision='highest'`` reproduces the real
RAFT256-PIV checkpoint to ~5e-4 max absolute error. The looser default
tolerance below is entirely due to the inherited mixed-precision design (the
float16 inner convolutions and JAX's default correlation-volume matmul
precision), amplified by the convex-upsampling softmax over 16 iterations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowgym.flow.raft.process import convex_upsample
from flowgym.nn.raft256_model import RaftEstimatorModel256
from flowgym.nn.raft_model import RaftEstimatorModel
from flowgym.nn.raft_torch_to_jax import convert_raft_torch_state_dict

PATCH_SIZE = 256

# The convex upsampling softmax amplifies the float16/float32 mismatch of the
# inner convolutions across the 16 refinement iterations, so RAFT256 needs a
# slightly looser tolerance than RAFT32 (whose last iteration sits at ~2e-2).
_ATOL = 4e-2
_RTOL = 5e-2


def _build_jax_model(iters: int = 16) -> RaftEstimatorModel256:
    """Construct the Flax RAFT256 model with the production config.

    Args:
        iters: Number of refinement iterations.

    Returns:
        The Flax ``RaftEstimatorModel256`` in inference mode.
    """
    return RaftEstimatorModel256(
        hidden_dim=128,
        context_dim=128,
        corr_levels=4,
        corr_radius=4,
        iters=iters,
        norm_fn="instance",
        dropout=0.0,
        train=False,
    )


# ──────────────────────────────────────────────────────────────────────────
# Convex upsampling — exact-parity unit test (no torch).
# ──────────────────────────────────────────────────────────────────────────


def _reference_convex_upsample(
    flow: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Explicit, unambiguous reference for ``convex_upsample``.

    Mirrors the PyTorch ``RAFT256.upsample_flow`` semantics with plain Python
    loops, so it is an independent cross-check of the vectorised version.

    Args:
        flow: (N, H, W, 2) low-resolution flow.
        mask: (N, H, W, 9*8*8) convex-combination logits.

    Returns:
        (N, 8H, 8W, 2) upsampled flow.
    """
    N, H, W, _ = flow.shape
    m = mask.reshape(N, H, W, 9, 8, 8).astype(np.float64)
    # softmax over the nine neighbours
    m = np.exp(m - m.max(axis=3, keepdims=True))
    m /= m.sum(axis=3, keepdims=True)

    scaled = 4.0 * flow.astype(np.float64)
    out = np.zeros((N, H * 8, W * 8, 2), dtype=np.float64)
    for n in range(N):
        for h in range(H):
            for w in range(W):
                taps = np.zeros((9, 2), dtype=np.float64)
                for kh in range(3):
                    for kw in range(3):
                        ih, iw = h + kh - 1, w + kw - 1
                        if 0 <= ih < H and 0 <= iw < W:
                            taps[kh * 3 + kw] = scaled[n, ih, iw]
                for sh in range(8):
                    for sw in range(8):
                        weights = m[n, h, w, :, sh, sw]
                        out[n, h * 8 + sh, w * 8 + sw] = (
                            weights[:, None] * taps
                        ).sum(0)
    return out


def test_convex_upsample_matches_reference():
    """``convex_upsample`` matches the explicit-loop reference exactly."""
    rng = np.random.default_rng(0)
    N, H, W = 2, 3, 4
    flow = rng.standard_normal((N, H, W, 2)).astype(np.float32)
    mask = rng.standard_normal((N, H, W, 9 * 8 * 8)).astype(np.float32)

    got = np.asarray(convex_upsample(jnp.asarray(flow), jnp.asarray(mask)))
    expected = _reference_convex_upsample(flow, mask)

    assert got.shape == (N, H * 8, W * 8, 2)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


# ──────────────────────────────────────────────────────────────────────────
# Parameter tree — RAFT256 reuses the RAFT32 converter (no torch).
# ──────────────────────────────────────────────────────────────────────────


def test_raft256_param_tree_matches_raft32():
    """RAFT256 and RAFT32 produce structurally identical parameter trees.

    The two variants differ only in encoder strides and whether the upsampling
    mask is used; neither changes parameter shapes or names. Combined with the
    converter tests in ``test_raft_torch_to_jax`` (which validate the converter
    against the RAFT32 tree), this transitively proves the shared converter
    fits RAFT256.
    """
    key = jax.random.PRNGKey(0)

    model256 = _build_jax_model(iters=4)
    dummy256 = jnp.zeros((1, PATCH_SIZE, PATCH_SIZE, 2), dtype=jnp.float32)
    tree256 = model256.init(key, dummy256, dummy256)["params"]

    model32 = RaftEstimatorModel(
        hidden_dim=128,
        context_dim=128,
        corr_levels=4,
        corr_radius=4,
        iters=4,
        norm_fn="instance",
        dropout=0.0,
        train=False,
    )
    dummy32 = jnp.zeros((1, 32, 32, 2), dtype=jnp.float32)
    tree32 = model32.init(key, dummy32, dummy32)["params"]

    shapes256 = jax.tree_util.tree_map(lambda x: x.shape, tree256)
    shapes32 = jax.tree_util.tree_map(lambda x: x.shape, tree32)
    assert shapes256 == shapes32


def test_raft256_forward_output_shape():
    """The model upsamples 1/8-resolution flow back to full resolution."""
    patch = 64  # divisible by 8
    iters = 3
    model = _build_jax_model(iters=iters)
    key = jax.random.PRNGKey(1)
    images = jnp.zeros((2, patch, patch, 2), dtype=jnp.float32)
    flow_init = jnp.zeros((2, patch, patch, 2), dtype=jnp.float32)

    params = model.init(key, images, flow_init)["params"]
    flows = model.apply({"params": params}, images, flow_init)

    assert flows.shape == (iters, 2, patch, patch, 2)


# ──────────────────────────────────────────────────────────────────────────
# PyTorch parity tests — slow and require the optional ``torch`` dependency.
# ──────────────────────────────────────────────────────────────────────────


class _Args:
    """Minimal argparse-like holder consumed by ``RAFT256.forward``."""

    amp = False
    iters = 16


def _torch_to_jax_inputs(torch_module, img1: np.ndarray, img2: np.ndarray):
    """Build matching torch and JAX inputs from two raw image arrays.

    Args:
        torch_module: The imported ``torch`` module.
        img1: First-frame batch, shape ``(B, 1, H, W)``.
        img2: Second-frame batch, same shape as ``img1``.

    Returns:
        Tuple ``(torch_input, jax_input)``. ``torch_input`` is normalised to
        ``[0, 1]`` to match the upstream pipeline; ``jax_input`` is the raw
        ``(B, H, W, 2)`` stack consumed by ``RaftEstimatorModel256`` (which
        normalises internally).
    """
    torch_input = torch_module.from_numpy(
        np.concatenate([img1, img2], axis=1) / 256.0
    )
    jax_input = jnp.asarray(
        np.concatenate([img1, img2], axis=1).transpose(0, 2, 3, 1)
    )
    return torch_input, jax_input


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_converted_raft256_model_matches_pytorch_output():
    """Converted JAX RAFT256 matches PyTorch RAFT256 closely."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    from flowgym.nn.raft_torch_nn.flowNetsRAFT256 import RAFT256

    state_dict = RAFT256().state_dict()
    torch_model = RAFT256()
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    params = convert_raft_torch_state_dict(state_dict)
    jax_model = _build_jax_model(iters=16)

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

    np.testing.assert_allclose(jax_out, torch_out, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_raft256_all_intermediate_flow_iterations_match():
    """All refinement iterations should agree, not just the last one."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    from flowgym.nn.raft_torch_nn.flowNetsRAFT256 import RAFT256

    # A 128x128 patch (16x16 feature maps) keeps the all-pairs correlation and
    # convex upsampling well-conditioned; the degenerate 8x8 regime makes the
    # untrained PyTorch reference's plain InstanceNorm diverge/NaN.
    iters = 8
    patch = 128

    class _Args8:
        amp = False
        iters = 8

    state_dict = RAFT256().state_dict()
    torch_model = RAFT256()
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    params = convert_raft_torch_state_dict(state_dict)
    jax_model = _build_jax_model(iters=iters)

    rng = np.random.default_rng(123)
    img1 = rng.uniform(0, 255, size=(2, 1, patch, patch)).astype(np.float32)
    img2 = rng.uniform(0, 255, size=(2, 1, patch, patch)).astype(np.float32)

    torch_input, jax_input = _torch_to_jax_inputs(torch, img1, img2)
    with torch.no_grad():
        flow_preds, _ = torch_model(
            torch_input, torch.zeros_like(torch_input), args=_Args8()
        )

    flow_init = jnp.zeros((2, patch, patch, 2), dtype=jnp.float32)
    jax_iters = jax_model.apply({"params": params}, jax_input, flow_init)
    jax_iters_np = np.asarray(jax_iters).transpose(0, 1, 4, 2, 3)

    assert len(flow_preds) == jax_iters_np.shape[0] == iters
    for i, torch_flow in enumerate(flow_preds):
        np.testing.assert_allclose(
            jax_iters_np[i], torch_flow.numpy(), atol=_ATOL, rtol=_RTOL
        )
