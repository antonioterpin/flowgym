"""Unit tests for the LIMA Flax model."""

import jax
import jax.numpy as jnp
import pytest

from flowgym.nn.lima_model import LimaModel


def _kernel_shapes(params):
    """Collect the shapes of all convolution kernels in a param tree."""
    leaves = jax.tree_util.tree_leaves_with_path(params)
    return [
        tuple(v.shape)
        for path, v in leaves
        if path[-1].key == "kernel" and v.ndim == 4
    ]


def _param_count(params):
    """Total number of scalar parameters in a param tree."""
    return sum(int(v.size) for v in jax.tree_util.tree_leaves(params))


def _init(model, size=64, key=0):
    images = jnp.zeros((1, size, size, 2))
    return model.init(jax.random.PRNGKey(key), images)["params"]


def test_forward_returns_per_level_flows_coarse_to_fine():
    """The model returns one flow per refined level, coarsest first."""
    model = LimaModel(refine_levels=6, search_range=2)
    params = _init(model, size=64)
    images = jax.random.normal(jax.random.PRNGKey(1), (2, 64, 64, 2)) * 50
    flows = model.apply({"params": params}, images)
    assert isinstance(flows, list)
    assert len(flows) == 6
    # coarse -> fine: strides 64, 32, 16, 8, 4, 2 -> sizes 1,2,4,8,16,32
    sizes = [f.shape[1] for f in flows]
    assert sizes == [1, 2, 4, 8, 16, 32]
    for f in flows:
        assert f.shape[0] == 2 and f.shape[-1] == 2  # (B, h, w, 2)


def test_encoder_decoder_kernel_shapes_match_paper_tables():
    """Conv kernel shapes match LIMA Table I (encoder) and III (decoder)."""
    model = LimaModel(refine_levels=6, search_range=2)
    shapes = _kernel_shapes(_init(model))
    expected_encoder = [
        (3, 3, 1, 16),
        (3, 3, 16, 32),
        (3, 3, 32, 64),
        (3, 3, 64, 96),
        (3, 3, 96, 128),
        (3, 3, 128, 196),
    ]
    # decoder input = (2R+1)^2 + 2 = 27 for R=2, then Table III channels,
    # then a 2-channel flow head.
    expected_decoder = [
        (3, 3, 27, 128),
        (3, 3, 128, 128),
        (3, 3, 128, 128),
        (3, 3, 128, 96),
        (3, 3, 96, 64),
        (3, 3, 64, 32),
        (3, 3, 32, 2),
    ]
    for k in expected_encoder + expected_decoder:
        assert k in shapes, f"missing conv kernel {k}"
    # exactly 6 encoder + 7 decoder kernels (shared, not duplicated per level)
    assert len(shapes) == 13


@pytest.mark.parametrize("refine_levels", [2, 4, 6])
def test_weight_sharing_param_count_invariant_to_levels(refine_levels):
    """Param count is independent of refine_levels (shared decoder)."""
    model = LimaModel(refine_levels=refine_levels, search_range=2)
    params = _init(model)
    # Encoder always builds all 6 levels; decoder is shared across levels.
    assert len(_kernel_shapes(params)) == 13
    # Closed-form from Tables I + III (R=2), conv weights + biases.
    assert _param_count(params) == 926_886


def test_search_range_sets_decoder_input_channels():
    """The decoder's first conv input is (2R+1)^2 + 2 channels."""
    for sr in (1, 2, 4):
        model = LimaModel(refine_levels=3, search_range=sr)
        shapes = _kernel_shapes(_init(model))
        first_in = (2 * sr + 1) ** 2 + 2
        assert any(s == (3, 3, first_in, 128) for s in shapes)


@pytest.mark.parametrize(
    "padding_mode", ["zeros", "replicate", "reflect", "circular"]
)
def test_padding_modes_run_and_match_shapes(padding_mode):
    """All padding schemes run (incl. 1x1 coarsest level) and match shapes."""
    model = LimaModel(
        refine_levels=6, search_range=2, padding_mode=padding_mode
    )
    params = _init(model)
    images = jax.random.normal(jax.random.PRNGKey(2), (1, 64, 64, 2)) * 30
    flows = model.apply({"params": params}, images)
    assert flows[-1].shape == (1, 32, 32, 2)
    assert jnp.all(jnp.isfinite(flows[-1]))


def test_padding_mode_changes_output():
    """Zero vs replicate padding produce different fields (LIMA0 vs LIMAR).

    Guards the central contribution of the 2025 paper: a regression that
    silently ignored padding_mode would pass the shape tests but fail here.
    """
    images = jax.random.normal(jax.random.PRNGKey(2), (1, 64, 64, 2)) * 30
    m_zeros = LimaModel(refine_levels=6, search_range=2, padding_mode="zeros")
    m_repl = LimaModel(
        refine_levels=6, search_range=2, padding_mode="replicate"
    )
    # Identical params so only the padding scheme differs.
    params = _init(m_zeros)
    out_zeros = m_zeros.apply({"params": params}, images)[-1]
    out_repl = m_repl.apply({"params": params}, images)[-1]
    assert out_zeros.shape == out_repl.shape
    assert not jnp.allclose(out_zeros, out_repl, atol=1e-5)


def test_forward_is_deterministic():
    """Repeated forward passes give identical results (no RNG in eval)."""
    model = LimaModel(refine_levels=4, search_range=2)
    params = _init(model)
    images = jax.random.normal(jax.random.PRNGKey(3), (1, 64, 64, 2)) * 10
    out1 = model.apply({"params": params}, images)
    out2 = model.apply({"params": params}, images)
    assert jnp.allclose(out1[-1], out2[-1])


def test_flow_init_seeds_coarsest_level():
    """A nonzero flow_init shifts the prediction (temporal warm start)."""
    model = LimaModel(refine_levels=6, search_range=2)
    params = _init(model)
    images = jax.random.normal(jax.random.PRNGKey(4), (1, 64, 64, 2)) * 10
    zero = model.apply({"params": params}, images)[-1]
    init = jnp.ones((1, 64, 64, 2)) * 3.0
    warm = model.apply({"params": params}, images, init)[-1]
    assert not jnp.allclose(zero, warm)
