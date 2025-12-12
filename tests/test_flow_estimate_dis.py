"""Test the DISFlowFieldEstimator class."""

import pytest
import numpy as np
import jax.numpy as jnp

from flowgym.flow.dis import DISFlowFieldEstimator


@pytest.fixture
def dis_estimator():
    """Create a DISFlowFieldEstimator with sane defaults."""
    return DISFlowFieldEstimator(
        preset=1,
        patch_size=8,
        patch_stride=4,
        grad_desc_iters=4,
        var_refine_iters=0,
        use_mean_normalization=True,
        use_spatial_propagation=True,
    )


@pytest.mark.parametrize("preset", [-1, 4, "fast"])
def test_invalid_preset_raises(preset):
    with pytest.raises(ValueError):
        DISFlowFieldEstimator(
            preset=preset,
        )


@pytest.mark.parametrize("patch_size", [0, -5, 3.2])
def test_invalid_patch_size_raises(patch_size):
    with pytest.raises(ValueError):
        DISFlowFieldEstimator(
            patch_size=patch_size,
        )


@pytest.mark.parametrize("patch_stride", [0, -1, 2.5, []])
def test_invalid_patch_stride_raises(patch_stride):
    with pytest.raises(ValueError):
        DISFlowFieldEstimator(
            patch_stride=patch_stride,
        )


@pytest.mark.parametrize(
    "iters,name",
    [
        (-1, "grad_desc_iters"),
        (-2, "var_refine_iters"),
    ],
)
def test_invalid_iteration_counts_raise(iters, name):
    kwargs = dict(
        preset=1,
        patch_size=8,
        patch_stride=4,
        grad_desc_iters=4,
        var_refine_iters=0,
    )
    kwargs[name] = iters
    with pytest.raises(ValueError):
        DISFlowFieldEstimator(**kwargs)  # type: ignore


def test_call_on_constant_images_returns_zero_flow(dis_estimator):
    batch, H, W = 2, 16, 16
    # initial frame and current frame identical zeros
    init_image = jnp.zeros((batch, H, W), dtype=jnp.float32)
    # create initial state (history filled with init_image twice)
    state = dis_estimator.create_state(
        init_image, image_history_size=1, estimates=jnp.zeros((batch, H, W, 2))
    )

    # call estimator on same zeros -> flow should be zero
    state, _ = dis_estimator(init_image, state, None)
    flow = state["estimates"][:, -1, ...]

    # check output type and shape
    assert isinstance(flow, jnp.ndarray)
    assert flow.shape == (batch, H, W, 2)

    # all displacements should be zero
    np.testing.assert_allclose(np.asarray(flow), 0.0, atol=1e-6)


def test_call_on_simple_shift(dis_estimator):
    _, H, W = 1, 32, 32
    rng = np.random.RandomState(0)
    # random texture ensures DIS can match
    base = rng.randint(0, 256, size=(H, W), dtype=np.uint8)
    base_f = base.astype(np.float32) / 255.0
    shifted = np.roll(base, shift=1, axis=1).astype(np.float32) / 255.0

    # bootstrap two identical frames
    init_image = jnp.asarray(base_f[np.newaxis, ...])
    state = dis_estimator.create_state(
        init_image, image_history_size=1, estimates=jnp.zeros((_, H, W, 2))
    )
    state, _ = dis_estimator(init_image, state, None)

    # now provide shifted frame
    curr_image = jnp.asarray(shifted[np.newaxis, ...])
    state, _ = dis_estimator(curr_image, state, None)
    flow = state["estimates"][0, -1]
    flow_np = np.asarray(flow)

    u = flow_np[..., 0]
    v = flow_np[..., 1]

    # expect a rightward displacement: median u > 0.5
    assert np.median(u) > 0.9, f"Median horizontal flow too small: {np.median(u)}"
    assert (
        np.median(np.abs(v)) < 0.1
    ), f"Median vertical flow too large: {np.median(np.abs(v))}"
