import pytest
import jax.numpy as jnp
import numpy as np
from skimage import exposure

from flowgym.common.preprocess import (
    clahe,
    intensity_capping,
    intensity_clipping,
    high_pass_filter,
)


@pytest.mark.parametrize(
    "clip_limit, tile_grid_size, nbins",
    [
        (2.0, (8, 8), 256),
        (1.0, (16, 16), 128),
        (0.5, (32, 32), 64),
    ],
)
@pytest.mark.parametrize("image_shape", [(64, 64), (128, 128), (256, 256)])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_clahe_reference_batched(
    image_shape: tuple,
    clip_limit: float,
    tile_grid_size: tuple,
    nbins: int,
    seed: int,
    batch_size: int,
):
    # Generate a batch of random images
    rng = np.random.default_rng(seed)
    batch_shape = (batch_size,) + image_shape
    images = rng.random(batch_shape).astype(np.float32)

    # Build reference output by applying skimage CLAHE to each image independently
    min_val = np.min(images, axis=(1, 2), keepdims=True)
    max_val = np.max(images, axis=(1, 2), keepdims=True)
    images = (images - min_val) / (max_val - min_val)
    references = []
    for img in images:
        ref = exposure.equalize_adapthist(
            img, clip_limit=clip_limit, kernel_size=tile_grid_size, nbins=nbins
        )
        references.append(ref)
    reference_batch = np.stack(references, axis=0)

    # Run our CLAHE on the whole batch at once
    processed = clahe(
        jnp.asarray(images),
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        nbins=nbins,
    )
    processed = np.asarray(processed)

    # Check that the output shape matches (batch_size, H, W)
    assert (
        processed.shape == (batch_size,) + image_shape
    ), f"Expected output shape {(batch_size,) + image_shape}, got {processed.shape}"

    # Check that each frame in the batch matches the reference
    np.testing.assert_allclose(
        processed,
        reference_batch,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Batched processed output does not match reference per‐frame",
    )


def test_zero_image_returns_zero():
    img = jnp.zeros((16, 16))
    out = high_pass_filter(img[None, ...], sigma=0.1)
    # High-pass of zero image should be zero
    assert jnp.allclose(out, jnp.zeros_like(img)), "High-pass of zero image is not zero"


def test_constant_image_near_zero():
    img = jnp.ones((16, 16)) * 5.0
    out = high_pass_filter(img[None, ...], sigma=1.0)
    # High-pass of constant image should be (practically) zero
    assert jnp.allclose(
        out, jnp.zeros_like(img), atol=1e-6
    ), "High-pass of constant image is not near zero"


def test_shape_preserved():
    # Create a batch of 4 images of size 8x8
    batch = jnp.stack([jnp.eye(8) for _ in range(4)], axis=0)
    out = high_pass_filter(batch, 0.3)
    assert out.shape == batch.shape, "Batched filter did not preserve batch shape"


@pytest.mark.parametrize("cutoff", [0.1, 0.5])
def test_high_pass_effect_decreases_with_cutoff(cutoff):
    # Input impulse at center
    img = jnp.zeros((16, 16))
    img = img.at[8, 8].set(1.0)
    out = high_pass_filter(img[None, ...], sigma=cutoff)
    # The sum of absolute response should decrease as cutoff increases (wider gaussian)
    total = jnp.sum(jnp.abs(out))
    assert total < 1.0, f"Unexpected total response {total} for cutoff {cutoff}"


def test_intensity_capping_basic():
    # simple 1×2×2 image
    img = jnp.array([[[0.0, 1.0], [2.0, 3.0]]])
    # compute expected by applying the same formula
    median = jnp.median(img, axis=(1, 2), keepdims=True)
    std = jnp.std(img, axis=(1, 2), keepdims=True)
    upper = median + 1.0 * std
    expected = jnp.minimum(img, upper)

    out = intensity_capping(img, n=1.0)
    assert out.shape == img.shape
    # element‐wise close up to a reasonable tol
    assert jnp.allclose(out, expected, atol=1e-6)


def test_intensity_capping_n_zero_behaviour():
    # when n==0, upper_limit == median, so anything above median must be clipped to median
    img = jnp.array([[[10.0, 20.0], [30.0, 40.0]]])
    out = intensity_capping(img, n=0.0)
    # median of [10,20,30,40] = 25
    assert jnp.all(out <= 25.0)
    # values below median should pass through
    assert out[0, 0, 0] == pytest.approx(10.0)
    # values above should be exactly capped
    assert out[0, 1, 1] == pytest.approx(25.0)


def test_intensity_clipping_basic():
    img = jnp.array([[[0.0, 1.0], [2.0, 3.0]]])
    median = jnp.median(img, axis=(1, 2), keepdims=True)
    std = jnp.std(img, axis=(1, 2), keepdims=True)
    lower = median - std
    upper = median + std
    clipped = jnp.clip(img, lower, upper)

    out = intensity_clipping(img, n=1.0)
    assert out.shape == img.shape
    assert jnp.allclose(out, clipped, atol=1e-6)


def test_batch_processing_preserves_shape():
    # a batch of two 3×3 images
    img1 = jnp.arange(9).reshape(3, 3)
    img2 = jnp.arange(9, 18).reshape(3, 3)
    batch = jnp.stack([img1, img2])
    cap = intensity_capping(batch, n=1.0)
    clip = intensity_clipping(batch, n=1.0)
    assert cap.shape == batch.shape
    assert clip.shape == batch.shape
