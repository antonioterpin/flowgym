import pytest
import timeit
import cv2

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from flowgym.utils import load_configuration

from flowgym.flow.dis.process import (
    gather,
    sample_patch,
    extract_patches,
    compute_flow_level,
    warp_image,
    build_pyramid,
    flow_between,
    estimate_dis_flow,
    densify,
    extract_patches_grad_hess,
    img_resize,
    query_flow_at_points,
)

config = load_configuration("src/flowgym/config/testing.yaml")

REPETITIONS = config["REPETITIONS"]

EXECUTIONS_SAMPLE_PATCH = config["EXECUTIONS_SAMPLE_PATCH"]
EXECUTIONS_COMPUTE_FLOW_LEVEL = config["EXECUTIONS_COMPUTE_FLOW_LEVEL"]
EXECUTIONS_BUILD_PYRAMID = config["EXECUTIONS_BUILD_PYRAMID"]
EXECUTIONS_FLOW_BETWEEN = config["EXECUTIONS_FLOW_BETWEEN"]
EXECUTIONS_ESTIMATE_DIS_FLOW = config["EXECUTIONS_ESTIMATE_DIS_FLOW"]
EXECUTIONS_DENSIFY = config["EXECUTIONS_DENSIFY"]
EXECUTIONS_EXTRACT_PATCHES_GRAD_HESS = config["EXECUTIONS_EXTRACT_PATCHES_GRAD_HESS"]
EXECUTIONS_QUERY_FLOW_AT_POINTS = config["EXECUTIONS_QUERY_FLOW_AT_POINTS"]


def test_gather():
    # Create a sample image
    img = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Define coordinates
    y = jnp.array([[0, 1], [1, 2]])
    x = jnp.array([[1, 2], [0, 1]])

    # Expected result
    expected = jnp.array([[2, 6], [4, 8]])

    # Call the function
    result = gather(img, y, x)

    # Assert the result
    assert jnp.array_equal(result, expected), f"Expected {expected}, but got {result}"


# Test for sample_patch without displacement
def test_sample_patch_no_displacement():
    img = jnp.arange(16).reshape(4, 4)
    disp = jnp.array([0.0, 0.0])
    patch = sample_patch(img, disp, patch_size=3)
    expected = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 4.0, 5.0]])
    assert jnp.array_equal(patch, expected)


def test_sample_patch_with_displacement():
    img = jnp.arange(16).reshape(4, 4)
    disp = jnp.array([1.0, 1.0])
    patch = sample_patch(img, disp, patch_size=3)
    expected = jnp.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [8.0, 9.0, 10.0]])
    assert jnp.array_equal(patch, expected)


def test_sample_patch_with_float_displacement():
    img = jnp.arange(16).reshape(4, 4)
    disp = jnp.array([0.5, 0.5])
    patch = sample_patch(img, disp, patch_size=3)
    expected = jnp.array([[0.0, 0.25, 0.75], [1.0, 2.5, 3.5], [3.0, 6.5, 7.5]])
    assert jnp.array_equal(patch, expected)


def test_sample_patch_with_uneven_displacement():
    img = jnp.arange(16).reshape(4, 4)
    disp = jnp.array([0.5, 1.5])
    patch = sample_patch(img, disp, patch_size=3)
    expected = jnp.array([[0.25, 0.75, 1.25], [2.5, 3.5, 4.5], [6.5, 7.5, 8.5]])
    assert jnp.array_equal(patch, expected)


# Test for extract_patches
def test_extract_patches_basic():
    img = jnp.arange(16).reshape(4, 4)
    # Pad the image with zeros
    img_pad = jnp.pad(img[None, ...], ((0, 0), (1, 1), (1, 1)), mode="constant")
    patches = extract_patches(img_pad, patch_size=3, patch_stride=1).squeeze()
    # There should be 1 patch of shape (3,3)
    assert patches.shape == (16, 3, 3)

    # Top-left corner patch should be the top-left 3x3 of the image
    # which includes padding apart from the bottom right 2x2 corner
    assert jnp.array_equal(
        patches[0],
        jnp.array(
            [[0, 0, 0], [0, img[0, 0], img[0, 1]], [0, img[1, 0], img[1, 1]]],
            dtype=jnp.float32,
        ),
    )
    # Bottom-right corner patch should be the bottom-right 3x3 of the image
    # which includes padding apart from the top left 2x2 corner
    assert jnp.array_equal(
        patches[15],
        jnp.array(
            [[img[-2, -2], img[-2, -1], 0], [img[-1, -2], img[-1, -1], 0], [0, 0, 0]],
            dtype=jnp.float32,
        ),
    )


# Test for warp_image identity flow
def test_warp_image_identity():
    img = jnp.arange(9).reshape(3, 3)
    flow = jnp.zeros((3, 3, 2))
    warped = warp_image(img, flow, dt=1)
    assert jnp.array_equal(warped, img)


# Test for build_pyramid
def test_build_pyramid():
    img = jnp.arange(16).reshape(4, 4)
    pyr = build_pyramid(img, levels=2, start_level=0)
    assert len(pyr) == 2
    # Coarsest should be 2x2
    assert pyr[0].shape == (2, 2)
    # Finest equals original
    assert pyr[1].shape == (4, 4)


# Test for flow_between with no motion
def test_flow_between_no_movement():
    prev = jnp.ones((12, 12)).astype(jnp.float32)
    curr = jnp.ones((12, 12)).astype(jnp.float32)
    # with jax.disable_jit(): #activate this to be able to print variables
    flow = flow_between(
        prev,
        curr,
        levels=1,
        start_level=0,
        level_steps=1,
        grad_iters=1,
        patch_stride=1,
        patch_size=3,
        output_full_res=True,
        starting_flow=jnp.zeros((12, 12, 2), dtype=jnp.float32),
        var_refine_iters=0,
    )
    assert flow.shape == (12, 12, 2)
    assert jnp.allclose(flow, 0.0)


@pytest.mark.parametrize(
    "test_images",
    [
        {"shift": (1.0, 1.0), "image_size": (128, 128)},
        {"shift": (1.0, -1.0), "image_size": (128, 128)},
        {"shift": (-1.0, 1.0), "image_size": (128, 128)},
        {"shift": (-1.0, -1.0), "image_size": (128, 128)},
    ],
    indirect=True,
)
def test_flow_between(test_images, visualize=True):
    """Test for flow_between function."""
    prev, curr, true_shift = test_images
    levels = 2
    # with jax.disable_jit(): #activate this to bee able to print variables
    flow = flow_between(
        prev,
        curr,
        levels=levels,
        start_level=0,
        level_steps=1,
        grad_iters=4,
        patch_stride=5,
        patch_size=7,
        output_full_res=True,
        starting_flow=jnp.zeros((prev.shape[0], prev.shape[1], 2), dtype=jnp.float32),
        var_refine_iters=0,
    )

    # Add padding to remove border effects
    pad = 5
    cropped_flow = flow[pad:-pad, pad:-pad]

    if visualize:
        import matplotlib.pyplot as plt

        plt.imshow(cropped_flow[..., 0], cmap="gray")
        plt.colorbar()
        plt.title("Flow along x")
        plt.imsave("flow_x.png", cropped_flow[..., 0], cmap="gray")
        plt.imshow(cropped_flow[..., 1], cmap="gray")
        plt.colorbar()
        plt.title("Flow along y")
        plt.imsave("flow_y.png", cropped_flow[..., 1], cmap="gray")

    assert flow.shape == prev.shape + (2,)
    # Compute the average displacement in the cropped flow
    avg_disp_crop = jnp.mean(cropped_flow, axis=(0, 1))

    # Count the number of pixels with large errors
    counter = jnp.sum(
        (jnp.abs(cropped_flow[..., 0] - true_shift[0]) > 0.1)
        | (jnp.abs(cropped_flow[..., 1] - true_shift[1]) > 0.1)
    )
    assert jnp.abs(avg_disp_crop[0] - true_shift[0]) < 0.05, (
        f"Flow along x should match true shift, "
        f"average error = {avg_disp_crop[0] - true_shift[0]}"
    )
    assert jnp.abs(avg_disp_crop[1] - true_shift[1]) < 0.05, (
        f"Flow along y should match true shift, "
        f"average error = {avg_disp_crop[1] - true_shift[1]}"
    )
    assert (
        counter < 0.05 * cropped_flow.size
    ), f"Number of pixels with large error: {counter}"


# Test for estimate_dis_flow batch
def test_estimate_dis_flow_batch_identity():
    batch = jnp.ones((2, 4, 4))
    flows = estimate_dis_flow(
        batch,
        batch,
        levels=1,
        start_level=0,
        level_steps=1,
        grad_desc_iters=0,
        patch_stride=2,
        patch_size=3,
        output_full_res=True,
        starting_flow=jnp.zeros((2, 4, 4, 2), dtype=jnp.float32),
        var_refine_iters=0,
    )
    assert flows.shape == (2, 4, 4, 2)
    assert jnp.allclose(flows, 0.0)


@pytest.fixture(scope="module")
def test_images(request):
    """Fixture to create test images with a specified shift and size.

    A larger image is generated and the first image is generated by sampling from
    the center of the larger image. The second image is generated by sampling from
    the larger image with the opposite of the specified shift using the function
    sample_patch. It only works for square images.

    Args:
        request: pytest fixture request object containing parameters for the test.

    Returns:
        tuple: A tuple containing the first image, the second image, and the shift.
    """
    # Extract parameters from the request
    shift = request.param["shift"]
    image_size = request.param["image_size"]

    # Generate a larger image
    key = random.PRNGKey(0)  # TODO: improve the structure of the test

    # Movement along y --> larger height
    # Movement along x --> larger width
    bounds = jnp.array(
        [
            2 * jnp.ceil(jnp.abs(shift[1])).astype(int),  # larger height
            2 * jnp.ceil(jnp.abs(shift[0])).astype(int),  # larger width
        ]
    )
    larger_size = (image_size[0] + bounds[0], image_size[1] + bounds[1])  # (H, W)
    img_to_sample_from = random.uniform(key, shape=larger_size)

    # Extract the center of the image (y, x)
    center = jnp.array(
        [bounds[0] // 2 + image_size[0] // 2, bounds[1] // 2 + image_size[1] // 2]
    )

    # Sample the first image from the center of the larger image
    prev = sample_patch(
        img_to_sample_from,
        disp=center,
        patch_size=image_size[0],
    )

    shift_jnp = jnp.array(shift)
    shift_jnp = shift_jnp[::-1]

    # Sample the second image with the opposite shift
    # The shift is negated to simulate the displacement
    curr = sample_patch(
        img_to_sample_from,
        disp=center - shift_jnp,
        patch_size=image_size[0],
    )
    return prev, curr, shift


@pytest.mark.parametrize(
    "test_images",
    [
        {"shift": (1.0, 1.0), "image_size": (10, 10)},
    ],
    indirect=True,
)
@pytest.mark.parametrize("patch_size", [5])
@pytest.mark.parametrize("patch_stride", [4])
@pytest.mark.parametrize("grad_iters", [4])
def test_output_shape(test_images, patch_size, patch_stride, grad_iters):
    prev, curr, _ = test_images
    pp, centers, grads, hessians_inv = extract_patches_grad_hess(
        prev, patch_size=patch_size, patch_stride=patch_stride
    )
    starting_flow = jnp.zeros((centers.shape), dtype=jnp.float32)
    flow, _ = compute_flow_level(
        curr,
        grad_iters=grad_iters,
        patch_size=patch_size,
        starting_flow=starting_flow,
        pp=pp,
        centers=centers,
        grads=grads,
        hessians_inv=hessians_inv,
    )
    assert flow.shape == starting_flow.shape, "Output shape must be (num_patches, 2)"


@pytest.mark.parametrize(
    "patch_size, patch_stride, grad_iters",
    [
        (3, 1, 3),
        (5, 4, 1),
        (7, 3, 3),
        (9, 4, 1),
    ],
)
@pytest.mark.parametrize(
    "test_images", [{"shift": (0.0, 0.0), "image_size": (128, 128)}], indirect=True
)
def test_zero_flow_identity(patch_size, patch_stride, grad_iters, test_images):
    prev, curr, _ = test_images
    pp, centers, grads, hessians_inv = extract_patches_grad_hess(
        prev, patch_size=patch_size, patch_stride=patch_stride
    )
    starting_flow = jnp.zeros((centers.shape), dtype=jnp.float32)
    flow, _ = compute_flow_level(
        curr,
        grad_iters=grad_iters,
        patch_size=patch_size,
        starting_flow=starting_flow,
        pp=pp,
        centers=centers,
        grads=grads,
        hessians_inv=hessians_inv,
    )
    assert jnp.allclose(flow, 0, atol=1), "Flow should be zero for identical images"


@pytest.mark.parametrize(
    "test_images",
    [
        {"shift": (0.5, 1.0), "image_size": (128, 128)},
        {"shift": (0.5, -1.0), "image_size": (128, 128)},
        {"shift": (-1.0, 0.5), "image_size": (128, 128)},
        {"shift": (-1.0, -0.5), "image_size": (128, 128)},
    ],
    indirect=True,
)
@pytest.mark.parametrize("patch_size", [7])
@pytest.mark.parametrize("patch_stride", [1])
@pytest.mark.parametrize("grad_iters", [100])
def test_small_translation(test_images, patch_size, patch_stride, grad_iters):
    prev, curr, true_shift = test_images
    pp, centers, grads, hessians_inv = extract_patches_grad_hess(
        prev, patch_size=patch_size, patch_stride=patch_stride
    )
    starting_flow = jnp.zeros((centers.shape), dtype=jnp.float32)

    # TODO: add visualize option

    # Compute the flow using the function
    flow, errors = compute_flow_level(
        curr,
        grad_iters=grad_iters,
        patch_size=patch_size,
        starting_flow=starting_flow,
        pp=pp,
        centers=centers,
        grads=grads,
        hessians_inv=hessians_inv,
    )
    densify_flow = densify(prev.shape, flow, centers, patch_size, errors)

    # Add padding to remove border effects
    pad = 5
    cropped_flow = densify_flow[pad:-pad, pad:-pad]

    # Compute the average displacement in the cropped flow
    avg_disp_crop = jnp.mean(cropped_flow, axis=(0, 1))

    # Count the number of pixels with large errors
    counter = jnp.sum(
        (jnp.abs(cropped_flow[..., 0] - true_shift[0]) > 0.2)
        | (jnp.abs(cropped_flow[..., 1] - true_shift[1]) > 0.2)
    )
    assert jnp.abs(avg_disp_crop[0] - true_shift[0]) < 0.05, (
        f"Flow along x should match true shift, "
        f"average error = {avg_disp_crop[0] - true_shift[0]}"
    )
    assert jnp.abs(avg_disp_crop[1] - true_shift[1]) < 0.05, (
        f"Flow along y should match true shift, "
        f"average error = {avg_disp_crop[1] - true_shift[1]}"
    )
    assert (
        counter < 0.05 * cropped_flow.size
    ), f"Number of pixels with large error: {counter}"


@pytest.mark.parametrize(
    "test_images",
    [
        {"shift": (1.0, 1.0), "image_size": (10, 10)},
    ],
    indirect=True,
)
@pytest.mark.parametrize("patch_size", [5])
@pytest.mark.parametrize("patch_stride", [4])
@pytest.mark.parametrize("grad_iters", [5])
def test_dtype_and_finiteness(test_images, patch_size, patch_stride, grad_iters):
    prev, curr, _ = test_images
    pp, centers, grads, hessians_inv = extract_patches_grad_hess(
        prev, patch_size=patch_size, patch_stride=patch_stride
    )
    starting_flow = jnp.zeros((centers.shape), dtype=jnp.float32)
    flow, _ = compute_flow_level(
        curr,
        grad_iters=grad_iters,
        patch_size=patch_size,
        starting_flow=starting_flow,
        pp=pp,
        centers=centers,
        grads=grads,
        hessians_inv=hessians_inv,
    )
    assert flow.dtype == prev.dtype, "Flow dtype must match input dtype"
    assert jnp.isfinite(flow).all(), "Flow values must be finite"


@pytest.mark.parametrize("image_size", [(128, 128)])
@pytest.mark.parametrize("new_size", [(257, 257)])
def test_img_resize(image_size, new_size):
    img = jax.random.uniform(random.PRNGKey(0), shape=image_size)
    resized_img = img_resize(img, new_size)
    gound_truth = jax.image.resize(
        img, shape=new_size, method="bilinear", antialias=False
    )
    assert resized_img.shape == new_size, "Resized image shape mismatch"
    assert jnp.allclose(
        resized_img, gound_truth, atol=1e-5
    ), "Resized image values mismatch"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_size", [(1936, 1216)])
@pytest.mark.parametrize("patch_size", [31])
@pytest.mark.parametrize("disp", [(0.5, 0.5)])
def test_speed_sample_patch(
    image_size,
    patch_size,
    disp,
):
    """Test the speed of the sample_patch function."""
    disp = jnp.array(disp)
    # Limit time in seconds
    limit_time = 4e-5

    # Create a random image
    key = random.PRNGKey(0)
    img = random.uniform(key, shape=(image_size[0], image_size[1]))

    # Create the jit function
    sample_patch_jit = jax.jit(
        lambda img, disp: sample_patch(img, disp, patch_size=patch_size)
    )

    def run_jit_fn():
        patch = sample_patch_jit(img, disp)
        patch.block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_SAMPLE_PATCH,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_SAMPLE_PATCH

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_size", [(1936, 1216)])
@pytest.mark.parametrize("patch_size", [31])
@pytest.mark.parametrize("patch_stride", [20])
@pytest.mark.parametrize("grad_iters", [4])
@pytest.mark.parametrize("disp", [(0.5, 0.5)])
def test_speed_compute_flow_level(
    image_size,
    patch_size,
    patch_stride,
    disp,
    grad_iters,
):
    """Test the speed of the compute_flow_level function."""
    # Limit time in seconds
    limit_time = 9.4e-4

    # Create random parameters
    key = random.PRNGKey(0)
    prev = random.uniform(key, shape=(image_size[0], image_size[1]))
    curr = warp_image(prev, jnp.array(disp), 1)
    pp, centers, grads, hessians_inv = extract_patches_grad_hess(
        prev, patch_size=patch_size, patch_stride=patch_stride
    )
    starting_flow = jnp.zeros((centers.shape), dtype=jnp.float32)

    # Create the jit function
    compute_flow_level_jit = jax.jit(
        lambda curr, starting_flow, pp, centers, grads, h_inv: compute_flow_level(
            curr,
            grad_iters=grad_iters,
            patch_size=patch_size,
            starting_flow=starting_flow,
            pp=pp,
            centers=centers,
            grads=grads,
            hessians_inv=h_inv,
        )
    )

    def run_jit_fn():
        flow, error = compute_flow_level_jit(
            curr, starting_flow, pp, centers, grads, hessians_inv
        )
        flow.block_until_ready()
        error.block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_COMPUTE_FLOW_LEVEL,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_COMPUTE_FLOW_LEVEL

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_size", [(1024, 436)])
@pytest.mark.parametrize("levels", [3])
@pytest.mark.parametrize("start_level", [3])
@pytest.mark.parametrize("level_steps", [1])
def test_speed_build_pyramid(
    image_size,
    levels,
    start_level,
    level_steps,
):
    """Test the speed of the build_pyramid function."""
    # Limit time in seconds
    limit_time = 6e-5

    # Create a random image
    key = random.PRNGKey(0)
    img = random.uniform(key, shape=(image_size[0], image_size[1]))

    # Create the jit function
    build_pyramid_jit = jax.jit(
        lambda img: build_pyramid(
            img, levels=levels, start_level=start_level, steps=level_steps
        )
    )

    def run_jit_fn():
        pyr = build_pyramid_jit(img)
        [pyr[i].block_until_ready() for i in range(len(pyr))]

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_BUILD_PYRAMID,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_BUILD_PYRAMID

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_size", [(1024, 436)])
@pytest.mark.parametrize("levels", [3])
@pytest.mark.parametrize("start_level", [3])
@pytest.mark.parametrize("level_steps", [1])
@pytest.mark.parametrize("grad_iters", [16])
@pytest.mark.parametrize("patch_stride", [5])
@pytest.mark.parametrize("patch_size", [7])
@pytest.mark.parametrize("disp", [(0.5, 0.5)])
@pytest.mark.parametrize("output_full_res", [True])
@pytest.mark.parametrize("var_refine_iters", [0])
def test_speed_flow_between(
    image_size,
    levels,
    start_level,
    level_steps,
    grad_iters,
    patch_stride,
    patch_size,
    disp,
    output_full_res,
    var_refine_iters,
):
    """Test the speed of the flow_between function."""
    # Limit time in seconds
    limit_time = 5.3e-4

    # Create a random image
    key = random.PRNGKey(0)
    prev = random.uniform(key, shape=(image_size[0], image_size[1]))
    curr = warp_image(prev, jnp.array(disp), 1)
    starting_flow = jnp.zeros((image_size[0], image_size[1], 2), dtype=jnp.float32)

    # Create the jit function
    flow_between_jit = jax.jit(
        lambda prev, curr, starting_flow: flow_between(
            prev,
            curr,
            levels=levels,
            start_level=start_level,
            level_steps=level_steps,
            grad_iters=grad_iters,
            patch_stride=patch_stride,
            patch_size=patch_size,
            output_full_res=output_full_res,
            starting_flow=starting_flow,
            var_refine_iters=var_refine_iters,
        )
    )

    def run_jit_fn():
        flow = flow_between_jit(prev, curr, starting_flow)
        flow.block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_FLOW_BETWEEN,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_FLOW_BETWEEN

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_size", [(1024, 436)])
@pytest.mark.parametrize("levels", [3])
@pytest.mark.parametrize("start_level", [3])
@pytest.mark.parametrize("level_steps", [1])
@pytest.mark.parametrize("grad_iters", [16])
@pytest.mark.parametrize("patch_stride", [5])
@pytest.mark.parametrize("patch_size", [7])
@pytest.mark.parametrize("disp", [(0.5, 0.5)])
@pytest.mark.parametrize("output_full_res", [True])
@pytest.mark.parametrize("var_refine_iters", [0])
def test_speed_estimate_dis_flow(
    batch_size,
    image_size,
    levels,
    start_level,
    level_steps,
    grad_iters,
    patch_stride,
    patch_size,
    disp,
    output_full_res,
    var_refine_iters,
):
    """Test the speed of the estimate_dis_flow function."""
    # Limit time in seconds
    limit_time = 5.5e-4

    # Create a random image
    key = random.PRNGKey(0)
    prev_batch = random.uniform(key, shape=(batch_size, image_size[0], image_size[1]))
    curr_batch = jax.vmap(warp_image, in_axes=(0, None, None))(
        prev_batch, jnp.array(disp), 1
    )
    starting_flow = jnp.zeros(
        (batch_size, image_size[0], image_size[1], 2), dtype=jnp.float32
    )

    # Create the jit function
    estimate_dis_flow_jit = jax.jit(
        lambda prev_batch, curr_batch, starting_flow: estimate_dis_flow(
            prev_batch,
            curr_batch,
            levels=levels,
            start_level=start_level,
            level_steps=level_steps,
            grad_desc_iters=grad_iters,
            patch_stride=patch_stride,
            patch_size=patch_size,
            output_full_res=output_full_res,
            starting_flow=starting_flow,
            var_refine_iters=var_refine_iters,
        )
    )

    def run_jit_fn():
        flow_batch = estimate_dis_flow_jit(prev_batch, curr_batch, starting_flow)
        flow_batch.block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_ESTIMATE_DIS_FLOW,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_ESTIMATE_DIS_FLOW

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_shape", [(1936, 1216)])
@pytest.mark.parametrize("num_patches", [5917])
@pytest.mark.parametrize("patch_size", [31])
def test_speed_densify(
    image_shape,
    num_patches,
    patch_size,
):
    """Test the speed of the densify function."""
    # Limit time in seconds
    limit_time = 4.2e-4

    # Create random sparse flow centers and errors
    key = random.PRNGKey(0)
    centers_y = random.randint(key, num_patches, 0, image_shape[0])
    centers_x = random.randint(key, num_patches, 0, image_shape[1])
    centers = jnp.stack((centers_y, centers_x), axis=-1)
    sparse_flow = random.uniform(key, shape=(num_patches, 2))
    patch_errors = random.uniform(key, shape=(num_patches, patch_size, patch_size))

    # Create the jit function
    densify_jit = jax.jit(
        lambda sparse_flow, centers, patch_errors: densify(
            image_shape=image_shape,
            sparse_flow=sparse_flow,
            centers=centers,
            patch_size=patch_size,
            patch_errors=patch_errors,
        )
    )

    def run_jit_fn():
        densified_flow = densify_jit(
            sparse_flow,
            centers,
            patch_errors,
        )
        densified_flow.block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_DENSIFY,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_DENSIFY

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_shape", [(128, 54)])
@pytest.mark.parametrize("patch_size", [9])
@pytest.mark.parametrize("patch_stride", [6])
def test_speed_extract_patches_grad_hess(
    image_shape,
    patch_size,
    patch_stride,
):
    """Test the speed of the extract_patches_hess_grad function."""
    # Limit time in seconds
    limit_time = 1.5e-3

    # Create random image
    key = random.PRNGKey(0)
    image = random.uniform(key, shape=image_shape)

    # Create the jit function
    extract_patches_grad_hess_jit = jax.jit(
        lambda image: extract_patches_grad_hess(
            img=image, patch_size=patch_size, patch_stride=patch_stride, eps=1e-4
        )
    )

    def run_jit_fn():
        patches = extract_patches_grad_hess_jit(image)
        patches[0].block_until_ready()
        patches[1].block_until_ready()
        patches[2].block_until_ready()
        patches[3].block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_EXTRACT_PATCHES_GRAD_HESS,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_EXTRACT_PATCHES_GRAD_HESS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


# skipif is used to skip the test if the user is not connected to the server
@pytest.mark.skipif(
    not all(d.device_kind == "NVIDIA GeForce RTX 4090" for d in jax.devices()),
    reason="user not connect to the server.",
)
@pytest.mark.parametrize("image_shape", [(1936, 1216)])
@pytest.mark.parametrize("next_level", [0])
@pytest.mark.parametrize("level", [1])
@pytest.mark.parametrize("patch_size", [31])
@pytest.mark.parametrize("patch_stride", [20])
def test_speed_query_flow_at_points(
    image_shape, next_level, level, patch_size, patch_stride
):
    """Test the speed of the query_flow_at_points function."""
    # Limit time in seconds
    limit_time = 3.1e-4

    # Create patches
    key = random.PRNGKey(0)
    subkey1, subkey2, subkey3 = random.split(key, 3)
    prev = random.uniform(
        subkey1,
        shape=(image_shape[0] // (2**next_level), image_shape[1] // (2**next_level)),
    )

    _, next_centers, _, _ = extract_patches_grad_hess(
        prev, patch_size=patch_size, patch_stride=patch_stride
    )
    img_resized = img_resize(
        prev, (image_shape[0] // (2**level), image_shape[1] // (2**level))
    )
    pp, centers, _, _ = extract_patches_grad_hess(
        img_resized, patch_size=patch_size, patch_stride=patch_stride
    )

    errors = random.uniform(subkey2, shape=pp.shape)
    sparse_flow = random.uniform(subkey3, shape=centers.shape)

    # Create the jit function
    query_flow_at_points_jit = jax.jit(
        lambda flow, centers, errors, query_coords: query_flow_at_points(
            sparse_flow=flow,
            centers=centers,
            patch_size=patch_size,
            patch_errors=errors,
            query_coords=query_coords,
        )
    )

    def run_jit_fn():
        queried_flow = query_flow_at_points_jit(
            sparse_flow,
            centers,
            errors,
            next_centers,
        )
        queried_flow.block_until_ready()

    # Warm up the function
    run_jit_fn()

    # Measure the time of the jit function
    total_time_jit = timeit.repeat(
        stmt=run_jit_fn,
        number=EXECUTIONS_QUERY_FLOW_AT_POINTS,
        repeat=REPETITIONS,
    )

    # Average time
    average_time_jit = min(total_time_jit) / EXECUTIONS_QUERY_FLOW_AT_POINTS

    # Check if the time is less than the limit
    assert (
        average_time_jit < limit_time
    ), f"The average time is {average_time_jit}, time limit: {limit_time}"


def test_error_vs_opencv_dis():
    """Test the error between the flow computed by the function and by OpenCV."""
    # Because of implementation differences the error is not zero, but it should be small.
    # For example, the error

    # Create a random image
    shape = (128, 128)
    key = random.PRNGKey(0)
    img = random.uniform(key, shape=shape)

    # Create a flow with a small displacement
    flow = jnp.ones((shape[0], shape[1], 2), dtype=jnp.float32) * jnp.array([1, 1])
    # Warp the image using the flow
    warped_img = warp_image(img, flow, dt=1)

    # Set the parameters for OpenCV and for dis_jax
    patch_size = 17
    patch_stride = 5
    levels = 3
    start_level = 0
    grad_desc_iters = 16

    dis = cv2.DISOpticalFlow_create(1)  # type: ignore
    dis.setPatchSize(patch_size)
    dis.setPatchStride(patch_stride)
    dis.setGradientDescentIterations(grad_desc_iters)
    dis.setVariationalRefinementIterations(0)
    dis.setFinestScale(start_level)
    dis.setUseSpatialPropagation(False)

    # Compute the flow using OpenCV
    cv_flow = dis.calc(
        np.clip(np.asarray(img) * 255.0, 0, 255).astype(np.uint8),
        np.clip(np.asarray(warped_img) * 255.0, 0, 255).astype(np.uint8),
        None,
    )

    dis_jax_flow = flow_between(
        img,
        warped_img,
        levels=levels,
        start_level=start_level,
        level_steps=1,
        grad_iters=grad_desc_iters,
        patch_stride=patch_stride,
        patch_size=patch_size,
        output_full_res=True,
        starting_flow=jnp.zeros((img.shape[0], img.shape[1], 2), dtype=jnp.float32),
        var_refine_iters=0,
    )
    cv_flow = cv_flow.reshape(*shape, 2)

    error_wrt_gt_dis_jax = jnp.mean(
        jnp.linalg.norm(dis_jax_flow[17:-17, 17:-17] - flow[17:-17, 17:-17], axis=-1)
    )

    error_wrt_gt_cv = jnp.mean(
        jnp.linalg.norm(cv_flow[17:-17, 17:-17] - flow[17:-17, 17:-17], axis=-1)
    )

    # Compute the error between the two flows
    # excluding the borders because we handle them differently
    error = jnp.mean(
        jnp.linalg.norm(dis_jax_flow[17:-17, 17:-17] - cv_flow[17:-17, 17:-17], axis=-1)
    )

    assert error < 0.01, (
        f"Error is too high: {error}, "
        f"Error OpenCV flow: {error_wrt_gt_cv}, "
        f"Error dis_jax flow: {error_wrt_gt_dis_jax}"
    )
