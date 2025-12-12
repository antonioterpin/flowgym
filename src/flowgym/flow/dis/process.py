"""Module for PIV images processing in JAX."""

import jax
import jax.numpy as jnp
from jax import lax

from flowgym.common.filters import sobel
from flowgym.flow.utils import inv_hessian
from flowgym.flow.process import img_resize
from flowgym.flow.process import apply_flow_to_image_backward as warp_image


def estimate_dis_flow(
    prev_batch: jnp.ndarray,
    curr_batch: jnp.ndarray,
    start_level: int,
    levels: int,
    level_steps: int,
    grad_desc_iters: int,
    patch_stride: int,
    patch_size: int,
    output_full_res: bool,
    starting_flow: jnp.ndarray,
    var_refine_iters: int,
) -> jnp.ndarray:
    """Batch version of the DIS optical flow estimation algorithm.

    Args:
        prev_batch: batch of previous images (B, H, W)
        curr_batch: batch of current images (B, H, W)
        start_level: starting level of the pyramid
        levels: number of pyramid levels
        level_steps: Number of steps between levels.
        grad_desc_iters: number of gradient descent iterations
        patch_stride: stride between patches
        patch_size: size of the patches to extract
        output_full_res: if True, output the flow at full resolution
        starting_flow: initial flow of shape (B, H, W, 2)
        var_refine_iters: number of iterations for the variational refinement

    Returns:
        estimated batch of flow field (B, H, W, 2)
    """
    return jax.vmap(
        flow_between, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None)
    )(
        prev_batch,
        curr_batch,
        starting_flow,
        start_level,
        levels,
        level_steps,
        grad_desc_iters,
        patch_stride,
        patch_size,
        output_full_res,
        var_refine_iters,
    )


def variational_refinement(
    I0: jnp.ndarray,
    I1: jnp.ndarray,
    flow: jnp.ndarray,
    patch_size: int,
    fixed_iters: int,
    *,
    sor_iters: int = 5,
    alpha: float = 10.0,
    gamma: float = 10.0,
    delta: float = 5.0,
    omega: float = 1.6,
    zeta: float = 0.1,
    eps: float = 0.001,
) -> jnp.ndarray:
    """Refines the optical flow using a variational approach.

    Args:
        I0: first image of shape (H,W).
        I1: second image of shape (H,W).
        flow: initial flow of shape (H,W,2).
        patch_size: size of the patches to extract.
        fixed_iters: number of outer fixed-point iterations.
        sor_iters: number of inner SOR iterations.
        alpha: smoothness weight.
        gamma: gradient constancy weight.
        delta: data constancy weight.
        omega: SOR relaxation factor.
        zeta: small constant for smoothness weight.
        eps: small constant for robust norms.
    """
    # Gaussian kernel for structure tensor
    # TODO: try to precompute this
    box_filter_kernel = jnp.ones((patch_size, patch_size)) / (patch_size**2)

    # unpack flow
    u, v = flow[..., 0], flow[..., 1]
    H, W = u.shape

    # structure tensor on I0
    # TODO: try to precompute this
    I0xx, I0xy, I0yy = prepare_structure_tensor(I0, box_filter_kernel)

    # build a fixed red/black mask
    ii, jj = jnp.ogrid[:H, :W]
    red_mask = (ii + jj) % 2 == 0

    def fp_step(_, uv):
        u, v = uv

        flow = jnp.stack([u, v], axis=-1)

        # warp I1 → I1w with current (u,v)
        I1w = warp_image(I1, -flow)  # TODO verify this is correct, invert flow

        # compute residuals
        Ix, Iy = sobel_grad(I1w)
        Iz = I1w - I0

        # robust norms
        PsiI = jnp.sqrt(Iz**2 + eps**2)
        PsiG = jnp.sqrt((Ix**2 + Iy**2) + eps**2)

        # smoothness weights (same formula OpenCV uses)
        u_x, u_y = sobel_grad(u)
        v_x, v_y = sobel_grad(v)
        w_s = 1.0 / jnp.sqrt(u_x**2 + u_y**2 + v_x**2 + v_y**2 + zeta**2)

        # assemble per-pixel linear system coefficients exactly like OpenCV:
        Au11 = delta / PsiI + gamma * (I0xx / PsiG) + alpha * w_s
        Au22 = delta / PsiI + gamma * (I0yy / PsiG) + alpha * w_s
        Au12 = gamma * (I0xy / PsiG)

        bu = -(Ix * Iz) / PsiI
        bv = -(Iy * Iz) / PsiI

        # initialize increments
        dWu = jnp.zeros_like(u)
        dWv = jnp.zeros_like(v)

        # inner SOR: alternate red/black passes
        def sor_body(_, dW):
            dWu, dWv = dW
            # red pass
            dWu, dWv = sor_pass(dWu, dWv, Au11, Au12, Au22, bu, bv, omega, red_mask)
            # black pass
            dWu, dWv = sor_pass(dWu, dWv, Au11, Au12, Au22, bu, bv, omega, ~red_mask)
            return (dWu, dWv)

        dWu, dWv = jax.lax.fori_loop(0, sor_iters, sor_body, (dWu, dWv))

        # update flow
        return (u + dWu, v + dWv)

    # outer fixed-point
    u, v = jax.lax.fori_loop(0, fixed_iters, fp_step, (u, v))
    return jnp.stack([u, v], axis=-1)


def sobel_grad(img: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the Sobel gradient of an image using a separable kernel.

    Args:
        img: (H,W) single-channel image.

    Returns:
        - gradient in x direction of shape (H,W).
        - gradient in y direction of shape (H,W).
    """
    sobel_x, sobel_y = sobel()
    Ix = conv2d(img, sobel_x)
    Iy = conv2d(img, sobel_y)
    return Ix, Iy


# TODO: test jax.scipy.signal.fftconvolve on big patches
def prepare_structure_tensor(
    I0: jnp.ndarray, kernel: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the structure tensor of an image using Sobel gradients.

    Args:
        I0: (H,W) single-channel image.
        kernel: (kh,kw) single-channel kernel for smoothing.
    """
    I0x, I0y = sobel_grad(I0)
    # Convolve squared products
    Ixx = conv2d(I0x * I0x, kernel)
    Ixy = conv2d(I0x * I0y, kernel)
    Iyy = conv2d(I0y * I0y, kernel)
    return Ixx, Ixy, Iyy


def conv2d(img: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Perform a 2D convolution on a single-channel image using a separable kernel.

    Args:
        img: (H,W) single-channel image.
        kernel: (kh,kw) single-channel kernel.

    Returns:
        Convolved image of shape (H,W).
    """
    # reshape: (N,H,W,C) and (kh,kw,IC,OC)
    kh, kw = kernel.shape
    # Add batch and channel dims: NHWC
    img_nhwc = img[None, ..., None]  # shape (1,H,W,1)
    # Kernel: HWIO where in_ch=1, out_ch=1
    kernel_hwio = kernel[..., None, None]
    # Convolution
    out = lax.conv_general_dilated(
        lhs=img_nhwc,
        rhs=kernel_hwio,
        window_strides=(1, 1),
        padding=[(kh // 2, kh // 2), (kw // 2, kw // 2)],  # SAME padding
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return out[0, ..., 0]


def sor_pass(
    dWu: jnp.ndarray,
    dWv: jnp.ndarray,
    Au11: jnp.ndarray,
    Au12: jnp.ndarray,
    Au22: jnp.ndarray,
    bu: jnp.ndarray,
    bv: jnp.ndarray,
    omega: float,
    red_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Performs a single SOR pass on the red/black checkerboard.

    Args:
        dWu: current u displacement increments.
        dWv: current v displacement increments.
        Au11: linear system coefficients.
        Au12: linear system coefficients.
        Au22: linear system coefficients.
        bu: linear system first right-hand side.
        bv: linear system second right-hand side.
        omega: relaxation factor.
        red_mask: boolean mask for the red checkerboard.
    """
    # compute the unconstrained updates
    upd_u = (bu - Au12 * dWv) / Au11
    upd_v = (bv - Au12 * upd_u) / Au22

    # relaxation
    new_dWu = dWu + omega * (upd_u - dWu)
    new_dWv = dWv + omega * (upd_v - dWv)

    # only apply on the selected checkerboard
    dWu = jnp.where(red_mask, new_dWu, dWu)
    dWv = jnp.where(red_mask, new_dWv, dWv)
    return dWu, dWv


def flow_between(
    prev: jnp.ndarray,
    curr: jnp.ndarray,
    starting_flow: jnp.ndarray,
    start_level: int,
    levels: int,
    level_steps: int,
    grad_iters: int,
    patch_stride: int,
    patch_size: int,
    output_full_res: bool,
    var_refine_iters: int,
) -> jnp.ndarray:
    """Compute dense flow between two images using the DIS algorithm.

    The function is implemented using jax.lax.switch nested in a jax.lax.scan loop.
    The reason for this is to index the pyramid levels with a python int and now with
    a jax tracer. This is important for the JIT compilation to work correctly.

    Args:
        prev: Previous image of shape (H, W)
        curr: Current image of shape (H, W)
        starting_flow: Initial flow of shape (H, W, 2)
        start_level: Starting level of the pyramid
        levels: Number of pyramid levels
        level_steps: Number of steps between levels.
        grad_iters: Number of gradient descent iterations
        patch_stride: Stride between patches
        patch_size: Size of the patches to extract
        output_full_res: If True, output the flow at full resolution
        var_refine_iters: Number of iterations for variational refinement

    Returns:
        Estimated flow field of shape (H, W, 2)
    """
    # Build image pyramids as python lists (coarse to fine)
    prev_pyr = build_pyramid(
        prev, levels=levels, start_level=start_level, steps=level_steps
    )
    curr_pyr = build_pyramid(
        curr, levels=levels, start_level=start_level, steps=level_steps
    )

    # Extract patches, centers, gradients and inverted hessians for each level
    pp, centers, grads, hessians_inv = [], [], [], []

    for level in range(len(prev_pyr)):
        pp_level, centers_level, grads_level, hessians_inv_level = (
            extract_patches_grad_hess(
                prev_pyr[level], patch_size=patch_size, patch_stride=patch_stride
            )
        )
        pp.append(pp_level)
        centers.append(centers_level)
        grads.append(grads_level)
        hessians_inv.append(hessians_inv_level)

    # Wrap each level's processing in a function
    def make_level_fn(level):
        def fn(input):
            flow, final_errors = input

            # Get the number of patches at the current level
            num_patches = pp[level].shape[0]

            if level == levels - 1:

                flow_curr = flow[:num_patches, :]

                if level > 0:
                    flow_curr = flow_curr * 2**level_steps

                # Compute the flow at the current level
                next_flow, errors = compute_flow_level(
                    curr_pyr[level],
                    grad_iters,
                    patch_size,
                    flow_curr,
                    pp[level],
                    centers[level],
                    grads[level],
                    hessians_inv[level],
                )
                final_errors = errors
                updated = next_flow

                # For the last level, we don't need to query the next level
                next_num_patches = centers[level].shape[0]

            elif level == 0:

                flow_curr = flow[:num_patches]

                # Compute the flow at the current level
                next_flow, errors = compute_flow_level(
                    curr_pyr[level],
                    grad_iters,
                    patch_size,
                    flow_curr,
                    pp[level],
                    centers[level],
                    grads[level],
                    hessians_inv[level],
                )

                if var_refine_iters > 0:
                    # Densify the flow
                    next_flow = densify(
                        (curr_pyr[0].shape[0], curr_pyr[0].shape[1]),
                        next_flow,
                        centers[0],
                        patch_size,
                        errors,
                    )

                    # Apply variational refinement
                    next_flow = variational_refinement(
                        prev_pyr[0],
                        curr_pyr[0],
                        next_flow,
                        patch_size,
                        var_refine_iters,
                    )

                    # Compute the flow at the next level's patches center
                    updated = gather_2d(next_flow, centers[1] // (2**level_steps))
                else:
                    # Compute the flow at the next level's patches center
                    updated = query_flow_at_points(
                        next_flow,
                        centers[0] * (2**level_steps),
                        (patch_size // (2**level_steps)) * 2 * (2**level_steps) + 1,
                        errors,
                        centers[1],
                    )

                next_num_patches = centers[level + 1].shape[0]

            else:

                # Crop the flow coming from the previous level's estimate
                flow_curr = flow[:num_patches]

                # the next level's flow is twice the size of the current level
                flow_curr = flow_curr * 2**level_steps

                # Compute the flow at the current level
                next_flow, errors = compute_flow_level(
                    curr_pyr[level],
                    grad_iters,
                    patch_size,
                    flow_curr,
                    pp[level],
                    centers[level],
                    grads[level],
                    hessians_inv[level],
                )

                if var_refine_iters > 0:
                    # Densify the flow
                    next_flow = densify(
                        curr_pyr[level].shape,
                        next_flow,
                        centers[level],
                        patch_size,
                        errors,
                    )

                    # Apply variational refinement
                    next_flow = variational_refinement(
                        prev_pyr[level],
                        curr_pyr[level],
                        next_flow,
                        patch_size,
                        var_refine_iters,
                    )

                    # Compute the flow at the next level's patches center
                    updated = gather_2d(
                        next_flow, centers[level + 1] // (2**level_steps)
                    )
                else:
                    # Compute the flow at the next level's patches center
                    updated = query_flow_at_points(
                        next_flow,
                        centers[level] * (2**level_steps),
                        (patch_size // (2**level_steps)) * 2 * (2**level_steps) + 1,
                        errors,
                        centers[level + 1],
                    )

                next_num_patches = centers[level + 1].shape[0]
            # Update the flow field with the new flow
            flow = flow.at[:next_num_patches, :].set(updated)

            return (flow, final_errors), None

        return fn

    # Initialize the flow field
    flow = jnp.zeros((pp[levels - 1].shape[0], 2))
    final_errors = jnp.zeros(pp[levels - 1].shape)

    # Gather the starting flow at the centers of the coarsest level
    s = starting_flow.shape[0] // prev_pyr[0].shape[0]  # scale factor
    sparse_starting_flow = gather_2d(starting_flow, centers[0] * s) / s

    # Set the starting flow for the coarsest level
    flow = flow.at[: centers[0].shape[0], :].set(sparse_starting_flow)

    # Create a list of functions for each level
    level_fns = [make_level_fn(level) for level in range(levels)]

    # Create a function to process each level
    def body_fn(input, lvl):
        flow, final_errors = input
        (flow, final_errors), _ = jax.lax.switch(lvl, level_fns, (flow, final_errors))
        return (flow, final_errors), None

    # Iterate over the levels from coarsest to finest
    (flow, final_errors), _ = lax.scan(
        body_fn, (flow, final_errors), jnp.arange(levels)
    )

    # Densify the flow
    flow = densify(
        (curr_pyr[levels - 1].shape[0], curr_pyr[levels - 1].shape[1]),
        flow,
        centers[levels - 1],
        patch_size,
        final_errors,
    )

    # Apply variational refinement
    if var_refine_iters > 0:
        flow = variational_refinement(
            prev_pyr[levels - 1],
            curr_pyr[levels - 1],
            flow,
            patch_size,
            var_refine_iters,
        )

    # Resize the flow to match the original image size
    if start_level > 0 and output_full_res:
        resized_flow_x = img_resize(flow[..., 0], (prev.shape[0], prev.shape[1]))
        resized_flow_y = img_resize(flow[..., 1], (prev.shape[0], prev.shape[1]))
        flow = jnp.stack((resized_flow_x, resized_flow_y), axis=-1) * 2**start_level

    return flow


def densify(
    image_shape: tuple[int, int],
    sparse_flow: jnp.ndarray,
    centers: jnp.ndarray,
    patch_size: int,
    patch_errors: jnp.ndarray,
) -> jnp.ndarray:
    """Densification step for optical flow estimation.

    This function takes the sparse flow estimates and their corresponding
    centers, and computes a dense flow field by averaging the flow estimates
    within a patch around each center. The averaging is weighted by the
    corresponding patch errors.

    Args:
        image_shape: shape of the image (H, W).
        sparse_flow: sparse flow estimates of shape (num_patches, 2).
        centers: centers of the patches of shape (num_patches, 2).
        patch_size: size of the patches to extract.
        patch_errors: errors associated with each pixel of the patch.

    Returns:
        dense_flow: dense flow field of shape (H, W, 2).
    """
    H, W = image_shape
    half_patch = patch_size // 2

    # Initialize the flow and weight accumulators with padding
    flow_acc = jnp.zeros((H, W, 2), dtype=jnp.float32)
    weight_acc = jnp.zeros((H, W), dtype=jnp.float32)

    def single_flow_update(i):
        cy, cx = centers[i]
        flow = sparse_flow[i]
        error = patch_errors[i]

        # Compute and clip patch boundaries
        y0, y1 = -half_patch, half_patch + 1
        x0, x1 = -half_patch, half_patch + 1

        # Compute the weights based on the error
        weights = 1.0 / jnp.maximum(1.0, jnp.abs(error.reshape(-1)))

        # Create coordinates and updates
        ys = jnp.arange(y0, y1) + cy
        xs = jnp.arange(x0, x1) + cx
        coords = jnp.stack(jnp.meshgrid(ys, xs, indexing="ij"), axis=-1).reshape(-1, 2)
        updates = flow * weights[:, None]

        # Valid mask: check if each coord is within image bounds
        mask_y = (coords[:, 0] >= 0) & (coords[:, 0] < H)
        mask_x = (coords[:, 1] >= 0) & (coords[:, 1] < W)
        valid_mask = mask_y & mask_x

        # Zero out coords outside image to avoid undefined indices
        coords = jnp.where(valid_mask[:, None], coords, 0)
        updates = jnp.where(valid_mask[:, None], updates, 0.0)
        weights = jnp.where(valid_mask, weights, 0.0)

        return coords, updates, weights

    # Vectorize the update step across all patches
    coords_all, updates_all, weights_all = jax.vmap(single_flow_update)(
        jnp.arange(centers.shape[0])
    )

    # Flatten and accumulate
    coords_flat = coords_all.reshape(-1, 2)
    updates_flat = updates_all.reshape(-1, 2)
    weights_flat = weights_all.reshape(-1)

    flow_acc = flow_acc.at[coords_flat[:, 0], coords_flat[:, 1]].add(updates_flat)
    weight_acc = weight_acc.at[coords_flat[:, 0], coords_flat[:, 1]].add(weights_flat)

    dense_flow = flow_acc / weight_acc[..., None]

    return dense_flow


def build_pyramid(
    img: jnp.ndarray, levels: int, start_level: int, steps: int = 1
) -> list[jnp.ndarray]:
    """Build an image pyramid from coarsest to finest resolution.

    The pyramid is built by downsampling the image by a factor of 2 at each level.

    Args:
        img: jnp.ndarray of shape (H, W) or (H, W, C)
        levels: number of pyramid levels
        start_level: starting level of the pyramid
        steps: number of steps between levels

    Returns:
        List of jnp.ndarrays from coarsest to finest.
    """
    H, W = img.shape[:2]

    def downsample(level):
        """Downsample the image by a factor of 2^level.

        Args:
            level: level of the pyramid

        Returns:
            downsampled image of shape (H // factor, W // factor)
        """
        factor = 2 ** (level * steps + start_level)
        shape = (H // factor, W // factor)
        return img_resize(img, shape)

    # Build the pyramid from coarsest to finest
    # We use a list because we want to store images of different shapes
    pyramid = [downsample(i) for i in reversed(range(levels))]
    return pyramid


def compute_flow_level(
    curr: jnp.ndarray,
    grad_iters: int,
    patch_size: int,
    starting_flow: jnp.ndarray,
    pp: jnp.ndarray,
    centers: jnp.ndarray,
    grads: jnp.ndarray,
    hessians_inv: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute dense flow at a single resolution level.

    Args:
        curr: current image of shape (H, W)
        grad_iters: number of inverse compositional updates
        patch_size: size of the patches to extract
        starting_flow: initial flow of shape (num_patches, 2)
        pp: patches of shape (num_patches, patch_size, patch_size)
        centers: centers of the patches in the original image
        grads: gradients of the patches of shape (num_patches, p*p, 2)
        hessians_inv: inverted hessian matrices of shape (num_patches, 2, 2)

    Returns:
        flow: jnp.ndarray of shape (num_patches, 2) with the estimated flow
        error: jnp.ndarray of shape (num_patches, patch_size, patch_size) with the
            error of the final flow
    """

    def solve_patch(
        p_prev: jnp.ndarray,
        center: jnp.ndarray,
        grad: jnp.ndarray,
        hess_inv: jnp.ndarray,
        u0: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Solve the flow for a single patch.

        Args:
            p_prev: previous patch of shape (p, p)
            center: center of the patch in the original image
            grad: gradient of the patch of shape (p*p, 2)
            hess_inv: inverted hessian of the patch of shape (2, 2)
            u0: initial flow of shape (2,)

        Returns:
            flow: estimated flow of shape (2,) (dx, dy)
            error: error of the final flow
        """

        def body_fun(_, u):
            # Sample the patch at the current displacement
            # Sample patch expects the displacement to be in (dy, dx) order
            sampled = sample_patch(curr, center + u[::-1], patch_size=patch_size)

            # Compute the error
            error = (sampled - p_prev).ravel()  # shape (p*p,)

            # Compute the update to minimize the error
            # The update is in (dx, dy) order
            delta_u = hess_inv @ grad.T @ error  # shape (2,)

            return u - delta_u

        # Perform the gradient descent iterations
        flow = lax.fori_loop(0, grad_iters, body_fun, u0)

        # If the flow is larger than the patch size, substitute it with the
        # initial flow
        # TODO check if where can work with scalar, check if cond is faster
        flow = jnp.where(jnp.linalg.norm(flow - u0) > patch_size, u0, flow)

        # Compute the final error
        sampled = sample_patch(curr, center + flow[::-1], patch_size=patch_size)
        error = p_prev - sampled  # shape (p, p)

        return flow, error  # type: ignore

    # Vectorize across all patches
    solve_all = jax.vmap(solve_patch, in_axes=(0, 0, 0, 0, 0))
    disps, errors = solve_all(pp, centers, grads, hessians_inv, starting_flow)

    return disps, errors


def sample_patch(img: jnp.ndarray, disp: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Extract a p x p patch from `img` at subpixel displacement `disp`.

    'disp' is a 2D vector (dy, dx) that indicates the center of the patch in the image.

    Args:
        img: of shape (H, W), the source image.
        disp: (dy, dx) position of the patch center in the image.
        patch_size: the size of the patch to extract.

    Returns:
        patch: sampled patch of shape (p, p).
    """
    # Pad the image to handle out-of-bounds coordinates
    half = patch_size // 2

    # Generate integer grid coordinates for the patch centered at (0, 0)
    coords = jnp.stack(
        jnp.meshgrid(
            jnp.arange(patch_size) - half,
            jnp.arange(patch_size) - half,
            indexing="ij",
        ),
        axis=-1,
    )

    # Shift the coordinates by the displacement
    # and take the padding into account
    coords_f = coords + disp + half

    # Split into y and x coordinates
    yf, xf = coords_f[..., 0], coords_f[..., 1]

    # Apply zero padding to the image
    img_padded = jnp.pad(
        img,
        ((half, half), (half, half)),
        mode="constant",
    )

    # Floor and ceil, clamp
    x0 = jnp.floor(xf).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(yf).astype(jnp.int32)
    y1 = y0 + 1

    # Fractional weights
    wx = xf - x0
    wy = yf - y0

    # Extract the four neighboring pixels
    I00 = gather(img_padded, y0, x0)
    I10 = gather(img_padded, y0, x1)
    I01 = gather(img_padded, y1, x0)
    I11 = gather(img_padded, y1, x1)

    # Bilinear interpolation
    patch = (
        (1 - wx) * (1 - wy) * I00
        + wx * (1 - wy) * I10
        + (1 - wx) * wy * I01
        + wx * wy * I11
    )
    return patch


def gather_2d(array: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """Fetch values from `array` using 2D integer indices.

    Args:
        array: A 2D jnp.ndarray of shape (H, W).
        indices: A (N, 2) jnp.ndarray where each row is [row_idx, col_idx].

    Returns:
        A 1D jnp.ndarray of gathered values.
    """
    return jax.vmap(lambda i: array[i[0], i[1]])(indices)


def extract_patches_grad_hess(
    img: jnp.ndarray, patch_size: int, patch_stride: int, eps: float = 1e-4
):
    """Extract patches, their centers, image gradients and inverted hessian for an image.

    This function extracts patches from the image and returns them in a jnp.ndarray
    in the shape (num_patches, patch_size, patch_size). It also extracts their centers,
    computes each patch's photometric gradient and the inverted hessian matrix.
    This function is used in the DIS algorithm to pre-compute these values for each
    template image.

    Args:
        img: image of size (H, W)
        patch_size: size of the patches to extract
        patch_stride: stride between patches
        eps: small value to avoid division by zero in the hessian inversion

    Returns:
        patches:
            extracted patches of shape (num_patches, patch_size, patch_size)
        centers:
            centers of the patches in the original image
        grad_patches:
            gradients of the patches of shape (num_patches, patch_size * patch_size, 2)
        invH:
            inverted hessian matrices of shape (num_patches, 2, 2)

    """
    H, W = img.shape
    half = patch_size // 2

    # --- compute padding so that (H_p - patch_size) divisible by stride ---
    pad_top = half
    mod_h = (H + pad_top - patch_size) % patch_stride
    rem_h = (-mod_h) % patch_stride

    # Ensure pad_bottom >= half but < half + stride
    if rem_h >= half:
        pad_bottom = rem_h
    else:
        pad_bottom = rem_h + patch_stride

    pad_left = half
    mod_w = (W + pad_left - patch_size) % patch_stride
    rem_w = (-mod_w) % patch_stride
    if rem_w >= half:
        pad_right = rem_w
    else:
        pad_right = rem_w + patch_stride

    # pad image
    img_pad = jnp.pad(
        img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant"
    )
    H_p = H + pad_top + pad_bottom
    W_p = W + pad_left + pad_right

    # Calculate gradients using Sobel operator
    # SAME padding keeps the original image size
    kx, ky = sobel()
    Ix = lax.conv_general_dilated(
        img_pad[None, None, ...], kx[None, None, ...], (1, 1), padding="SAME"
    )[0, 0]
    Iy = lax.conv_general_dilated(
        img_pad[None, None, ...], ky[None, None, ...], (1, 1), padding="SAME"
    )[0, 0]

    # Compute the inverted Hessian matrix for each patch
    invH = inv_hessian(Ix, Iy, patch_size, patch_stride, eps=eps)

    # Extract intensity and gradient patches
    patches = extract_patches(
        img_pad[None, ...], patch_size=patch_size, patch_stride=patch_stride
    ).squeeze()

    grad_patches = extract_patches(
        jnp.stack([Ix, Iy]),
        patch_size=patch_size,
        patch_stride=patch_stride,
    ).reshape(-1, patch_size * patch_size, 2)

    # Generate the Y/X centre grid
    # The number of patches depends on whether there need to be patches
    # centered within the padding
    # --- compute centers in padded coords and re-anchor to original image ---
    ys = jnp.arange(0, H_p - patch_size + 1, patch_stride)
    xs = jnp.arange(0, W_p - patch_size + 1, patch_stride)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    centres_p = jnp.stack([grid_y + half, grid_x + half], axis=-1).reshape(-1, 2)
    centres = centres_p - jnp.array([pad_top, pad_left])

    return (patches, centres, grad_patches, invH)


def extract_patches(
    img: jnp.ndarray, patch_size: int, patch_stride: int
) -> jnp.ndarray:
    """Extract sliding patches from a multi-channel image.

    Args:
        img: image of size (C, H, W)
        patch_size: size of the patches to extract
        patch_stride: stride between patches

    Returns:
        patches:
            extracted patches of shape (num_patches, patch_size, patch_size, C)
    """
    C, _, _ = img.shape

    win = (patch_size, patch_size)
    st = (patch_stride, patch_stride)

    # Extract the patches using lax.conv_general_dilated_patches
    # The input image is reshaped to (N, C, H, W) for the convolution
    # with a pxp filter
    lhs = img.astype(jnp.float32)[None, ...]  # (N,C,H,W)
    patches = lax.conv_general_dilated_patches(
        lhs,
        filter_shape=win,
        window_strides=st,
        padding="VALID",
    )

    # (1, p*p*C, Oh, Ow) -> (1, Oh, Ow, p*p*C) -> (1, Oh, Ow, p, p, C)
    _, N, Oh, Ow = patches.shape
    patches = patches.reshape(1, C, patch_size, patch_size, Oh, Ow)
    patches = patches.transpose(0, 4, 5, 2, 3, 1)
    patches = patches.reshape(-1, patch_size, patch_size, C)

    return patches


def query_flow_at_points(
    sparse_flow: jnp.ndarray,
    centers: jnp.ndarray,
    patch_size: int,
    patch_errors: jnp.ndarray,
    query_coords: jnp.ndarray,  # shape (N, 2)
) -> jnp.ndarray:
    """Computes optical flow only at specific coordinates.

    Args:
        sparse_flow: (num_patches, 2) flow vectors.
        centers: (num_patches, 2) patch centers.
        patch_size: size of the patch.
        patch_errors: (num_patches, patch_size, patch_size) error maps.
        query_coords: (N, 2) coordinates to query.

    Returns:
        queried_flow: (N, 2) estimated flow vectors at query points.
    """
    half_patch = patch_size // 2

    def single_query_flow(coord):
        qy, qx = coord

        def influence_from_patch(i):
            cy, cx = centers[i]
            y0, y1 = cy - half_patch, cy + half_patch + 1
            x0, x1 = cx - half_patch, cx + half_patch + 1

            # Check if query point lies within this patch
            inside = (qx >= x0) & (qx < x1) & (qy >= y0) & (qy < y1)

            def weighted_flow():
                ix = qx - x0
                iy = qy - y0
                weight = 1.0 / jnp.maximum(1.0, jnp.abs(patch_errors[i, iy, ix]))
                return sparse_flow[i] * weight, weight

            return jax.lax.cond(inside, weighted_flow, lambda: (jnp.zeros(2), 0.0))

        flow_contribs, weights = jax.vmap(influence_from_patch)(
            jnp.arange(centers.shape[0])
        )
        total_weight = jnp.sum(weights)
        return jnp.sum(flow_contribs, axis=0) / total_weight

    return jax.vmap(single_query_flow)(query_coords)


def gather(img: jnp.ndarray, y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Gather pixels from the image at the specified coordinates.

    Args:
        img: Image of shape (H, W).
        y: y-coordinates of shape (N, M).
        x: x-coordinates of shape (N, M).

    Returns:
        Gathered pixels of shape (N, M).
    """
    all_batches = jnp.stack([y, x], axis=-1)
    return jax.vmap(lambda batch: jax.vmap(lambda idx: img[tuple(idx)])(batch))(
        all_batches
    )


def patch_centers(N: int, stride: int) -> jnp.ndarray:
    """Compute the centers of patches for a given dimension and stride.

    Args:
        N: Size of the dimension (height or width).
        stride: Stride between patches.

    Returns:
        jnp.ndarray: Array of patch center indices. Shape (num_centers,).
    """
    # Generate all multiples of stride less than N
    centers = jnp.arange(0, N, stride)
    # If the last center is not N-1, append N-1
    if centers.size == 0 or (N - 1) % stride:
        centers = jnp.concatenate([centers, jnp.array([N - 1])])
    return centers


def photometric_error_with_patches(
    prev: jnp.ndarray,
    curr: jnp.ndarray,
    flow: jnp.ndarray,
    patch_size: int = 3,
    patch_stride: int = 1,
    squared: bool = False,
) -> jnp.ndarray:
    """Compute the photometric error for each patch (as a matrix).

    This function computes the photometric error between the previous and current images
    by extracting patches from the previous image, applying the flow sampled from the
    center of the patch, and computing the error between the sampled patch and the
    corresponding patch in the current image and then taking the mean of the errors
    across all pixels in the patch.

    Note:
    WIP, for now it works only for patch_stride = 1.
    TODO: support patch_stride > 1.

    Args:
        prev (jnp.ndarray): Previous image of shape (H, W).
        curr (jnp.ndarray): Current image of shape (H, W).
        flow (jnp.ndarray): Flow field of shape (H, W, 2).
        patch_size (int): Size of the patches to extract.
        patch_stride (int): Stride between patches.
        squared (bool): If True, compute squared error.

    Returns:
        jnp.ndarray: Error matrix of shape (H-2*half, W-2*half)
    """
    # Pad the image before extracting patches
    H, W = prev.shape
    half = patch_size // 2

    # Extract patches from the previous image
    patches = extract_patches(
        prev[None, ...], patch_size=patch_size, patch_stride=patch_stride
    ).squeeze()  # shape (num_patches, patch_size, patch_size)

    # Compute grid of centers
    grid_h = H - 2 * half
    grid_w = W - 2 * half
    centers = patch_grid(grid_h, grid_w, patch_stride, patch_stride) + jnp.array(
        [half, half]
    )

    y = centers[..., 0]
    x = centers[..., 1]
    sampled_flow = gather(flow, y, x)  # shape (out_H, out_W, 2)

    # Compute displaced patch centers in curr
    displaced_centers = centers + sampled_flow
    displaced_centers = displaced_centers.reshape(-1, 2)

    # Sample patches from curr at displaced centers
    sampled_patches = jax.vmap(
        sample_patch,
        in_axes=(None, 0, None),
    )(curr, displaced_centers, patch_size)

    # Compute the mean photometric error in each patch
    errors = patches - sampled_patches  # shape (num_patches, patch_size, patch_size)
    if not squared:
        patch_errors = jnp.linalg.norm(errors, axis=(1, 2))  # shape (num_patches,)
    else:
        patch_errors = jnp.sum(errors**2, axis=(1, 2))  # shape (num_patches,)

    # Compute output shape
    out_H = (H - 2 * half) // patch_stride
    out_W = (W - 2 * half) // patch_stride

    # Reshape error vector into matrix
    error_matrix = patch_errors.reshape(out_H, out_W)

    return error_matrix


def patch_grid(H, W, stride_y, stride_x):
    """Generate a grid of patch centers for an image."""
    centers_y = patch_centers(H, stride_y)
    centers_x = patch_centers(W, stride_x)
    grid_y, grid_x = jnp.meshgrid(centers_y, centers_x, indexing="ij")
    grid = jnp.stack([grid_y, grid_x], axis=-1)  # shape (num_y, num_x, 2)
    return grid
