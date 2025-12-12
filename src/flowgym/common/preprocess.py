"""Module for preprocessing PIV images."""

import jax.image as jim
import jax.numpy as jnp
import numpy as np

from flowgym.common.filters import gaussian_smoothing
from flowgym.utils import DEBUG


def stretch_contrast_validate_params():
    """Validate parameters for contrast stretching."""
    pass


def stretch_contrast(image: jnp.ndarray) -> jnp.ndarray:
    """Stretch contrast of an image to enhance visibility.

    Args:
        image: Input image.

    Returns:
        Contrast-stretched image.
    """
    if DEBUG:
        assert isinstance(image, jnp.ndarray), "Input must be a JAX array."
        assert image.ndim in (2, 3), "Input must be 2D or 3D."

    min_val = jnp.min(image, axis=(1, 2), keepdims=True)
    max_val = jnp.max(image, axis=(1, 2), keepdims=True)
    stretched_image = (image - min_val) / (max_val - min_val + 1e-22)
    return jnp.clip(stretched_image, 0, 1)


def intensity_capping_validate_params(n: float):
    """Validate parameters for intensity capping."""
    if not isinstance(n, (int, float)):
        raise TypeError(f"n must be a number, got {type(n)}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")


def intensity_capping(image: jnp.ndarray, n: float = 2.0) -> jnp.ndarray:
    """Intensity capping for cross-correlation improvement.

    See https://link.springer.com/article/10.1007/s00348-006-0233-7.
    See https://openresearchsoftware.metajnl.com/articles/10.5334/jors.bl.

    Args:
        image: Input image.
        n: Number of standard deviations to cap.

    Returns:
        Capped image.
    """
    median_val = jnp.median(image, axis=(1, 2), keepdims=True)
    std_val = jnp.std(image, axis=(1, 2), keepdims=True)
    upper_limit = median_val + n * std_val
    return jnp.minimum(image, upper_limit)


def intensity_clipping_validate_params(n: float):
    """Validate parameters for intensity clipping."""
    if not isinstance(n, (int, float)):
        raise TypeError(f"n must be a number, got {type(n)}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")


def intensity_clipping(image: jnp.ndarray, n: float = 2.0) -> jnp.ndarray:
    """Intensity clipping for cross-correlation improvement.

    Args:
        image (jnp.ndarray): Input image.
        n (float): Number of standard deviations to clip.

    Returns:
        jnp.ndarray: Clipped image.
    """
    median_val = jnp.median(image, axis=(1, 2), keepdims=True)
    std_val = jnp.std(image, axis=(1, 2), keepdims=True)
    upper_limit = median_val + n * std_val
    lower_limit = median_val - n * std_val
    return jnp.clip(image, lower_limit, upper_limit)


def clahe_validate_params(
    clip_limit: float,
    tile_grid_size: tuple[int, int] | None = None,
    nbins: int = 256,
):
    """Validate parameters for CLAHE preprocessing.

    Args:
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of the grid for adaptive histogram equalization.
        nbins: Number of bins for the histogram.
    """
    if not isinstance(clip_limit, (int, float)):
        raise TypeError(f"clip_limit must be a number, got {type(clip_limit)}")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")
    if tile_grid_size is not None:
        if not isinstance(tile_grid_size, tuple):
            raise TypeError(
                f"tile_grid_size must be a tuple, got {type(tile_grid_size)}"
            )
        if len(tile_grid_size) != 2:
            raise ValueError(
                f"tile_grid_size must be a tuple of length 2, got {len(tile_grid_size)}"
            )
        if not all(isinstance(x, int) and x > 0 for x in tile_grid_size):
            raise ValueError(
                f"tile_grid_size must contain positive integers, got {tile_grid_size}"
            )
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError(f"nbins must be a positive integer, got {nbins}")


def clahe(
    images: jnp.ndarray,
    clip_limit: float = 0.01,
    tile_grid_size: tuple[int, int] | None = None,
    nbins: int = 256,
) -> jnp.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) on a batch of 2D images.

    Args:
        images: Input images of shape (B, H, W) or (H, W).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size:
            Size of the grid for adaptive histogram equalization.
            If None, defaults to (8, 8).
        nbins: Number of bins for the histogram.

    Returns:
        Processed images with enhanced contrast.
    """
    from skimage import exposure
    # stretch contrast first
    images = stretch_contrast(images)
    # move to NumPy
    imgs = np.asarray(images)
    # ensure batch dimension
    if imgs.ndim == 2:
        imgs = imgs[None, ...]
    # apply skimage CLAHE per image
    out_list = []
    for img in imgs:
        out = exposure.equalize_adapthist(
            img,
            clip_limit=clip_limit,
            kernel_size=tile_grid_size,
            nbins=nbins,
        )
        out_list.append(out.astype(np.float32))
    out_np = np.stack(out_list, axis=0)
    # back to JAX
    return jnp.asarray(out_np)


def high_pass_filter_validate_params(sigma: float, truncate: float = 4.0):
    """Validate parameters for high-pass filter."""
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"sigma must be a number, got {type(sigma)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not isinstance(truncate, (int, float)):
        raise TypeError(f"truncate must be a number, got {type(truncate)}")
    if truncate <= 0:
        raise ValueError(f"truncate must be positive, got {truncate}")


def high_pass_filter(
    image: jnp.ndarray, sigma: float, truncate: float = 4.0
) -> jnp.ndarray:
    """Apply a high-pass filter to a batch of images.

    Args:
        image: Input array of shape (B, H, W).
        sigma: Standard deviation of the Gaussian kernel in pixels.
        truncate: Truncate parameter for the Gaussian kernel.

    Returns:
        High-pass filtered output of same shape as input.
    """
    # Low-pass via Gaussian smoothing
    blurred = gaussian_smoothing(
        image[..., None], sigma=sigma, truncate=truncate, mode="reflect"
    )[..., 0]

    # High-pass result
    return image - blurred


def background_suppression_validate_params(
    threshold: float,
):
    """Validate parameters for background suppression."""
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a number, got {type(threshold)}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")


def background_suppression(
    image: jnp.ndarray,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """Suppress background noise in the image.

    Args:
        image: Input image.
        threshold: Background suppression factor.

    Returns:
        Image with suppressed background.
    """
    if DEBUG:
        assert isinstance(image, jnp.ndarray), "Input must be a JAX array."
        assert image.ndim in (2, 3), "Input must be 2D or 3D."

    if threshold > 0:
        return jnp.clip(image - threshold, 0, None)
    return image


def crop_special_validate_params(
    target_height: int,
    target_width: int,
    fraction_v: float = 0.5,
    fraction_h: float = 0.5,
):
    """Validate parameters for special cropping."""
    if not isinstance(target_height, int) or target_height <= 0:
        raise ValueError(
            f"target_height must be a positive integer, got {target_height}"
        )
    if not isinstance(target_width, int) or target_width <= 0:
        raise ValueError(f"target_width must be a positive integer, got {target_width}")
    if not (0 <= fraction_v <= 1):
        raise ValueError(f"fraction_v must be in [0, 1], got {fraction_v}")
    if not (0 <= fraction_h <= 1):
        raise ValueError(f"fraction_h must be in [0, 1], got {fraction_h}")


def crop_special(
    image: jnp.ndarray,
    target_height: int,
    target_width: int,
    fraction_v: float = 0.5,
    fraction_h: float = 0.5,
) -> jnp.ndarray:
    """Crop the image to a specific height and width.

    Args:
        image: Input image.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        fraction_v: Fractional position for vertical cropping.
        fraction_h: Fractional position for horizontal cropping.

    Returns:
        jnp.ndarray: Cropped image.
    """
    H, W = image.shape[-2:]
    v_margin = int((H - target_height) * fraction_v)
    h_margin = int((W - target_width) * fraction_h)
    if DEBUG:
        assert isinstance(image, jnp.ndarray), "Input must be a JAX array."
        assert image.ndim in (2, 3), "Input must be 2D or 3D."
        assert H >= target_height and W >= target_width, (
            f"Image size {H}x{W} is smaller than "
            f"target size {target_height}x{target_width}."
        )
        assert v_margin >= 0 and h_margin >= 0, "Margins must be non-negative."
    return image[
        ...,
        v_margin : target_height + v_margin,
        h_margin : target_width + h_margin,
    ]


def resize_image_validate_params(
    target_height: int,
    target_width: int,
):
    """Validate parameters for image resizing."""
    if not isinstance(target_height, int) or target_height <= 0:
        raise ValueError(
            f"target_height must be a positive integer, got {target_height}"
        )
    if not isinstance(target_width, int) or target_width <= 0:
        raise ValueError(f"target_width must be a positive integer, got {target_width}")


def resize_image(
    image: jnp.ndarray,
    target_height: int,
    target_width: int,
) -> jnp.ndarray:
    """Resize the image to a specific height and width.

    Args:
        image: Input image.
        target_height: Desired height after resizing.
        target_width: Desired width after resizing.

    Returns:
        Resized image.
    """
    if DEBUG:
        assert isinstance(image, jnp.ndarray), "Input must be a JAX array."
        assert image.ndim in (2, 3), "Input must be 2D or 3D."
    return jim.resize(
        image, (image.shape[0], target_height, target_width), method="bilinear"
    )


def validate_params(
    preprocessing_step_name: str,
    **kwargs,
):
    """Validate preprocessing step parameters.

    Args:
        preprocessing_step_name: Name of the preprocessing step.
        **kwargs: Parameters for the preprocessing step.

    Returns:
        True if parameters are valid, False otherwise.
    """
    validate_func_name = f"{preprocessing_step_name}_validate_params"
    if validate_func_name in globals():
        validate_func = globals()[validate_func_name]
        validate_func(**kwargs)
    else:
        raise ValueError(f"Unknown preprocessing step {preprocessing_step_name}")


def apply_preprocessing(
    image: jnp.ndarray,
    name: str,
    **kwargs,
) -> jnp.ndarray:
    """Apply a series of preprocessing steps to an image.

    Args:
        image: Input image.
        name: Name of the preprocessing function to apply.
        **kwargs: Additional parameters for the preprocessing function.

    Returns:
        Preprocessed image.
    """
    return globals()[name](image, **kwargs)
