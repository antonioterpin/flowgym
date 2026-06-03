"""Synthpix particle-image-pair fixtures for flow-estimator tests.

These helpers are deliberately method-agnostic so every flow
re-implementation (OpenPIV, DIS, DeepFlow, ...) can reuse them to:

- generate realistic particle image pairs that are consistent with a known
  flow field, and
- check a JAX re-implementation against its reference implementation on the
  *same* images, and/or that it recovers the imposed displacement.

The key entry points are the ``synthpix_pair_factory`` fixture and the flow
builders (:func:`uniform_flow_field`, :func:`random_smooth_flow_field`).
``synthpix`` is a dev dependency, so these are always importable in the test
environment; comparisons against an *optional* baseline (e.g. ``openpiv``)
belong in the per-method test module, gated in ``tests/conftest.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthpix.data_generate import generate_images_from_flow
from synthpix.types import ImageGenerationSpecification


def uniform_flow_field(batch_shifts, image_shape):
    """Build a batch of spatially uniform flow fields.

    Args:
        batch_shifts: Sequence of ``(dx, dy)`` displacements, one per sample.
        image_shape: ``(height, width)`` of each flow field.

    Returns:
        Array of shape ``(N, height, width, 2)`` where channel 0 is ``dx``
        (x-displacement) and channel 1 is ``dy`` (y-displacement).
    """
    height, width = image_shape
    flow = np.zeros((len(batch_shifts), height, width, 2), dtype=np.float32)
    for n, (dx, dy) in enumerate(batch_shifts):
        flow[n, ..., 0] = dx
        flow[n, ..., 1] = dy
    return flow


def random_smooth_flow_field(key, batch_size, image_shape, max_disp=2.5):
    """Build a batch of smooth, low-frequency random flow fields.

    A coarse random grid is bicubically upsampled to the image resolution so
    the displacement varies slowly across each interrogation window, which is
    the regime where window-based PIV is well posed.

    Args:
        key: JAX PRNG key.
        batch_size: Number of flow fields.
        image_shape: ``(height, width)`` of each flow field.
        max_disp: Maximum absolute displacement in pixels.

    Returns:
        Array of shape ``(batch_size, height, width, 2)``.
    """
    height, width = image_shape
    coarse = jax.random.uniform(
        key, (batch_size, 4, 4, 2), minval=-1.0, maxval=1.0
    )
    smooth = jax.image.resize(
        coarse, (batch_size, height, width, 2), method="cubic"
    )
    return np.asarray(smooth * max_disp, dtype=np.float32)


def synthpix_pairs_from_flow(
    key,
    flow_field,
    *,
    seeding_density=0.05,
    noise_std=0.0,
    position_bounds=None,
):
    """Generate particle image pairs consistent with a flow field.

    Args:
        key: JAX PRNG key for particle sampling.
        flow_field: Array ``(N, H, W, 2)`` of velocity fields (pixels/dt).
        seeding_density: Particle seeding density (particles per pixel).
        noise_std: Standard deviation of additive Gaussian image noise.
        position_bounds: ``(height, width)`` bounds for particle positions;
            defaults to the flow-field spatial shape.

    Returns:
        Tuple ``(images1, images2)`` of numpy arrays, each ``(N, H, W)``.
    """
    flow_field = np.asarray(flow_field)
    batch_size, height, width = flow_field.shape[:3]
    spec = ImageGenerationSpecification(
        batch_size=batch_size,
        image_shape=(height, width),
        img_offset=(0, 0),
        seeding_density_range=(seeding_density, seeding_density),
        noise_gaussian_std=noise_std,
        p_hide_img1=0.0,
        p_hide_img2=0.0,
    )
    images1, images2, _ = generate_images_from_flow(
        key,
        jnp.asarray(flow_field),
        spec,
        position_bounds=position_bounds or (height, width),
    )
    return np.asarray(images1), np.asarray(images2)


@pytest.fixture
def synthpix_pair_factory():
    """Factory fixture building ``(images1, images2, flow_field)`` triples.

    Returns a callable ``make(flow_field, *, seed=0, **kwargs)`` that renders
    particle image pairs for the given flow field with a deterministic key.
    Extra keyword arguments are forwarded to
    :func:`synthpix_pairs_from_flow` (e.g. ``seeding_density``, ``noise_std``).
    """

    def make(flow_field, *, seed=0, **kwargs):
        key = jax.random.PRNGKey(seed)
        images1, images2 = synthpix_pairs_from_flow(key, flow_field, **kwargs)
        return images1, images2, np.asarray(flow_field)

    return make
