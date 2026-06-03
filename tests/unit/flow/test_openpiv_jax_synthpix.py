"""End-to-end agreement of JAX OpenPIV with the reference on synthpix images.

What matters for a re-implementation is that the JAX pipeline reproduces the
original ``openpiv`` displacement field on the *same* realistic images,
across window/overlap/search-area parameters. Particle image pairs are
generated on the fly with synthpix from known flow fields (via the shared
``synthpix_pair_factory`` fixture), then the JAX
``extended_search_area_piv`` is compared window-by-window against
``pyprocess.extended_search_area_piv`` in the configuration the JAX code
reproduces (circular correlation, gaussian sub-pixel, normalized,
vectorized).

Residual note: the agreement is bounded by float32 precision because
openpiv's ``normalize_intensity`` hard-casts to ``float32`` internally. With
the normalization order and sub-pixel ``eps`` matched, peaks and the
sub-pixel fit are otherwise identical, so the disagreement is ~1e-6 px for
standard windows and grows only to a few 1e-3 px for very small (<=16 px)
windows whose correlation peak is razor-sharp and amplifies that round-off.
"""

import numpy as np
import pytest
from openpiv import pyprocess

from flowgym.flow.open_piv.process import extended_search_area_piv
from tests.mocks.piv_data import (
    random_smooth_flow_field,
    uniform_flow_field,
)

# (window_size, search_area_size, overlap) — standard FFT PIV and extended
# search-area configurations a user would actually run.
PARAM_GRID = [
    (32, 32, 16),
    (32, 32, 0),
    (16, 32, 8),
    (24, 32, 12),
    (64, 64, 32),
]

# Per-window agreement tolerance. Dominated by float32 round-off in openpiv's
# normalize_intensity, amplified by sharp correlation peaks on small windows.
MAX_ATOL = 2e-2
# The bulk of windows agree far more tightly than the worst case.
MEDIAN_ATOL = 1e-3


def _reference_field(frame_a, frame_b, window_size, search_area_size, overlap):
    """Reference displacement field in the configuration JAX reproduces."""
    u, v, _ = pyprocess.extended_search_area_piv(
        frame_a.astype(np.float32).copy(),
        frame_b.astype(np.float32).copy(),
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="circular",
        subpixel_method="gaussian",
        sig2noise_method="peak2peak",
        normalized_correlation=True,
        use_vectorized=True,
    )
    return np.asarray(u), np.asarray(v)


def _assert_matches_reference(images1, images2, window_size, sa, overlap):
    """Assert the JAX batch matches the per-sample openpiv reference."""
    flow = np.asarray(
        extended_search_area_piv(
            images1,
            images2,
            window_size=window_size,
            overlap=overlap,
            search_area_size=sa,
        )
    )
    max_err = 0.0
    abs_errs = []
    for n in range(images1.shape[0]):
        u_ref, v_ref = _reference_field(
            images1[n], images2[n], window_size, sa, overlap
        )
        u_jax, v_jax = flow[n, ..., 0], flow[n, ..., 1]

        # Invalid (NaN) windows must agree exactly.
        np.testing.assert_array_equal(~np.isfinite(u_jax), ~np.isfinite(u_ref))
        np.testing.assert_array_equal(~np.isfinite(v_jax), ~np.isfinite(v_ref))

        finite = np.isfinite(u_jax) & np.isfinite(u_ref)
        assert finite.any()
        du = np.abs(u_jax[finite] - u_ref[finite])
        dv = np.abs(v_jax[finite] - v_ref[finite])
        abs_errs.append(np.concatenate([du, dv]))
        max_err = max(max_err, du.max(), dv.max())

    abs_errs = np.concatenate(abs_errs)
    assert max_err < MAX_ATOL, f"max per-window error {max_err:.3e} px"
    assert np.median(abs_errs) < MEDIAN_ATOL, (
        f"median error {np.median(abs_errs):.3e} px"
    )


@pytest.mark.parametrize("window_size, sa, overlap", PARAM_GRID)
def test_uniform_flow_matches_reference(
    synthpix_pair_factory, window_size, sa, overlap
):
    """JAX matches openpiv across parameters on uniformly-shifted images."""
    shifts = [(3.0, -2.0), (1.5, 4.0), (-3.5, 1.0), (0.5, -0.5)]
    flow = uniform_flow_field(shifts, image_shape=(128, 128))
    images1, images2, _ = synthpix_pair_factory(flow, seed=0)
    _assert_matches_reference(images1, images2, window_size, sa, overlap)


@pytest.mark.parametrize("window_size, sa, overlap", PARAM_GRID)
def test_smooth_flow_matches_reference(
    synthpix_pair_factory, window_size, sa, overlap
):
    """JAX matches openpiv across parameters on smooth random flows."""
    import jax

    flow = random_smooth_flow_field(
        jax.random.PRNGKey(7), batch_size=3, image_shape=(128, 128)
    )
    images1, images2, _ = synthpix_pair_factory(flow, seed=1)
    _assert_matches_reference(images1, images2, window_size, sa, overlap)


def test_standard_window_is_tight(synthpix_pair_factory):
    """The default 32px window reproduces openpiv to near float32 precision."""
    shifts = [(2.0, -3.0), (4.0, 1.0)]
    flow = uniform_flow_field(shifts, image_shape=(128, 128))
    images1, images2, _ = synthpix_pair_factory(flow, seed=2)
    flow_jax = np.asarray(
        extended_search_area_piv(
            images1, images2, window_size=32, overlap=16, search_area_size=32
        )
    )
    for n in range(images1.shape[0]):
        u_ref, v_ref = _reference_field(images1[n], images2[n], 32, 32, 16)
        u_jax, v_jax = flow_jax[n, ..., 0], flow_jax[n, ..., 1]
        finite = np.isfinite(u_jax) & np.isfinite(u_ref)
        # Standard windows are not peak-sharpness limited: very tight match.
        np.testing.assert_allclose(u_jax[finite], u_ref[finite], atol=5e-3)
        np.testing.assert_allclose(v_jax[finite], v_ref[finite], atol=5e-3)
