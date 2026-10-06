"""Golden cache-id suffixes for committed paper candidate configs.

Each EPE cache directory is named ``<cache_id_base><suffix>``, where the
suffix hashes the estimator's ``get_config``. The configs below name existing
caches and selected libraries; if a suffix changes, those no longer match and
the caches must be re-collected.
"""

import pytest

from flowgym.flow.dis.dis_jax import DISJAXFlowFieldEstimator
from flowgym.flow.open_piv.openpiv_jax import OpenPIVJAXEstimator

_DIS_COMMON = {
    "preset": 1,
    "start_level": 0,
    "levels": 3,
    "level_steps": 1,
    "var_refine_iters": 0,
    "use_mean_normalization": False,
    "use_spatial_propagation": False,
    "use_temporal_propagation": False,
    "output_full_res": True,
}


@pytest.mark.parametrize(
    "estimator_cls, kwargs, suffix",
    [
        (
            DISJAXFlowFieldEstimator,
            {
                **_DIS_COMMON,
                "patch_size": 13,
                "patch_stride": 2,
                "grad_desc_iters": 8,
            },
            "_c007d9a94",
        ),
        (
            OpenPIVJAXEstimator,
            {
                "window_size": 16,
                "search_size": 16,
                "overlap": 8,
                "subpixel_method": "gaussian",
                "correlation_method": "circular",
                "n_passes": 3,
            },
            "_cf8badbaa",
        ),
    ],
)
def test_cache_id_suffix_golden(estimator_cls, kwargs, suffix):
    """Suffixes of committed candidate configs stay stable."""
    assert estimator_cls(**kwargs).get_cache_id_suffix(None) == suffix
