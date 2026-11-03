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


_DIS_13 = {
    **_DIS_COMMON,
    "patch_size": 13,
    "patch_stride": 2,
    "grad_desc_iters": 8,
}


def _crop(fraction):
    return {
        "name": "crop_special",
        "target_height": 32,
        "target_width": 32,
        "fraction_v": fraction,
        "fraction_h": fraction,
    }


def _suffixes(variants):
    return {
        DISJAXFlowFieldEstimator(**_DIS_13, **kw).get_cache_id_suffix(None)
        for kw in variants
    }


def test_empty_processing_keeps_golden_suffix():
    """Empty or absent pre/post-processing leaves existing cache ids intact."""
    model = DISJAXFlowFieldEstimator(
        **_DIS_13, preprocessing_steps=[], postprocessing_steps=[]
    )
    assert model.get_cache_id_suffix(None) == "_c007d9a94"


def test_preprocessing_changes_cache_id():
    """The same DIS parameters on different crops need different caches."""
    variants = [
        {"preprocessing_steps": [_crop(fraction)]}
        for fraction in (0.0, 0.5, 1.0)
    ]
    suffixes = _suffixes(variants)
    assert len(suffixes) == 3
    assert "_c007d9a94" not in suffixes


def test_postprocessing_changes_cache_id():
    """Postprocessing changes the estimate, so it changes the cache id."""
    step = {"name": "universal_median_test", "r_threshold": 2.0, "radius": 1}
    variants = [
        {},
        {"postprocessing_steps": [step]},
        {"postprocessing_steps": [{**step, "r_threshold": 3.0}]},
    ]
    assert len(_suffixes(variants)) == 3
