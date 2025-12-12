"""Initialization module for OpenPIV flow estimators."""
from flowgym.utils import optional_import, MissingDependency

openpiv_mod = optional_import("flowgym.flow.open_piv.openpiv")
if openpiv_mod is not None:
    from flowgym.flow.open_piv.openpiv import OpenPIVEstimator
else:
    OpenPIVEstimator = MissingDependency("openpiv", "other_methods")

from flowgym.flow.open_piv.openpiv_jax import OpenPIVJAXEstimator

__all__ = ["OpenPIVEstimator", "OpenPIVJAXEstimator"]
