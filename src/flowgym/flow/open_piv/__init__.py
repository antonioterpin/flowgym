"""OpenPIV flow integration."""

from flowgym.flow.open_piv.openpiv_jax import OpenPIVJAXEstimator
from flowgym.utils import MissingDependency, optional_import

openpiv_mod = optional_import("flowgym.flow.open_piv.openpiv")
if openpiv_mod is not None:
    from flowgym.flow.open_piv.openpiv import OpenPIVEstimator
else:
    OpenPIVEstimator = MissingDependency("openpiv", "other_methods")


__all__ = ["OpenPIVEstimator", "OpenPIVJAXEstimator"]
