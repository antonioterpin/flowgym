"""Initialization module for DIS flow estimators."""

from flowgym.flow.dis.dis_jax import DISJAXFlowFieldEstimator
from flowgym.utils import optional_import, MissingDependency
dis_mod = optional_import("flowgym.flow.dis.dis")

if dis_mod is not None:
    DISFlowFieldEstimator = dis_mod.DISFlowFieldEstimator
else:
    DISFlowFieldEstimator = MissingDependency("dis", "extra")

__all__ = ["DISFlowFieldEstimator", "DISJAXFlowFieldEstimator"]
