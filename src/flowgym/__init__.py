"""Module for Estimator classes."""

__version__ = "0.1.0"

from flowgym.common.base import Estimator
from flowgym.density.simple import SimpleDensityEstimator
from flowgym.density.nn import NNDensityEstimator
from flowgym.flow.open_piv import OpenPIVJAXEstimator
from flowgym.flow.dis import DISJAXFlowFieldEstimator
from flowgym.flow.consensus import ConsensusFlowEstimator
from flowgym.flow.raft.raft_jax import RaftJaxEstimator
from flowgym.utils import optional_import, MissingDependency

raft_mod = optional_import("flowgym.flow.raft.raft_piv_pytorch")
if raft_mod is not None:
    RaftTorchEstimator = raft_mod.RaftTorchEstimator
else:
    RaftTorchEstimator = MissingDependency("raft_piv_pytorch", ["other_methods"])

deepflow_mod = optional_import("flowgym.flow.deepflow")
if deepflow_mod is not None:
    DeepFlowEstimator = deepflow_mod.DeepFlowEstimator
else:
    DeepFlowEstimator = MissingDependency("deepflow", ["other_methods"])
    
openpiv_mod = optional_import("flowgym.flow.open_piv.openpiv")
if openpiv_mod is not None:
    OpenPIVEstimator = openpiv_mod.OpenPIVEstimator
else:
    OpenPIVEstimator = MissingDependency("openpiv", ["other_methods"])

hornschunck_mod = optional_import("flowgym.flow.hornschunck")
if hornschunck_mod is not None:
    HornSchunckEstimator = hornschunck_mod.HornSchunckEstimator
else:
    HornSchunckEstimator = MissingDependency("hornschunck", ["other_methods"])

farneback_mod = optional_import("flowgym.flow.farneback")
if farneback_mod is not None:
    FarnebackEstimator = farneback_mod.FarnebackEstimator
else:
    FarnebackEstimator = MissingDependency("farneback", ["other_methods"])

dis_mod = optional_import("flowgym.flow.dis.dis")
if dis_mod is not None:
    DISFlowFieldEstimator = dis_mod.DISFlowFieldEstimator
else:
    DISFlowFieldEstimator = MissingDependency("dis", ["other_methods"])

ALL_ESTIMATORS: dict[str, type[Estimator] | MissingDependency] = {
    "simple": SimpleDensityEstimator,
    "nn_density": NNDensityEstimator,
    "farneback": FarnebackEstimator,
    "deepflow": DeepFlowEstimator,
    "openpiv": OpenPIVEstimator,
    "dis": DISFlowFieldEstimator,
    "openpiv_jax": OpenPIVJAXEstimator,
    "dis_jax": DISJAXFlowFieldEstimator,
    "horn_schunck": HornSchunckEstimator,
    "consensus": ConsensusFlowEstimator,
    "raft_jax": RaftJaxEstimator,
    "raft_torch": RaftTorchEstimator,
}
