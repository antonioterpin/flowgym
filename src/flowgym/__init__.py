"""Module for Estimator classes."""

__version__ = "0.1.0"

from flowgym.common.base import Estimator
from flowgym.density.nn import NNDensityEstimator
from flowgym.density.simple import SimpleDensityEstimator
from flowgym.flow.consensus import ConsensusFlowEstimator
from flowgym.flow.dis import DISJAXFlowFieldEstimator
from flowgym.flow.dummy import DummyEstimator
from flowgym.flow.lima.lima_piv import LimaPivEstimator
from flowgym.flow.open_piv import OpenPIVJAXEstimator
from flowgym.flow.postprocess.oracle_threshold import (
    LearnedOracleThresholdEstimator,
)
from flowgym.flow.raft.raft256_jax import RaftJax256Estimator
from flowgym.flow.raft.raft_jax import RaftJaxEstimator
from flowgym.utils import MissingDependency, optional_import

# Explicit alias of the RAFT32-PIV default, symmetric with the 256 variant.
RaftJax32Estimator = RaftJaxEstimator

raft_mod = optional_import("flowgym.flow.raft.raft_piv_pytorch")
if raft_mod is not None:
    RaftTorchEstimator = raft_mod.RaftTorchEstimator
    RaftTorch256Estimator = raft_mod.RaftTorch256Estimator
else:
    RaftTorchEstimator = MissingDependency(
        "raft_piv_pytorch", ["other_methods"]
    )
    RaftTorch256Estimator = MissingDependency(
        "raft_piv_pytorch", ["other_methods"]
    )

# Explicit alias of the RAFT32-PIV default, symmetric with the 256 variant.
RaftTorch32Estimator = RaftTorchEstimator

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
    "raft_jax": RaftJaxEstimator,  # default; RAFT32-PIV (32x32 windows)
    "raft_jax_32": RaftJax32Estimator,  # explicit alias of the default
    "raft_jax_256": RaftJax256Estimator,
    "lima_piv": LimaPivEstimator,
    "raft_torch": RaftTorchEstimator,  # default; RAFT32-PIV (32x32 windows)
    "raft_torch_32": RaftTorch32Estimator,  # explicit alias of the default
    "raft_torch_256": RaftTorch256Estimator,
    "dummy": DummyEstimator,
    "learned_oracle_threshold": LearnedOracleThresholdEstimator,
}
