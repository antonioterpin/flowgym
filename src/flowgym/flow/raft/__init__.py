"""RAFT-PIV estimators.

``raft`` is the shared scaffolding (base estimator classes, NN blocks and the
PyTorch -> Flax converter) for the RAFT-PIV family. The unqualified
``Raft*Estimator`` classes are the **RAFT32-PIV** default (32x32 interrogation
windows); :class:`RaftJax256Estimator` / :class:`RaftTorch256Estimator` are the
larger-window **RAFT256-PIV** variant. ``RaftJax32Estimator`` /
``RaftTorch32Estimator`` are explicit aliases of the default for callers that
want to name the variant unambiguously.
"""

from flowgym.utils import MissingDependency, optional_import

from .raft256_jax import RaftJax256Estimator
from .raft_jax import RaftJaxEstimator

# Explicit alias of the RAFT32-PIV default, symmetric with the 256 variant.
RaftJax32Estimator = RaftJaxEstimator

raft_torch = optional_import("flowgym.flow.raft.raft_piv_pytorch")
if raft_torch is not None:
    RaftTorchEstimator = raft_torch.RaftTorchEstimator
    RaftTorch256Estimator = raft_torch.RaftTorch256Estimator
else:
    RaftTorchEstimator = MissingDependency(
        "raft_piv_pytorch", ["other_methods"]
    )
    RaftTorch256Estimator = MissingDependency(
        "raft_piv_pytorch", ["other_methods"]
    )

# Explicit alias of the RAFT32-PIV default, symmetric with the 256 variant.
RaftTorch32Estimator = RaftTorchEstimator

__all__ = [
    "RaftJax32Estimator",
    "RaftJax256Estimator",
    "RaftJaxEstimator",
    "RaftTorch32Estimator",
    "RaftTorch256Estimator",
    "RaftTorchEstimator",
]
