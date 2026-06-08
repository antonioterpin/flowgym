"""LIMA: lightweight image matching architecture for PIV.

Re-implementation of the lightweight optical-flow CNN of Manickathan,
Mucignat & Lunati (Exp. Fluids 2023) with the padding and search-range
improvements of Mucignat, Zdybał & Lunati (Phys. Fluids 2025).
"""

from flowgym.flow.lima.lima_piv import LimaPivEstimator

__all__ = ["LimaPivEstimator"]
