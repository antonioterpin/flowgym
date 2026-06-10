# LIMA estimator

JAX/Flax re-implementation of LIMA, a lightweight CNN for PIV.

LIMA ("lightweight image matching architecture") is a lean PWC-Net / IRR-PWC
style network: it builds a feature pyramid, warps features across scales and
forms a local cost volume, then applies weight-shared *iterative residual
refinement* to predict the flow coarse-to-fine. It is trained on synthetic
particle images with a multi-level Jacobian-penalised L1 loss, and at ~0.93M
parameters is far smaller than general-purpose flow networks.

**Reference:** L. Manickathan, C. Mucignat, I. Lunati, *A lightweight neural
network designed for fluid velocimetry*, Experiments in Fluids 64, 161 (2023)
([doi:10.1007/s00348-023-03695-8](https://doi.org/10.1007/s00348-023-03695-8)).

```{eval-rst}
.. automodule:: flowgym.flow.lima.lima_piv
   :members:
```
