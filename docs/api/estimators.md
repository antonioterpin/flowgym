# Estimators

This page documents the estimator implementations currently available in
Flow Gym.

Some estimators are JAX-native, while others wrap methods from optional
external libraries. The optional integrations are still part of the public
estimator surface, but they may require extra dependencies at runtime.

```{toctree}
:maxdepth: 1

estimators/flow-base
estimators/dis-jax
estimators/dis-opencv
estimators/raft-jax
estimators/raft-jax-256
estimators/raft-pytorch
estimators/raft-pytorch-256
estimators/lima
estimators/openpiv-jax
estimators/openpiv
estimators/farneback
estimators/deepflow
estimators/horn-schunck
estimators/consensus
estimators/admm-refinement
estimators/dummy
estimators/density-simple
estimators/density-nn
```
