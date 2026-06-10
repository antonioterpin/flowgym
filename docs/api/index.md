# API reference

This page summarizes the public Flow Gym API. For learning-oriented material,
start with [Getting started](../getting-started/index.md) or the
[User guide](../user-guide/index.md).

- [Top-level package and factory](package.md)
  - [Package surface](package.md#package-surface)
  - [Factory helpers](package.md#factory-helpers)

- [Base interfaces and environment](base.md)
  - [Base estimator API](base.md#base-estimator-api)
  - [Environment](base.md#environment)

- [Estimators](estimators.md)
  - [Flow estimator base](estimators/flow-base.md)
  - [DIS JAX estimator](estimators/dis-jax.md)
  - [DIS OpenCV estimator](estimators/dis-opencv.md)
  - [RAFT JAX estimator](estimators/raft-jax.md)
  - [RAFT256 JAX estimator](estimators/raft-jax-256.md)
  - [RAFT PyTorch estimator](estimators/raft-pytorch.md)
  - [RAFT256 PyTorch estimator](estimators/raft-pytorch-256.md)
  - [LIMA estimator](estimators/lima.md)
  - [OpenPIV JAX estimator](estimators/openpiv-jax.md)
  - [OpenPIV estimator](estimators/openpiv.md)
  - [Farneback estimator](estimators/farneback.md)
  - [DeepFlow estimator](estimators/deepflow.md)
  - [Horn-Schunck estimator](estimators/horn-schunck.md)
  - [Consensus estimator](estimators/consensus.md)
  - [ADMM refinement](estimators/admm-refinement.md)
  - [Dummy estimator](estimators/dummy.md)
  - [Simple density estimator](estimators/density-simple.md)
  - [Neural density estimator](estimators/density-nn.md)

- [Evaluation helpers and caching](evaluation.md)
  - [Evaluation helpers](evaluation.md#evaluation-helpers)
  - [Caching](evaluation.md#caching)

- [Training and optimization](optimization.md)
  - [Optimizers](optimization.md#optimizers)
  - [Schedules](optimization.md#schedules)
  - [Losses](optimization.md#losses)
  - [Replay buffer](optimization.md#replay-buffer)
  - [Target transforms](optimization.md#target-transforms)
  - [Exploration policies](optimization.md#exploration-policies)

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:

package
base
estimators
evaluation
optimization
```
