# API reference

This page summarizes the public FlowGym API. For learning-oriented material,
start with [Getting started](../getting-started/index.md) or the
[User guide](../user-guide/index.md).

- [Top-level package and factory](package.md)
  - [Package surface](package.md#package-surface)
  - [Factory helpers](package.md#factory-helpers)

- [Base interfaces and environment](base.md)
  - [Base estimator API](base.md#base-estimator-api)
  - [Environment](base.md#environment)

- [Estimator implementations](estimators.md)
  - [Flow estimator base](estimators.md#flow-estimator-base)
  - [DIS JAX estimator](estimators.md#dis-jax-estimator)
  - [RAFT JAX estimator](estimators.md#raft-jax-estimator)
  - [OpenPIV JAX estimator](estimators.md#openpiv-jax-estimator)
  - [Density estimators](estimators.md#density-estimators)

- [Evaluation helpers and caching](evaluation.md)
  - [Evaluation helpers](evaluation.md#evaluation-helpers)
  - [Caching](evaluation.md#caching)

- [Training and optimization](optimization.md)
  - [Optimizer setup](optimization.md#optimizer-setup)
  - [Schedules](optimization.md#schedules)

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
