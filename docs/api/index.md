# API reference

This section is intentionally small and curated for a first pass. It gives
you a browsable local entry point into the package while keeping the docs
site easy to maintain.

## Browse by area

- [Package and factory](package.md):
  top-level imports and estimator creation helpers.
- [Base classes and environment](base.md):
  the base estimator interface and the training environment wrapper.
- [Estimators](estimators.md):
  the flow and density estimator implementations that are most central to
  the current repository.
- [Evaluation and caching](evaluation.md):
  evaluation helpers and cache-backed data reuse.
- [Optimization](optimization.md):
  optimizer setup and learning-rate schedule construction.

## Quick jumps

- [flowgym](package.md#package-surface)
- [flowgym.make](package.md#factory-helpers)
- [flowgym.common.base.estimator](base.md#base-estimator-api)
- [flowgym.environment.fluid_env](base.md#environment)
- [flowgym.flow.dis.dis_jax](estimators.md#dis-jax-estimator)
- [flowgym.flow.raft.raft_jax](estimators.md#raft-jax-estimator)
- [flowgym.common.evaluation](evaluation.md#evaluation-helpers)
- [flowgym.training.optimizer](optimization.md#optimizer-setup)

```{toctree}
:maxdepth: 1
:titlesonly:

package
base
estimators
evaluation
optimization
```
