# Base classes and environment

This page covers the reusable interfaces that shape how estimators and
training flows fit together.

## Base estimator API

This is the main inheritance point for custom estimators and the best
place to understand the expected estimator interface.

```{eval-rst}
.. automodule:: flowgym.common.base.estimator
   :members:
```

## Environment

This wraps synthetic data generation into a training-friendly interface.

```{eval-rst}
.. automodule:: flowgym.environment.fluid_env
   :members:
```
