# Estimators

This page documents the estimator implementations currently available in
FlowGym.

Some estimators are JAX-native, while others wrap methods from optional
external libraries. The optional integrations are still part of the public
estimator surface, but they may require extra dependencies at runtime.

## Flow estimator base

`FlowFieldEstimator` — the abstract base class every flow estimator
subclasses. Defines the shared call signature, postprocessing hooks,
and supported output types.

```{eval-rst}
.. automodule:: flowgym.flow.base
   :members:
```

## DIS JAX estimator

JAX-backed Dense Inverse Search (DIS).

```{eval-rst}
.. automodule:: flowgym.flow.dis.dis_jax
   :members:
```

## DIS OpenCV estimator

OpenCV-backed Dense Inverse Search (DIS).

```{eval-rst}
.. automodule:: flowgym.flow.dis.dis
   :members:
```

## RAFT JAX estimator

JAX-backed RAFT (learning-based flow).

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft_jax
   :members:
```

## RAFT PyTorch estimator

PyTorch-backed RAFT (learning-based flow).

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft_piv_pytorch
   :members:
```

## OpenPIV JAX estimator

JAX-backed OpenPIV cross-correlation.

```{eval-rst}
.. automodule:: flowgym.flow.open_piv.openpiv_jax
   :members:
```

## OpenPIV estimator

OpenPIV-backed cross-correlation.

```{eval-rst}
.. automodule:: flowgym.flow.open_piv.openpiv
   :members:
```

## Farneback estimator

OpenCV-backed Farneback optical flow.

```{eval-rst}
.. automodule:: flowgym.flow.farneback
   :members:
```

## DeepFlow estimator

OpenCV-backed DeepFlow.

```{eval-rst}
.. automodule:: flowgym.flow.deepflow
   :members:
```

## Horn-Schunck estimator

`pyoptflow`-backed Horn–Schunck optical flow.

```{eval-rst}
.. automodule:: flowgym.flow.hornschunck
   :members:
```

## Consensus estimator

Consensus combination of multiple flow estimators.

```{eval-rst}
.. automodule:: flowgym.flow.consensus.consensus
   :members:
```

## Dummy estimator

Minimal no-op estimator used in tests and the supervised-training example.

```{eval-rst}
.. automodule:: flowgym.flow.dummy
   :members:
```

## Density estimators

Density-focused estimators (`SimpleDensityEstimator` and
`NNDensityEstimator`).

```{eval-rst}
.. automodule:: flowgym.density.simple
   :members:
```

```{eval-rst}
.. automodule:: flowgym.density.nn
   :members:
```
