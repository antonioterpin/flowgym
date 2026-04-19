# Estimators

This page documents the estimator implementations currently available in
FlowGym.

Some estimators are JAX-native, while others wrap methods from optional
external libraries. The optional integrations are still part of the public
estimator surface, but they may require extra dependencies at runtime.

## Flow estimator base

Start here if you want the shared behavior across flow estimators.

```{eval-rst}
.. automodule:: flowgym.flow.base
   :members:
```

## DIS JAX estimator

The main JAX-backed DIS implementation.

```{eval-rst}
.. automodule:: flowgym.flow.dis.dis_jax
   :members:
```

## DIS OpenCV estimator

This is the OpenCV-backed DIS variant exposed through the same estimator
interface.

```{eval-rst}
.. automodule:: flowgym.flow.dis.dis
   :members:
```

## RAFT JAX estimator

The JAX RAFT implementation used for learned flow estimation.

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft_jax
   :members:
```

## RAFT PyTorch estimator

This is the PyTorch-backed RAFT integration.

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft_piv_pytorch
   :members:
```

## OpenPIV JAX estimator

The JAX-based OpenPIV estimator variant.

```{eval-rst}
.. automodule:: flowgym.flow.open_piv.openpiv_jax
   :members:
```

## OpenPIV estimator

This is the OpenPIV-backed integration.

```{eval-rst}
.. automodule:: flowgym.flow.open_piv.openpiv
   :members:
```

## Farneback estimator

OpenCV Farneback optical-flow estimator exposed through the FlowGym
interface.

```{eval-rst}
.. automodule:: flowgym.flow.farneback
   :members:
```

## DeepFlow estimator

OpenCV DeepFlow integration exposed through the FlowGym interface.

```{eval-rst}
.. automodule:: flowgym.flow.deepflow
   :members:
```

## Horn-Schunck estimator

Horn-Schunck optical-flow integration exposed through the FlowGym
interface.

```{eval-rst}
.. automodule:: flowgym.flow.hornschunck
   :members:
```

## Consensus estimator

Consensus-based flow estimation built on top of multiple underlying
estimators.

```{eval-rst}
.. automodule:: flowgym.flow.consensus.consensus
   :members:
```

## Dummy estimator

Minimal estimator used in tests and workflow examples.

```{eval-rst}
.. automodule:: flowgym.flow.dummy
   :members:
```

## Density estimators

These are the density-focused estimator implementations.

```{eval-rst}
.. automodule:: flowgym.density.simple
   :members:
```

```{eval-rst}
.. automodule:: flowgym.density.nn
   :members:
```
