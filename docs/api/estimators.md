# Estimators

This page groups the main estimator implementations that are most useful
for browsing the repository today. It is intentionally curated rather than
trying to document every internal helper in one place.

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

## RAFT JAX estimator

The JAX RAFT implementation used for learned flow estimation.

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft_jax
   :members:
```

## OpenPIV JAX estimator

The JAX-based OpenPIV estimator variant.

```{eval-rst}
.. automodule:: flowgym.flow.open_piv.openpiv_jax
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
