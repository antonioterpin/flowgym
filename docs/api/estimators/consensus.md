# Consensus estimator

Consensus combination of multiple flow estimators. For how the refinement
works, see [ADMM refinement](admm-refinement.md).

**Reference:** A. Bonomi, F. Banelli, A. Terpin, *Particle Image Velocimetry
Refinement via Consensus ADMM*, 2025
([arXiv:2512.11695](https://arxiv.org/abs/2512.11695)).

```{eval-rst}
.. automodule:: flowgym.flow.consensus.consensus
   :members:
```

## Child state and resizing

Each child initializes its own state from the preceding parent input image,
so its preprocessing and extra history fields are applied consistently to
both frames. Child image grids may differ; their final flows must share the
consensus input grid. Parent postprocessing may then resize that grid.

When temporal propagation is enabled, the previous consensus flow is
interpolated onto the input grid before initializing each child's estimate
history. Otherwise children start with zero flow. Child state is rebuilt for
each pair; individual child estimates are not carried between calls.
