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

Each child rebuilds its state for every image pair with its own
`create_state`, so children may preprocess to different image sizes and the
parent may resize its output. Child flows must share the consensus input grid.
