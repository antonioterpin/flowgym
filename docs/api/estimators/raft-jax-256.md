# RAFT256 JAX estimator

JAX port of the RAFT256-PIV variant of RAFT (learning-based optical flow for
particle image velocimetry).

RAFT256-PIV is the 1/8-resolution variant of [RAFT32 JAX](raft-jax.md): the
feature and context encoders downsample each 256×256 interrogation window to
32×32 feature maps, the recurrent (GRU) refinement runs at that resolution, and
a learned *convex upsampling* head reconstructs the full-resolution flow at
every iteration. It shares the parameter tree of the RAFT32-PIV variant, so the
same PyTorch → Flax checkpoint converter applies.

**References:** C. Lagemann, K. Lagemann, S. Mukherjee, W. Schröder, *Deep
recurrent optical flow learning for particle image velocimetry data*, Nature
Machine Intelligence 3 (2021) 641–651
([doi:10.1038/s42256-021-00369-0](https://doi.org/10.1038/s42256-021-00369-0));
built on Z. Teed, J. Deng, *RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow*, ECCV 2020
([arXiv:2003.12039](https://arxiv.org/abs/2003.12039)). Reference
implementation: [Code Ocean capsule
7226151](https://codeocean.com/capsule/7226151/tree/v1).

```{eval-rst}
.. automodule:: flowgym.flow.raft.raft256_jax
   :members:
```

```{eval-rst}
.. automodule:: flowgym.nn.raft256_model
   :members:
```
