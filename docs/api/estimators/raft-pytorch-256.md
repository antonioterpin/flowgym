# RAFT256 PyTorch estimator

PyTorch-backed RAFT256-PIV (learning-based flow for particle image
velocimetry), used as the reference for the [RAFT256 JAX](raft-jax-256.md) port.

RAFT256-PIV is the 1/8-resolution variant of [RAFT32 PyTorch](raft-pytorch.md):
the encoder downsamples each 256×256 interrogation window to 32×32 feature maps
and a learned convex upsampling head reconstructs the full-resolution flow. The
PyTorch modules are ported verbatim (up to formatting) from the reference
implementation and reuse the RAFT32-PIV building blocks where they are
identical.

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
.. autoclass:: flowgym.flow.raft.raft_piv_pytorch.RaftTorch256Estimator
   :members:
```
