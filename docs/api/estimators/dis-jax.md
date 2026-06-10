# DIS JAX estimator

JAX port of OpenCV's Dense Inverse Search (DIS).

DIS runs coarse-to-fine on an image pyramid. At each level it splits the
image into overlapping patches and, for every patch, searches the
displacement $u$ that minimizes the photometric error

$$\sum_{x \in P} \big(I_1(x) - I_2(x + u)\big)^2,$$

using fast inverse-compositional Gauss–Newton steps (a precomputed inverse
Hessian). The per-patch displacements are then densified into a flow field
and polished with a few variational refinement iterations.

**Reference:** T. Kroeger, R. Timofte, D. Dai, L. Van Gool, *Fast Optical
Flow using Dense Inverse Search*, ECCV 2016
([arXiv:1603.03590](https://arxiv.org/abs/1603.03590)).

```{eval-rst}
.. automodule:: flowgym.flow.dis.dis_jax
   :members:
```
