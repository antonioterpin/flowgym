# DeepFlow estimator

OpenCV-backed DeepFlow.

DeepFlow minimizes a variational optical-flow energy — brightness- and
gradient-constancy data terms plus a smoothness term — augmented with a
*deep matching* correspondence term that anchors large displacements. The
energy is optimized coarse-to-fine on an image pyramid.

**Reference:** P. Weinzaepfel, J. Revaud, Z. Harchaoui, C. Schmid, *DeepFlow:
Large Displacement Optical Flow with Deep Matching*, ICCV 2013
([paper](https://openaccess.thecvf.com/content_iccv_2013/html/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.html)).

```{eval-rst}
.. automodule:: flowgym.flow.deepflow
   :members:
```
