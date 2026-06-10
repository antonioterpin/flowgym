# Horn-Schunck estimator

`pyoptflow`-backed Horn–Schunck optical flow.

Horn–Schunck is a global variational method. Assuming brightness constancy,
$I_x u + I_y v + I_t = 0$, and a smoothly varying flow $(u, v)$, it minimizes

$$\int \big(I_x u + I_y v + I_t\big)^2
  + \alpha^2 \big(\lVert \nabla u \rVert^2 + \lVert \nabla v \rVert^2\big)\, dx,$$

where $\alpha$ trades data fidelity against smoothness. The minimizer is
found by iterating the resulting linear updates, optionally coarse-to-fine.

**Reference:** B. K. P. Horn, B. G. Schunck, *Determining Optical Flow*,
Artificial Intelligence 17 (1981)
([doi:10.1016/0004-3702(81)90024-2](https://doi.org/10.1016/0004-3702(81)90024-2)).

```{eval-rst}
.. automodule:: flowgym.flow.hornschunck
   :members:
```
