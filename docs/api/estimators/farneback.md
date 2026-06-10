# Farneback estimator

OpenCV-backed Farneback optical flow.

Farnebäck approximates the neighborhood of each pixel in both frames by a
quadratic polynomial, $f(x) \approx x^\top A x + b^\top x + c$. The
displacement is recovered from how these polynomial coefficients change
between the two frames, solved over local windows and refined coarse-to-fine
on an image pyramid.

**Reference:** G. Farnebäck, *Two-Frame Motion Estimation Based on Polynomial
Expansion*, SCIA 2003
([doi:10.1007/3-540-45103-X_50](https://doi.org/10.1007/3-540-45103-X_50)).

```{eval-rst}
.. automodule:: flowgym.flow.farneback
   :members:
```
