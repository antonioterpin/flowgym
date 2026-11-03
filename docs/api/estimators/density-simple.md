# Simple density estimator

A lightweight, non-learned density estimator (`SimpleDensityEstimator`).

It counts the pixels whose intensity exceeds a threshold $\tau$ and divides
by the total number of pixels, returning a single scalar density per image —
the fraction of occupied pixels,

$$\rho = \frac{1}{HW} \sum_{x} \mathbb{1}\!\left[I(x) > \tau\right],$$

a particles-per-pixel proxy. No training is required.

**Reference:** [Flow Gym source](https://github.com/antonioterpin/flowgym).

```{eval-rst}
.. automodule:: flowgym.density.simple
   :members:
```
