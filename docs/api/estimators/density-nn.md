# Neural density estimator

A learned density estimator (`NNDensityEstimator`).

A small convolutional network (optionally with residual connections) is
trained to regress the density map directly from the image, learning the
mapping from pixel intensities to density rather than relying on a fixed
threshold.

**Reference:** [Flow Gym source](https://github.com/antonioterpin/flowgym).

```{eval-rst}
.. automodule:: flowgym.density.nn
   :members:
```
