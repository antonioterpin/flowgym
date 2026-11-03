# OpenPIV JAX estimator

JAX port of OpenPIV cross-correlation.

The frames are split into small interrogation windows. For each window pair
the displacement is the location of the peak of their cross-correlation,

$$R(s) = \sum_{x} I_1(x)\, I_2(x + s),$$

computed efficiently with the FFT; a Gaussian fit around the peak yields
sub-pixel accuracy. Windows are processed on a grid to produce the flow
field.

**Reference:** [OpenPIV](https://www.openpiv.net/) — open-source particle
image velocimetry
([algorithm basics](https://openpiv.readthedocs.io/en/latest/src/piv_basics.html)).

```{eval-rst}
.. automodule:: flowgym.flow.open_piv.openpiv_jax
   :members:
```

## Method selectors and array types

The processing functions and `OpenPIVJAXEstimator` share literal selectors
from `flowgym.flow.open_piv.types`: `SubpixelMethod` (`gaussian`, `parabolic`,
`centroid`), `CorrelationMethod` (`circular`, `linear`), and
`SignalToNoiseMethod` (`peak2peak`, `peak2mean`). These signatures let
basedpyright reject misspelled methods and preserve the return-type overloads
of `extended_search_area_piv`.

For strings read from configuration, use `parse_subpixel_method`,
`parse_correlation_method`, or `parse_sig2noise_method`. The processing
functions retain these thin runtime checks for untyped callers. Selectors
must still be static arguments when passed through `jax.jit`.

`sig2noise_val` declares a `jaxtyping.Bool[jax.Array, "..."]` return and
produces `jnp.bool_` elements in eager and compiled execution. `True` marks
an outlier, the inverse of FlowGym postprocessing's valid-mask convention.
The annotation describes element dtype; basedpyright does not prove array
element dtypes, so this guarantee is also covered by runtime regression tests.
