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
