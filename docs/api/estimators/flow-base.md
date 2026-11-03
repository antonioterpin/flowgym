# Flow estimator base

`FlowFieldEstimator` — the abstract base class every flow estimator
subclasses. Defines the shared call signature, postprocessing hooks,
and supported output types.

**Reference:** [Flow Gym source](https://github.com/antonioterpin/flowgym) —
the estimator interface every method implements.

```{eval-rst}
.. automodule:: flowgym.flow.base
   :members:
```

Cache enrichment compiles once per estimator, so discarded estimators are
garbage-collected without clearing global JAX caches.
