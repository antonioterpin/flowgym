# Package and factory

This page is the best starting point if you want to understand what the
package exposes at the top level and how estimators are created.

## Package surface

Use this when you want to see the names re-exported at the top level of
`flowgym`.

```{eval-rst}
.. automodule:: flowgym
   :members:
   :undoc-members:
```

## Factory helpers

Use this section when you want to understand how estimators are created,
compiled, saved, or restored. `compile_model(model, estimates)` returns
every entry point of a model, including its processing pipelines; see [the usage guide](../user-guide/estimator-api.md#access-preprocessing-and-postprocessing).

```{eval-rst}
.. automodule:: flowgym.make
   :members:
```
