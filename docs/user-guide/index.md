# User guide

The user guide is where FlowGym's main ideas are explained in more detail.
Use it when you already know how to install the package and want a clearer
mental model of the estimator API, common workflows, and configuration
structure.

- [Estimator API overview](estimator-api.md)
  - [What an `Estimator` is](estimator-api.md#what-an-estimator-is)
  - [The call signature](estimator-api.md#the-call-signature)
  - [Runtime state and trainable state](estimator-api.md#runtime-state-and-trainable-state)
  - [Using `make_estimator(...)`](estimator-api.md#using-make_estimator)

- [Training and evaluation workflows](training-and-evaluation.md)
  - [Package use](training-and-evaluation.md#package-use)
  - [Evaluation workflows](training-and-evaluation.md#evaluation-workflows)
  - [Training workflows](training-and-evaluation.md#training-workflows)
  - [Example workflows](training-and-evaluation.md#example-workflows)

- [Configuration and data flow](configuration-and-data.md)
  - [The two main kinds of config](configuration-and-data.md#the-two-main-kinds-of-config)
  - [How they come together](configuration-and-data.md#how-they-come-together)
  - [Where configs live](configuration-and-data.md#where-configs-live)

- [Example workflows](../examples/index.md)
  - [How to use this section](../examples/index.md#how-to-use-this-section)
  - [Available example pages](../examples/index.md#available-example-pages)
  - [Repository scripts worth knowing](../examples/index.md#repository-scripts-worth-knowing)
  - [First estimate](../examples/first-estimate.md)
  - [Flow evaluation](../examples/flow-eval.md)
  - [Supervised training](../examples/supervised-training.md)
  - [Density evaluation](../examples/density-eval.md)
  - [Caching examples](../examples/caching.md)

```{toctree}
:hidden:
:maxdepth: 2

estimator-api
training-and-evaluation
configuration-and-data
../examples/index
```
