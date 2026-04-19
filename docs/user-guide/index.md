# User guide

The user guide is where FlowGym's main ideas are explained in more detail.
Use it when you already know how to install the package and want a clearer
mental model of the estimator API, common workflows, and configuration
structure.

- [Estimator API overview](estimator-api.md)
  - [The main pieces](estimator-api.md#the-main-pieces)
  - [Why the API is split this way](estimator-api.md#why-the-api-is-split-this-way)
  - [The usual flow](estimator-api.md#the-usual-flow)
  - [What lives in the runtime state](estimator-api.md#what-lives-in-the-runtime-state)

- [Training and evaluation workflows](training-and-evaluation.md)
  - [Package use](training-and-evaluation.md#package-use)
  - [Evaluation workflows](training-and-evaluation.md#evaluation-workflows)
  - [Training workflows](training-and-evaluation.md#training-workflows)
  - [Where examples fit](training-and-evaluation.md#where-examples-fit)

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
../examples/first-estimate
../examples/flow-eval
../examples/supervised-training
../examples/density-eval
../examples/caching
```
