# User guide

The user guide is where Flow Gym's main ideas are explained in more detail.
Use it when you already know how to install the package and want a clearer
mental model of the estimator API, configuration structure, and common
workflows.

Pages are ordered the way most readers will want them: the API contract
first, then the configuration that drives it, then the workflows that
consume both, with end-to-end example walkthroughs at the end.

- [Estimator API overview](estimator-api.md)
  - [What an `Estimator` is](estimator-api.md#what-an-estimator-is)
  - [The call signature](estimator-api.md#the-call-signature)
  - [Runtime state and trainable state](estimator-api.md#runtime-state-and-trainable-state)
  - [Using `make_estimator(...)`](estimator-api.md#using-make_estimator)
  - [Hooks around estimation](estimator-api.md#hooks-around-estimation)

- [Configuration and data flow](configuration-and-data.md)
  - [The two main kinds of config](configuration-and-data.md#the-two-main-kinds-of-config)
  - [Preprocessing and postprocessing](configuration-and-data.md#preprocessing-and-postprocessing)
  - [How they come together](configuration-and-data.md#how-they-come-together)
  - [Where configs live](configuration-and-data.md#where-configs-live)

- [Training and evaluation workflows](training-and-evaluation.md)
  - [Repository workflows](training-and-evaluation.md#repository-workflows)
  - [Evaluation workflows](training-and-evaluation.md#evaluation-workflows)
  - [Training workflows](training-and-evaluation.md#training-workflows)
  - [What's in `flowgym.training`](training-and-evaluation.md#whats-in-flowgymtraining)

- [Example workflows](../examples/index.md)

```{toctree}
:hidden:
:maxdepth: 2

estimator-api
configuration-and-data
training-and-evaluation
../examples/index
```
