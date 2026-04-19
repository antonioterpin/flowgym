# Training and evaluation workflows

FlowGym supports more than one style of use.

When imported as a package, you can build an estimator directly and run it from
Python. When cloning the repository, you can also launch evaluation, training,
and comparison workflows from configuration files.

## Package use

Package use is the smallest loop:

1. build an estimator with `flowgym.make.make_estimator`
2. create a runtime state from an input frame
3. compute estimates frame by frame

This is the best place to start when you are learning the API.

## Evaluation workflows

Evaluation workflows are for running an estimator on a dataset or benchmark
through the repository orchestration layer.

These workflows usually involve:

- an estimator configuration
- a dataset configuration
- evaluation helpers under `flowgym.common.evaluation`
- the main repository entrypoint, `src/main.py`.

## Training workflows

Training workflows add optimization, schedules, replay/caching support, and
dataset/sampler orchestration around the estimator itself.

These workflows are the reason FlowGym has both package-level modules under
`flowgym.training.*` and repository scripts such as `src/train.py` and
`src/train_supervised.py`.

## Where examples fit

The pages under [Example workflows](../examples/index.md) show complete,
small repository-backed runs.

Use them when you want to see how estimator configuration, dataset
configuration, and `src/main.py` fit together in practice.

## Related pages

- [Estimator API overview](estimator-api.md)
- [Configuration and data flow](configuration-and-data.md)
- [Example workflows](../examples/index.md)
