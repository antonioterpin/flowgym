# Training and optimization

This page collects the training-side modules used to build optimization
pipelines, losses, replay buffers, target transforms, and exploration
policies.

## Optimizers

Use this when you want to understand how optimizer configs become Optax
transformations.

```{eval-rst}
.. automodule:: flowgym.training.optimizer
   :members:
```

## Schedules

Learning-rate schedule construction lives here.

```{eval-rst}
.. automodule:: flowgym.training.schedules
   :members:
```

## Losses

Loss builders and registries live here.

```{eval-rst}
.. automodule:: flowgym.training.losses
   :members:
```

## Replay buffer

Replay-buffer utilities for storing and sampling experiences live here.

```{eval-rst}
.. automodule:: flowgym.training.replay
   :members:
```

## Target transforms

Target transformation builders and registries live here.

```{eval-rst}
.. automodule:: flowgym.training.target_transforms
   :members:
```

## Exploration policies

Exploration policies for action selection live here.

```{eval-rst}
.. automodule:: flowgym.training.exploration
   :members:
```
