# Overview: Why Flow Gym?

Flow Gym exists to make flow-field quantification from tracer-particle images
easier to develop, compare, reproduce, and deploy.

Particle Image Velocimetry (PIV) and related optical-flow methods are widely
used in experimental fluid mechanics. Over time, the field has accumulated a
mix of classical methods, learning-based methods, custom training code,
dataset-specific evaluation scripts, and one-off deployment pipelines.

Implementations often live in different libraries, expose incompatible
interfaces, and make different choices about pre-processing,
post-processing, and evaluation. As a result, fair comparison and repeatable
benchmarking become harder than they need to be.

Flow Gym is meant to reduce that friction by giving these workflows a shared
software shape.

## What problem does Flow Gym solve?

Flow Gym aims at simplifying:

- comparing classical and learning-based estimators within the same
  pipeline
- reusing evaluation and training logic across methods
- rerunning experiments from configuration rather than ad-hoc glue code
- benchmarking methods more fairly by sharing surrounding workflow code
- carrying the same method from offline evaluation to practical
  deployment

This is the same kind of benefit that shared interfaces and benchmark-driven
software ecosystems have brought to neighboring fields such as computer
vision and reinforcement learning: less glue code, clearer comparisons, and
better reproducibility.

## What does Flow Gym provide?

- A shared estimator interface:
  classical and learning-based methods expose a consistent shape for state
  creation and estimate computation.
- JAX-first implementations:
  core estimators and training utilities are organized around JAX and Flax
  for accelerator-friendly execution.
- Interoperable wrappers:
  Flow Gym can still integrate representative external methods from libraries
  such as OpenCV, PyTorch, and OpenPIV.
- Shared workflows:
  `src/main.py` and the related scripts handle training, evaluation,
  benchmarking, and comparison from config files.
- Reusable data processing:
  pre-processing and post-processing steps can be shared across methods
  instead of being reimplemented in each experiment.
- Cache-backed execution:
  repeated dataset passes can reuse expensive derived quantities when that
  makes experiments cheaper or easier to reproduce.

The same interface is also used beyond flow estimation itself. The repo
includes related estimators such as tracer-particle density estimation.

## What does the API look like?

The part of Flow Gym most users touch first is the estimator API.

At a high level, you:

1. configure and build an estimator
2. create an estimator state from an input frame
3. compute an estimate on the next frame

The public factory for this is `flowgym.make.make_estimator`, which returns
the trained state together with helper callables for state creation and
estimate computation.

That interface is designed to support both:

- consecutive or recurrent workflows, where a method carries short-term state
  across frames
- independent workflows, where each estimate is computed without relying on
  sequence memory

This lets Flow Gym use the same mental model for a broad range of methods.

## What are the goals?

- Keep the public API small enough to learn, but broad enough to support
  multiple estimator families.
- Make benchmarking and training easier to repeat from configuration.
- Support both package-level use and repository-backed experimentation.
- Improve reproducibility without forcing all methods into one
  implementation style.
- Lower the barrier between research prototypes and real experimental use.
- Give users and contributors a shared mental model instead of a collection
  of one-off scripts.

## Next step

Continue to [Installation](installation.md) if you want to
set up the package locally, or jump to the
[Quick overview](quick-overview.md) if you want to see the API right away.
For a deeper explanation of the estimator interface and common workflows,
continue to the [User guide](../user-guide/index.md).
