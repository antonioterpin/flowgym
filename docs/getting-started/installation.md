# Installation

This page covers the common installation paths for Flow Gym.

## Install the package

To install the package using [uv](https://docs.astral.sh/uv/):

```bash
uv add flow-gym-suite
```

With `pip` instead:

```bash
pip install flow-gym-suite
```

## Version requirements

The current package metadata requires Python 3.11+.

## GPU / CUDA

Flow Gym runs on CPU out of the box. For GPU acceleration, install the
matching CUDA extra:

```bash
uv add "flow-gym-suite[cuda12]"   # CUDA 12
uv add "flow-gym-suite[cuda13]"   # CUDA 13 (requires Python 3.11+)
```

The `cuda13` extra pulls `jax[cuda13]`, which only exists in jax >=0.10 and
therefore requires Python 3.11+.

```{warning}
**Avoid installing the `other_methods` extra alongside `cuda13` in the same
environment.** `other_methods` pulls in PyTorch, which bundles its own
`nvidia-cudnn-cu12` (e.g. 9.10.2). That older cuDNN can shadow the
`nvidia-cudnn-cu13` (9.22) that `jax[cuda13]` was built against, in which case
JAX may fail on the GPU with:

    Loaded runtime CuDNN library: 9.10.2 but source was compiled with: 9.12.0
    RET_CHECK failure (...gpu_compiler.cc) dnn_support != nullptr

If you hit this, keep the PyTorch-based `other_methods` baselines and the JAX
CUDA 13 GPU stack in **separate environments**. (CPU runs are unaffected.)
```

```{warning}
The published `flow-gym-suite` wheel cannot currently run the minimal
API example in [Quick overview](quick-overview.md) on its own. The
public API at `flowgym.make.make_estimator` unconditionally imports
`synthpix` (and `orbax`), which is only available through the `dev`
group (git-pinned). Until those imports become lazy or land in
`[project] dependencies`, follow the [Working on the
repository](#working-on-the-repository) flow below for a runnable
setup.
```

## Working on the repository

If you want to contribute to Flow Gym itself, use the
[Contributing guide](../contribute/contributing.md) for development environment
setup, docs builds, tests, and local workflow.

## Next step

Move to the [Quick overview](quick-overview.md) for the
first public API example.
