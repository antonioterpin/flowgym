# 🌊 Flow Gym : A Research Toolkit for Fluid Flow Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/antonioterpin/flowgym?style=social)](https://github.com/antonioterpin/flowgym/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/flow-gym.svg)](https://pypi.org/project/flow-gym)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Flow Gym is a **toolkit for research and deployment in flow quantification**, inspired by **OpenAI Gym** and **Stable-Baselines3**.
It leverages **SynthPix** for synthetic image generation and provides a **unified interface** for testing, training, and deploying learning-based algorithms that estimate fluid flow from sequences of tracer-particle images.
The framework also includes a growing collection of **integrated algorithms** and **stable JAX re-implementations** of state-of-the-art methods.

![The Flow Gym environment](https://raw.githubusercontent.com/antonioterpin/flowgym/main/docs/flowgym.jpg)

## ✨ Features

- **Stable and reproducible environments**
  Flow Gym provides **FluidEnv**, a unified environment for both supervised and reinforcement learning. It abstracts away infrastructure details and ensures consistent training and evaluation across experiments.

- **Unified estimator framework**
  The **Estimator** API makes it easy to integrate, compare, and benchmark different flow estimation algorithms. It also supports extensions beyond velocity estimation — for example, pressure or tracer density — enabling ablation studies and cross-domain applications in fluid dynamics and control.

- **Ready for real-world deployment**
  Flow Gym includes efficient **JAX implementations** of state-of-the-art flow estimation methods, optimized for real-time performance and suitable for **closed-loop control** or other **fluid-sensing** applications.

## Getting Started 🚀

```bash
# Basic installation
uv add flow-gym-suite # or pip install flow-gym-suite

# With CUDA12 support
uv add "flow-gym-suite[cuda12]"
```

If you also want to use Flow Gym for training an evaluation, please clone the repository and install the development dependencies as outlined in the [CONTRIBUTING.md](./CONTRIBUTING.md) page.
There you can also find help if you have any issues installing Flow Gym.

### Basic usage / Deployment  🧭

```python
from flowgym.make import make_estimator
# Define the config (or load from YAML)
model_config = {
    "estimator": "dis_jax",
    "estimate_type": "flow",
    "config": {"jit": True, ..., },
}

# Create the estimator and associated functions
trained_state, create_state_fn, compute_estimate_fn, model = make_estimator(
    model_config=model_config,
    image_shape=image_shape
)

# Compute an estimate for an image pair
est_state = create_state_fn(prev)
new_est_state, metrics = compute_estimate_fn(curr, est_state)
```

### Training a model 🧠

```python
from flowgym.environment.fluid_env import FluidEnv

# Create the environment
env, env_state = FluidEnv.make(env_config)

for episode in range(num_episodes):
    obs, env_state, done = env.reset(env_state)
    est_state = create_state_fn(obs)
    train_step_fn = model.create_train_step()

    while not done.any():
        # Estimator forward pass
        est_state, metrics = compute_estimate_fn(obs, est_state, train_state)

        # Extract the action (the last estimate)
        action = estimation_state["estimates"][:, -1]
        # Environment step
        obs, env_state, reward, done = env.step(env_state, action)

        # Optimization step
        loss, train_state, _ = train_step_fn(
            est_state, train_state, reward
        )
```

### Implementing a custom estimator 🧩

```python
from flowgym.base import Estimator
class MyEstimator(Estimator):
    def init(self, rng, observation_space, action_space):
        # Initialize model parameters and state
        pass

    def _estimate(self, images, state, trainable_state, extras):
        # Define the forward pass of the model
        return new_state, extras, metrics

    def create_trainable_state(self, dummy_input, key):
        # Create the trainable state if needed (e.g., network structure, optimizer, etc.)
        pass

    def create_train_step(self):
        def train_step(self, estimation_state, trainable_state, rewards):
            # Define the training step (loss computation and optimization)
            return loss, new_trainable_state, metrics
```

## Custom Preprocessing  🧼

```python
# flowgym/common/preprocess.py
def custom_preprocess_validate_params(params: Dict[str, Any]) -> None:
    # Validate custom pre-processing parameters
    pass

def custom_preprocess(
    images: jnp.ndarray,
    params: Dict[str, Any]
) -> jnp.ndarray:
    # Implement custom pre-processing logic
    return processed_images

__all__ = [
    "custom_preprocess_validate_params",
    "custom_preprocess",
]
```

## Custom Postprocessing  🎨

```python
# flowgym/flow/postprocess/__init__.py
def custom_flow_postprocess_validate_params(params: Dict[str, Any]) -> None:
    # Validate custom post-processing parameters
    pass

def custom_flow_postprocess(
    estimates: jnp.ndarray,
    params: Dict[str, Any]
) -> jnp.ndarray:
    # Implement custom flow post-processing logic
    return processed_estimates

__all__ = [
    "custom_flow_postprocess_validate_params",
    "custom_flow_postprocess",
]
```

## Examples 📚

For more examples, please check our [training](https://raw.githubusercontent.com/antonioterpin/flowgym/main/src/flowgym/train.py) and [evaluation](https://raw.githubusercontent.com/antonioterpin/flowgym/main/src/flowgym/eval.py) scripts.

## Contributing 🤗

Contributions are more than welcome! 🙏 Please check out our [how to contribute page](https://raw.githubusercontent.com/antonioterpin/flowgym/main/CONTRIBUTING.md), and feel free to open an issue for problems and feature requests⚠️.

If you have developed a new method integrating with Flow Gym, please do open a PR! New methods should come with reproducible experiments showcasing the features. We collect these experiments in the folder `experiments/your-method`.

## Citation 📈

If you use this code in your research, please cite our paper:

```bibtex
   @article{banelli2025flowgym,
      title={Flow Gym},
      author={Banelli, Francesco and Terpin, Antonio and Bonomi, Alan and D'Andrea, Raffaello},
      year={2025}
   }
```

## License 📝

This project is licensed under the MIT License - see the [LICENSE](https://raw.githubusercontent.com/antonioterpin/flowgym/main/LICENSE) file for details.
