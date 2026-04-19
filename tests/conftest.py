"""Conftest.py for tests."""

import shutil
from datetime import datetime
from unittest.mock import MagicMock

import h5py
import jax.numpy as jnp
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Collection modifier
# ──────────────────────────────────────────────────────────────────────────────
def pytest_collection_modifyitems(config, items):
    """Skip tests unless explicitly selected with -m run_explicitly."""
    if config.getoption("-m") and "run_explicitly" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(
        reason="Skipped unless explicitly selected with -m run_explicitly"
    )
    for item in items:
        if "run_explicitly" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def clean_tmp_path(tmp_path):
    """tmp_path that gets removed after the test finishes."""
    yield tmp_path
    # Best-effort cleanup; ignore if something already removed it.
    shutil.rmtree(tmp_path, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# .mat helpers
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mat_test_dims():
    """Return default dimensions for .mat test files."""
    return {"height": 64, "width": 64}


@pytest.fixture
def mock_mat_files(tmp_path, mat_test_dims, request):
    """Create multiple temporary .mat files with random data."""
    param = getattr(request, "param", 2)

    if isinstance(param, dict):
        num_files = param.get("num_files", 2)
        dims = param.get("dims", mat_test_dims)
        h, w = dims["height"], dims["width"]
    else:
        num_files = param
        dims = mat_test_dims
        h, w = mat_test_dims["height"], mat_test_dims["width"]

    paths = []
    for t in range(1, num_files + 1):
        mat_path = tmp_path / f"flow_{t:04d}.mat"
        with h5py.File(mat_path, "w", libver="latest", userblock_size=512) as f:
            f.create_dataset(
                "I0",
                data=np.random.randint(0, 255, size=(h, w), dtype=np.uint8),
            )
            f.create_dataset(
                "I1",
                data=np.random.randint(0, 255, size=(h, w), dtype=np.uint8),
            )
            f.create_dataset(
                "V", data=np.random.rand(h, w, 2).astype(np.float32)
            )

        # write fake MATLAB header
        header = (
            (
                f"MATLAB 7.3 MAT-file, Platform: Python-h5py, "
                f"Created on {datetime.now():%c}"
            )
            .encode("ascii")
            .ljust(116, b" ")
        )
        header += b" " * (512 - 116)
        with open(mat_path, "r+b") as fp:
            fp.write(header)

        paths.append(mat_path)

    yield [str(p) for p in paths], dims


@pytest.fixture
def npy_flow_files(tmp_path):
    """Create .npy flow field files for SyntheticImageSampler testing."""
    flow_file = tmp_path / "flow_001.npy"
    np.save(flow_file, np.random.rand(128, 128, 2).astype(np.float32))
    return [str(flow_file)]


@pytest.fixture
def mock_dependencies():
    """Fixture providing mock model, env, observations for training tests."""
    model = MagicMock()
    model.create_train_step.return_value = MagicMock(
        return_value=(0.1, MagicMock(), {})
    )
    model.process_metrics.side_effect = lambda x: x

    env = MagicMock()
    # env.reset returns (obs, state, done)
    # obs is (prev, curr)
    obs = (jnp.zeros((2, 4, 4)), jnp.zeros((2, 4, 4)))
    env_state = (MagicMock(), jnp.zeros((2, 4, 4, 2)))
    done = jnp.array([False, False])
    env.reset.return_value = (obs, env_state, done)

    # env.step returns (obs, state, reward, done)
    reward = jnp.array([0.0, 0.0])
    env.step.return_value = (obs, env_state, reward, jnp.array([True, True]))

    return model, env, obs, env_state


# ──────────────────────────────────────────────────────────────────────────────
# Training integration test fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_trainable_state():
    """Create a minimal NNEstimatorTrainableState for testing."""
    import optax
    from flax.core import FrozenDict

    from flowgym.common.base.trainable_state import NNEstimatorTrainableState

    def apply_fn(params, x):
        return params["w"] * x

    params = FrozenDict({"w": jnp.array(1.0, jnp.float32)})
    tx = optax.sgd(0.01)
    return NNEstimatorTrainableState.create(
        apply_fn=apply_fn, params=params, tx=tx
    )


@pytest.fixture
def mock_sampler():
    """Create a mock sampler that yields synthetic batches."""
    B, H, W = 2, 32, 32
    batch = MagicMock()
    batch.images1 = jnp.zeros((B, H, W))
    batch.images2 = jnp.zeros((B, H, W))
    batch.flow_fields = jnp.zeros((B, H, W, 2))
    batch.params = None
    batch.mask = None

    sampler = MagicMock()
    # Return 10 batches then stop
    sampler.__iter__.return_value = iter([batch] * 10)
    sampler.shutdown = MagicMock()
    sampler.reset = MagicMock()
    # Set grain_iterator to None so save_model skips sampler serialization
    sampler.grain_iterator = None
    return sampler


@pytest.fixture
def mock_env():
    """Create a mock environment for RL training."""
    env = MagicMock()
    B, H, W = 2, 32, 32
    obs = (jnp.zeros((B, H, W)), jnp.zeros((B, H, W)))
    env_state = (MagicMock(), jnp.zeros((B, H, W, 2)))

    env.reset.return_value = (obs, env_state, jnp.array([False, False]))
    env.step.return_value = (
        obs,
        env_state,
        jnp.array([1.0, 1.0]),
        jnp.array([True, True]),
    )
    return env, obs, env_state


# ──────────────────────────────────────────────────────────────────────────────
# Caching test fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Create a temporary cache directory with optional pre-populated data.

    Yields the path to the cache directory. Use this fixture for testing
    cache read/write operations without hitting the real filesystem.
    """
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir
    # Cleanup is handled by tmp_path


@pytest.fixture
def populated_cache_dir(mock_cache_dir):
    """Create a pre-populated cache with sample EPE data.

    Returns (cache_dir, cache_id, keys, epe_values) tuple for verification.
    """

    from flowgym.training.caching import CacheManager

    cache_id = "test_estimator_cache"
    spec = {
        "epe": (np.dtype("float32"), ()),
        "relative_epe": (np.dtype("float32"), ()),
    }

    cm = CacheManager(
        root_dir=str(mock_cache_dir),
        cache_id=cache_id,
        spec=spec,
        warm_start="none",
    )

    # Write sample data
    keys = np.array([1001, 1002, 1003], dtype=np.uint64)
    payload = {
        "epe": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "relative_epe": np.array([0.01, 0.02, 0.03], dtype=np.float32),
    }
    cm.write(keys, payload)
    cm.flush()
    cm.close()

    return mock_cache_dir, cache_id, keys, payload


@pytest.fixture
def mock_synthpix_batch():
    """Create a mock SynthpixBatch for cache testing."""
    from synthpix import SynthpixBatch

    B, H, W = 4, 64, 64  # Small dims for fast tests
    return SynthpixBatch(
        images1=jnp.zeros((B, H, W)),
        images2=jnp.zeros((B, H, W)),
        flow_fields=jnp.zeros((B, H, W, 2)),
        keys=jnp.array([100, 101, 102, 103], dtype=jnp.uint64),
    )
