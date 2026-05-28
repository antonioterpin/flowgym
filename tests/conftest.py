"""Pytest configuration: collection hooks and shared-fixture registration.

Domain fixtures live under :mod:`tests.mocks` and are registered globally
via ``pytest_plugins`` below. This file keeps only the collection hooks and
the truly cross-cutting ``clean_tmp_path`` fixture.
"""

import importlib.util
import shutil

import pytest

# Register the shared mock/fixture modules as plugins so their fixtures are
# available to every test without an explicit import.
pytest_plugins = [
    "mocks.flow_data",
    "mocks.training",
    "mocks.caching",
]


# ──────────────────────────────────────────────────────────────────────────────
# Optional-dependency gating
# ──────────────────────────────────────────────────────────────────────────────
# Some test modules exercise the comparison/baseline methods that live behind
# the `other_methods` extra (OpenCV, OpenPIV, PyTorch). Skip collecting them
# when the underlying dependency is absent so the jax-only suite stays green;
# install `flow-gym-suite[other_methods]` to run them.
_OPTIONAL_DEPENDENCY_MODULES = {
    "test_dis_jax.py": "cv2",
    "test_flow_estimate_dis.py": "cv2",
    "test_openpiv_jax.py": "openpiv",
    "test_train_replay_integration.py": "cv2",
}


def pytest_ignore_collect(collection_path, config):
    """Skip modules whose optional dependency is not installed."""
    dependency = _OPTIONAL_DEPENDENCY_MODULES.get(collection_path.name)
    if dependency is not None and importlib.util.find_spec(dependency) is None:
        return True
    return None


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
