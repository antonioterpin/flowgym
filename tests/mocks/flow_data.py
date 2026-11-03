"""Synthetic flow-data fixtures (.mat / .npy files) for the test suite."""

from datetime import datetime

import h5py
import numpy as np
import pytest


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
