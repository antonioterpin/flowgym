"""Tests for the flow-grid diagnostics."""

import numpy as np
import pytest

from flowgym.flow.diagnostics import (
    flow_pair_diagnostics,
    particle_support_mask,
)


def test_support_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    image1 = rng.uniform(0, 24, (96, 128)).astype(np.float32)
    image2 = rng.uniform(0, 24, (96, 128)).astype(np.float32)
    expected = cv2.GaussianBlur((image1 + image2) / 2, (0, 0), 5) > 12
    expected[:24] = expected[-24:] = False
    expected[:, :24] = expected[:, -24:] = False
    expected = cv2.resize(expected.astype(np.float32), (16, 16))
    actual = particle_support_mask(image1, image2, (16, 16), border=24)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("shape", [(64, 80), (16, 16)])
def test_correlation_of_known_translation(shape):
    rng = np.random.default_rng(8)
    image1 = rng.uniform(0, 255, (64, 80)).astype(np.float32)
    image2 = np.roll(image1, (2, 3), axis=(0, 1))
    flow = np.broadcast_to(np.float32([3, 2]), (*shape, 2))
    right = flow_pair_diagnostics(image1, image2, flow)
    assert right["valid"].any() and not right["valid"].all()
    np.testing.assert_allclose(
        right["correlation"][right["valid"]], 1, atol=1e-4
    )
    wrong = flow_pair_diagnostics(image1, image2, 0 * flow)
    assert wrong["correlation"][wrong["valid"]].mean() < 0.2
    blank = np.zeros_like(image1)
    assert not flow_pair_diagnostics(blank, blank, flow)["valid"].any()
