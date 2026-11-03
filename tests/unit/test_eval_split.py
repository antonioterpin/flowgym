"""Tests for the validation/test block of a dataset config."""

from __future__ import annotations

import pytest

from flowgym.run_setup import _eval_split


def test_num_batches_all_needs_a_non_looping_dataset() -> None:
    """``all`` means every sample once, so it is None and cannot loop."""
    _, n = _eval_split({"dataset": {"loop": False}, "num_batches": "all"}, "v")
    assert n is None
    with pytest.raises(ValueError, match="loop: false"):
        _eval_split({"dataset": {"loop": True}, "num_batches": "all"}, "v")
