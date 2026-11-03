"""Tests for the per-worker goggles socket used under pytest-xdist."""

import os

import pytest
from mocks.worker_env import isolate_goggles_socket


def test_leaves_environment_alone_outside_xdist():
    environ = {"GOGGLES_SOCKET": "/tmp/goggles-mine.sock"}
    isolate_goggles_socket(environ)
    assert environ == {"GOGGLES_SOCKET": "/tmp/goggles-mine.sock"}


def test_workers_sharing_a_socket_get_distinct_ones():
    sockets = set()
    for worker in ("gw0", "gw1"):
        environ = {
            "PYTEST_XDIST_WORKER": worker,
            "GOGGLES_SOCKET": "/tmp/goggles-run.sock",
        }
        isolate_goggles_socket(environ)
        assert environ["GOGGLES_SOCKET"].startswith("/tmp/goggles-run.sock")
        sockets.add(environ["GOGGLES_SOCKET"])
    assert len(sockets) == 2


def test_workers_without_a_socket_get_distinct_ones():
    sockets = set()
    for worker in ("gw0", "gw1"):
        environ = {"PYTEST_XDIST_WORKER": worker}
        isolate_goggles_socket(environ)
        sockets.add(environ["GOGGLES_SOCKET"])
    assert len(sockets) == 2


def test_this_worker_has_its_own_socket():
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        pytest.skip("only meaningful under pytest-xdist")
    assert os.environ["GOGGLES_SOCKET"].endswith(worker)
