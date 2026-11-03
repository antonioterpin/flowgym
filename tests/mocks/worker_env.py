"""Per-process environment isolation for pytest-xdist workers."""

import os
import tempfile
from collections.abc import MutableMapping


def isolate_goggles_socket(environ: MutableMapping[str, str]) -> None:
    """Give a pytest-xdist worker its own goggles socket.

    Goggles routes every process that shares a socket through one logging
    host, so workers sharing one would mix their logs and lose logging when
    the first worker exits. Outside xdist ``environ`` is left untouched.

    Args:
        environ: Environment to update in place, usually ``os.environ``.
    """
    worker = environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return
    # Workers are children of the controller, so its pid scopes the default
    # base to this test run.
    base = environ.get("GOGGLES_SOCKET") or os.path.join(
        tempfile.gettempdir(), f"goggles-pytest-{os.getppid()}"
    )
    environ["GOGGLES_SOCKET"] = f"{base}-{worker}"
