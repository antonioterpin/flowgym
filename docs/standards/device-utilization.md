# Device utilization

## GPU management

- Always run `nvidia-smi` to check what GPUs are available.
- Unless specified differently by the user, run everything on the first free GPU.
- Make sure to kill running processes that you've started and don't leave them hanging at the end of your work.

## Goggles port isolation

- Every separate process started by an agent must use a unique `GOGGLES_PORT`.
- Do not reuse the same `GOGGLES_PORT` across concurrent agent-started processes on the same machine.
- Prefer assigning a random free port per process start, for example:
  ```bash
  GOGGLES_PORT="$(uv run python -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1]); s.close()')" uv run pytest
  ```
