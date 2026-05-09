# Flow Gym agent instructions

This file is the canonical agent-facing instruction set for Flow Gym.
Both human contributors and AI assistants follow the same guide; the
full version lives at `docs/contribute/contributing.md`.

## Essential rules

- Use `uv run` for all commands; do not assume global installs.
- Test-first: write a failing test, then implement the change.
- Run all quality gates before declaring work complete:

  ```bash
  uv run pre-commit run --all-files
  uv run pytest
  uv run pre-commit run --hook-stage push --all-files
  ```

- Commit messages follow the `.gitmessage` template
  (Conventional Commits).
- PRs target `dev`, not `main`. Never push to remote — pushes are
  central.
- Type hints belong in function signatures, not in docstrings; use
  Google-style docstrings with descriptions only.
- Prefer JAX for ML/numerical code. Make randomness explicit.
- GPU usage: run `nvidia-smi` first and use the first free GPU unless
  told otherwise.
- Allowed uppercase variable names for tensor dimensions: `H`, `W`,
  `B`, `N`, `T`.

## Where to look

- Full contributor guide (setup, conventions, testing, PR rules):
  `docs/contribute/contributing.md`
- Architecture and module map: `docs/contribute/architecture.md`
- Public API and user-facing docs: `docs/index.md`
