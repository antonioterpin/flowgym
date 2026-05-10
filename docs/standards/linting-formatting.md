# Linting & formatting policy

## Configuration

- **All linting and formatting rules live in `pyproject.toml`.**
  - Do not override rule selection via CLI flags in pre-commit or scripts.
  - Pre-commit controls *when* tools run, not *what* rules apply.

## Ruff as single source of truth

- **Ruff is the single source of truth** for:
  - unused variables/imports
  - import ordering
  - modern typing syntax
  - docstring style
  - low-noise bug patterns

## Allowed exceptions

- Allowed uppercase local variable names for tensor dimensions:
  - `H`, `W`, `B`, `N`, `T`

## Import organization

- **Imports must be at the top of the file** by default.
  - Local imports (inside functions/methods) are allowed **only** to avoid circular dependencies or for specific performance/lazy-loading reasons.
  - If a local import is used, it **must** be documented with a comment above it explaining why it's not at the top-level.
