# Contributing

This guide explains how to work in the current FlowGym repository without
guessing at historical setup or stale paths.

## Environment setup

FlowGym uses `uv` for dependency management.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync the local development environment:

```bash
uv sync --group dev
```

Add docs tooling when you are working on documentation:

```bash
uv sync --group dev --group docs
```

Optional extras are separate from dependency groups:

```bash
uv sync --group dev --extra cuda12
uv sync --group dev --extra other_methods
```

The current package metadata in `pyproject.toml` requires Python 3.10+.

## Daily commands

Run tests:

```bash
uv run pytest
```

Run the repository CLI:

```bash
uv run python src/main.py --help
```

Run formatting and lint checks:

```bash
uv run pre-commit run --all-files
```

Run the push-stage quality checks configured for the repo:

```bash
uv run pre-commit run --hook-stage push --all-files
```

Build the docs:

```bash
uv run --group docs make -C docs html
```

Run the strict docs build:

```bash
uv run --group docs make -C docs stricthtml
```

## How the repo is organized

- `src/flowgym/`:
  package code and most public APIs
- `src/main.py`:
  top-level CLI entrypoint
- `tests/`:
  unit and integration tests
- `examples/`:
  repository-backed walkthrough scripts
- `docs/`:
  canonical human-facing docs and contributor docs

For a mental model of the codebase, read the
[Architecture guide](architecture.md) before making broad changes.

## Working conventions

- Use topic branches and open a PR for review.
- Keep changes small enough that tests and docs can stay in sync.
- Prefer updating docs alongside behavior changes rather than batching docs
  later.
- When you change a public interface, verify both the guide layer and the
  API layer.

## Documentation maintenance

Keep documentation responsibilities split by purpose:

- `docs/index.md`:
  canonical landing page for humans
- `docs/getting-started/index.md`:
  onboarding index and first-time user path
- `docs/user-guide/`:
  concept-first package guidance beyond the quickstart
- `docs/examples/`:
  curated walkthroughs tied to files in `examples/`
- `docs/api/`:
  curated reference layer for public modules
- `docs/project-docs.md`:
  internal standards, workflows, and agent notes

When you update docs:

1. Edit the canonical page in `docs/`, not a wrapper or duplicate page.
2. Update any example snippet that depends on the changed behavior.
3. If you add a new public API area, list it in the right `docs/api/` page.
4. If you add a new example script worth teaching from, surface it in
   `docs/examples/`.

## Docs review checklist

Before finishing a docs-affecting change, verify:

- user-visible behavior is reflected in docs
- public examples match current signatures and import paths
- new or changed APIs are listed in the right reference page
- internal links render correctly
- `html` and `stricthtml` doc builds pass

This project treats broken examples and stale paths as documentation bugs,
not editorial nits.

## Related docs

- [Getting started](../getting-started/index.md)
- [User guide](../user-guide/index.md)
- [Example workflows](../examples/index.md)
- [API reference](../api/index.md)
- [Project docs](../project-docs.md)
