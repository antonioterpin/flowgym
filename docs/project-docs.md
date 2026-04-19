# Project docs

This section is for contributors and agent operators working inside the
repository. It gathers the internal reference material that sits behind the
main contributor guide.

If you are looking for setup or the day-to-day development workflow, start
with the [Contributing guide](guides/contributing.md).

## Standards

Use standards as the repository's rules of engagement.

- [API design](standards/api-design.md)
- [Change scope](standards/change-scope.md)
- [Code clarity](standards/code-clarity.md)
- [Code organization](standards/code-organization.md)
- [Code quality](standards/code-quality.md)
- [Device utilization](standards/device-utilization.md)
- [Environment tooling](standards/environment-tooling.md)
- [Exploration and validation](standards/exploration-validation.md)
- [Linting and formatting](standards/linting-formatting.md)
- [ML and numerical code](standards/ml-numerical.md)
- [Testing](standards/testing.md)
- [Typing and docstrings](standards/typing-docstrings.md)
- [Version control](standards/version-control.md)

```{toctree}
:hidden:
:maxdepth: 1

standards/api-design
standards/change-scope
standards/code-clarity
standards/code-organization
standards/code-quality
standards/device-utilization
standards/environment-tooling
standards/exploration-validation
standards/linting-formatting
standards/ml-numerical
standards/testing
standards/typing-docstrings
standards/version-control
```

## Workflows

Use workflows when you already know the kind of task you are doing.

- [API validation](workflows/api-validation.md)
- [Bugfix](workflows/bugfix.md)
- [Documentation](workflows/docs.md)
- [Feature](workflows/feature.md)
- [Orientation](workflows/orientation.md)
- [Refactor](workflows/refactor.md)

```{toctree}
:hidden:
:maxdepth: 1

workflows/api-validation
workflows/bugfix
workflows/docs
workflows/feature
workflows/orientation
workflows/refactor
```

## Agent-facing references

These pages are maintained for contributor tooling and agent-oriented
workflows. They are not part of the main human docs path.

The maintained agent role pages live under `docs/agents/`.

- [Implementer](agents/implementer.md)
- [Reviewer](agents/reviewer.md)

```{toctree}
:hidden:
:maxdepth: 1

agents/implementer
agents/reviewer
```

## Maintenance notes

- Human-facing docs start at [docs/index.md](index.md).
- The contributor workflow starts at
  [docs/guides/contributing.md](guides/contributing.md).
- User-facing concepts live under [docs/user-guide/index.md](user-guide/index.md).
- Example workflows live in [docs/examples/](examples/index.md) and should
  stay aligned with files in `examples/`.
- Compatibility wrappers in `.claude/`, `.github/`, `.agent/`, and the
  root `agents/` directory should link back here rather than duplicating
  full instructions.
