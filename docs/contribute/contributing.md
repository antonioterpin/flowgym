# Contributor's guide

Thanks for considering a contribution to FlowGym. This page covers the
practical path from a clean checkout to a merged pull request: how to
set up the environment, how the codebase is organized, what
conventions apply to new code, how to test and document changes, and
what a PR needs to be acceptable.

## Where to start

Useful first steps before writing code:

- Read the [Architecture guide](architecture.md) for the codebase's
  mental model and where things live.
- Skim [`src/main.py`](https://github.com/antonioterpin/flowgym/blob/dev/src/main.py)
  if you plan to touch evaluation or training, and the
  [example scripts](../examples/index.md) if you plan to touch the
  package surface.
- Open an issue (or comment on an existing one) before starting work
  on anything non-trivial. Small fixes can go straight to a PR; larger
  changes benefit from an early discussion of the approach.

## Setting up the development environment

FlowGym uses [uv](https://docs.astral.sh/uv/) for dependency and
environment management. Python 3.10 or newer is required.

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync the development environment:

```bash
uv sync --group dev
```

Add the docs extra when you intend to build documentation:

```bash
uv sync --group dev --extra docs
```

### Optional extras

Several estimator families depend on optional libraries that are not
installed by default. Pull them in with `--extra <name>`:

| Extra           | Pulls in                                                              | Required for                                                       |
|-----------------|-----------------------------------------------------------------------|--------------------------------------------------------------------|
| `cuda12`        | `jax[cuda12]`                                                         | running JAX on CUDA 12 GPUs                                        |
| `other_methods` | `torch`, `torchvision`, `opencv-python-headless`, `openpiv`, `pyoptflow` | RAFT-PyTorch, OpenCV-backed estimators (DIS, Farneback, DeepFlow), OpenPIV, Horn–Schunck |
| `wandb`         | `robo-goggles[wandb]`                                                 | logging training runs to Weights & Biases                          |

Combine groups and extras as needed:

```bash
uv sync --group dev --extra docs --extra other_methods
```

### Git and pre-commit setup

Configure git to use the project's commit-message template:

```bash
git config commit.template .gitmessage
```

Install pre-commit hooks. The repository configures three stages
(`pre-commit`, `pre-push`, `commit-msg`); installing once enables all
of them:

```bash
uv run pre-commit install
```

This wires Ruff (lint + format), pydoclint, basedpyright, gitlint
(conventional-commit message format), and a few hygiene hooks
(end-of-file fixer, trailing whitespace, YAML validation) into your
local git workflow.

## Daily commands

### Run the test suite

Full suite:

```bash
uv run pytest
```

Single file or directory:

```bash
uv run pytest tests/training/test_partial_checkpointing.py
```

Filter by pattern:

```bash
uv run pytest -k checkpoint
```

`pytest` runs with coverage by default (configured in
`pyproject.toml`); the HTML coverage report ends up in `htmlcov/`.

### Lint, format, and type-check

The default pre-commit run covers Ruff (lint + format), pydoclint, and
basedpyright on changed files when you commit. To run them across the
whole tree:

```bash
uv run pre-commit run --all-files
```

The `pre-push` stage adds the same checks (intentional — they catch
anything you bypassed locally) and the type-check gate:

```bash
uv run pre-commit run --hook-stage push --all-files
```

A change is not finished until both invocations pass.

### Build the docs

A standard build:

```bash
uv run --extra docs make -C docs html
```

A strict build that treats warnings as errors (this is what the docs
review checklist requires):

```bash
uv run --extra docs sphinx-build -b html docs docs/build/html -W --keep-going
```

A live-reload preview that rebuilds on save:

```bash
uv run --extra docs --with sphinx-autobuild \
    sphinx-autobuild docs docs/build/live --watch docs --ignore "docs/build/*"
```

The preview serves on `http://127.0.0.1:8000` by default.

### Run the repository CLI

```bash
uv run python src/main.py --help
```

For complete walkthroughs see [Training and evaluation
workflows](../user-guide/training-and-evaluation.md) and the
[examples](../examples/index.md).

## Coding conventions

### Code style and clarity

- Prefer clear, explicit code over clever or compact code. Match
  patterns already in the codebase before inventing new ones.
- Keep changes focused. Do not mix refactors, feature changes, and
  formatting in the same commit.
- Comments should explain **why**, not what or how. The code itself
  should be readable.
- Allowed uppercase local variable names for tensor dimensions: `H`,
  `W`, `B`, `N`, `T`.

### Typing and docstrings

- Type information lives in **function signatures**, not in
  docstrings.
- Use Google-style docstrings with descriptions only; no type hints in
  `Args:` or `Returns:` sections.
- All public functions and classes must be documented. All files,
  including tests, must have a file-level docstring.

```python
def resize(img: jax.Array, *, H: int, W: int) -> jax.Array:
    """Resize an image.

    Args:
        img: Input image.
        H: Output height.
        W: Output width.

    Returns:
        Resized image.
    """
```

`# type: ignore` and `# noqa` comments are debt and should be avoided.
When a third-party type stub leaves no clean alternative, prefer
`# pyright: ignore[<rule>]` so the suppression is targeted, and note
the upstream cause in a short comment.

### API design

- Do not over-optimize or over-generalize early. Generalization should
  follow real use cases, not anticipate them.
- Prefer simple, explicit APIs with one clear usage.

### ML and numerical code

- Prefer JAX for ML and numerical computation. Avoid mixing ML
  frameworks without a documented reason.
- Make randomness explicit: pass keys, set seeds, keep tests
  deterministic.

### GPU usage

- Always run `nvidia-smi` first to see which GPUs are free.
- Use the first free GPU unless told otherwise.
- Kill processes you start; do not leave them hanging.

### Markdown conventions (documentation)

- Use sentence case for headings: only the first word is capitalized,
  except proper nouns and acronyms (`API`, `JAX`, `GitHub`).
- Use repository-relative links between docs pages
  (`[Architecture](architecture.md)`), never absolute paths or raw
  GitHub URLs.

## Tests

FlowGym uses `pytest`. Tests live under `tests/`, organized to mirror
the package layout (`tests/training/`, `tests/caching/`,
`tests/base_estimator/`, …) where reasonable.

- Prefer multiple small unit tests over one large test.
- Prefer `pytest.mark.parametrize` over copy-pasted variants.
- Always include an explanatory message in `assert` statements:

  ```python
  assert x == y, f"Expected {y}, got {x}"
  ```

- Check `tests/conftest.py` for existing fixtures before adding new
  ones.
- Add an integration test when behavior crosses module boundaries.

## Implementation workflow

The standard rhythm for any code change in FlowGym is test-first:

1. Write a failing test that defines the intended behavior. For a bug,
   reproduce the bug minimally; for a new feature, define the desired
   API.
2. Run `uv run pytest` and confirm the test fails for the right
   reason.
3. Implement the minimal change in the right module under
   `src/flowgym/` (use the [Architecture guide](architecture.md) if
   you are not sure which module owns the change).
4. Run the lint and format gate:

   ```bash
   uv run pre-commit run --all-files
   ```

5. Run the full suite and confirm it passes:

   ```bash
   uv run pytest
   ```

6. Run the type-check gate:

   ```bash
   uv run pre-commit run --hook-stage push --all-files
   ```

7. Update docstrings and the relevant docs pages for any user-visible
   change.

A few common variants follow the same rhythm:

- **Refactor (no behavior change):** keep changes small and coherent.
  Confirm existing tests cover the behavior; if coverage is weak, add
  characterization tests first. Do not introduce new abstractions
  along the way.
- **External API integration:** validate the upstream behavior in a
  scratch script first (do not commit the scratch). Capture the
  conclusions in tests, the integration code, and a short docstring or
  comment explaining the upstream contract.
- **Documentation-only changes:** still run the lint gate if you
  touched any Python file (e.g. an example). Build the docs and
  confirm the strict build passes.

### Quality gates

A change is complete when, on the touched files:

- `uv run pre-commit run --all-files` passes
- `uv run pytest` passes
- `uv run pre-commit run --hook-stage push --all-files` passes
- Documentation reflects user-visible behavior changes

## Contributing to documentation

Documentation is treated as part of the surface area: broken examples
and stale paths are bugs, not editorial nits.

### Where pages live

- `docs/getting-started/` — first-time user path.
- `docs/user-guide/` — concept-first guidance and workflow pages.
- `docs/examples/` — curated walkthroughs tied to scripts under
  `examples/`.
- `docs/api/` — curated reference layer for public modules.
- `docs/contribute/` — this section: contributor's guide and
  architecture.

### Editing

- Edit the canonical page; never duplicate prose into wrappers or
  README-side copies.
- If you change a public API, update the relevant guide page **and**
  the matching `docs/api/` page in the same change.
- If you add a new example script, add a page under `docs/examples/`
  and surface it from the examples index.

### Building and previewing

The strict build (`-W --keep-going`) is what the docs review
checklist requires before merge. The live-reload server is the fastest
way to iterate on prose; both commands are listed in
[Daily commands](#daily-commands) above.

### Docs review checklist

Before finishing a docs-affecting change:

- user-visible behavior is reflected in docs
- public examples match current signatures and import paths
- new or changed APIs are listed in the right reference page
- internal links resolve and are repository-relative
- the strict Sphinx docs build passes

## Opening a pull request

### Branch naming

Use a topic branch with a short descriptive prefix and a
hyphen-separated description, e.g. `fix/raft-jax-cache-id`,
`feat/dummy-estimator-training`, `docs/contributor-guide-rewrite`.

### Target branch

PRs go against `dev`, not `main`. `dev` is the working integration
branch; `main` is updated centrally from `dev` on release.

### Commit messages

Commits follow the [Conventional Commits](https://www.conventionalcommits.org/)
format defined in `.gitmessage`. The subject line is
`<type>(<scope>): <subject>`, where `<type>` is one of
`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
`chore`, `ci`. Examples:

```
feat(estimator): add consensus flow estimator
fix(caching): respect warm_start=index when cache is empty
docs(user-guide): rewrite estimator API page
```

The PR title must follow the same format — it is checked by the
`commit-lint` workflow on every PR.

### What CI runs

Three GitHub Actions workflows live under `.github/workflows/`:

- **`test.yaml`** runs `pytest` on the PR.
- **`code-style.yaml`** runs the pre-commit suite.
- **`commit-lint.yaml`** validates the PR title against the conventional
  commit format.

These workflows are configured for manual dispatch, so a maintainer
will trigger them on your PR. Run the local equivalents
(`uv run pre-commit run --all-files`, `uv run pre-commit run
--hook-stage push --all-files`, and `uv run pytest`) before pushing so
the maintainer's run passes on the first try.

### After review

When review feedback comes in, prefer a new commit on the same branch
over force-pushing — it makes the diff between revisions easier to
follow. Squash on merge is up to the maintainer.

Do not push the merge yourself; the project maintainers handle the
push to `dev` (and from there to `main`) centrally.

## Related pages

- [Architecture guide](architecture.md)
- [Getting started](../getting-started/index.md)
- [User guide](../user-guide/index.md)
- [Example workflows](../examples/index.md)
- [API reference](../api/index.md)
