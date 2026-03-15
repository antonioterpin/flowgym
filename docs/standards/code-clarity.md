# Comments & code clarity

## Comment philosophy

- **Do not pollute the code with comments explaining reasoning step-by-step.**
- This applies to inline comments; docstrings are an exception.
- Comments should explain **why** something is done, not:
  - what the code does (the code should be readable)
  - how it does it (that should be clear from implementation)

## Good comment example

```python
# Keep gradient magnitudes comparable across resolutions to prevent training instability.
```

## Markdown header capitalization

- In markdown documentation, **only capitalize the first word** of headers.
- This rule applies to all Markdown files under `docs/`.
- Keep proper nouns/acronyms as needed (for example, `GitHub`, `API`, `JAX`).
- Examples:
  - ✅ `## Miss behavior` (not `## Miss Behavior`)
  - ✅ `## Mask handling` (not `## Mask Handling`)
  - ✅ `## Type ignore & lint suppressions` (not `## Type Ignore & Lint Suppressions`)

## Markdown link paths

- For links to files or folders in this repository, use **relative paths**.
- This rule applies to markdown documentation in the repository (for example `docs/`, `README.md`, `CONTRIBUTING.md`).
- Do not use absolute local paths such as `file:///...`, `/home/...`, or `C:\...`.
- Do not use repository-internal raw GitHub links (`https://raw.githubusercontent.com/...`) when the target file exists in this repo; use relative links so IDE navigation works.
- Examples:
  - ✅ `[Code clarity](../standards/code-clarity.md)`
  - ✅ `[Contributing](docs/guides/contributing.md)`
  - ❌ `[Code clarity](file:///home/user/project/docs/standards/code-clarity.md)`
  - ❌ `[Contributing](https://raw.githubusercontent.com/org/repo/main/CONTRIBUTING.md)`
