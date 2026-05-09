# Typing & docstrings

## Type information placement

- **Type information belongs in function signatures**, not in docstrings.
- **Do not include type hints in `Args:` or `Returns:` sections.**
- Use **Google-style docstrings** with descriptions only.

## Metrics boundary types

- Use plain mutable `dict` objects while building metrics locally.
- Use Goggles `Metrics` at function boundaries, especially return types and
  logger-facing APIs.
- Treat `Metrics` as a read-only mapping. Consumers that need to add or change
  fields should copy it first with `dict(metrics)`.

This keeps metric construction simple while preventing downstream consumers from
accidentally mutating producer-owned metric data.

## Example

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

## Type ignore & lint suppressions

- **Never use `type: ignore` comments** without explicitly telling the user first.
- **Never use `# noqa` for specific rules** without explicitly telling the user first.
- Both are code debt and should only be used with awareness and acknowledgment.

## Documentation requirements

- All **public functions and classes must be documented**.
- Private helpers may omit docstrings if their intent is obvious.
- **All files must have a file-level docstring at the top**, including tests.
