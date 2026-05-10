# Testing policy

## Test framework

- **Always use `pytest`** for testing.

## Test design

- Use **multiple small unit tests** rather than one large test.
- Add **integration tests** when behavior spans multiple components.

## Testing best practices

- Prefer `pytest.mark.parametrize` over repeated tests.
- **Always check `tests/conftest.py`** for existing fixtures before adding new ones.
- **Always include a descriptive message in `assert` statements** to explain what exactly failed and why (e.g., `assert x == y, f"Expected {y}, got {x}"`).
