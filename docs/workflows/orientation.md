---
description: Quick orientation before making changes
---

Use when you’re not sure where to implement something.

1. Locate the most relevant module under `src/flowgym/`:
   - preprocessing / utilities: `flowgym/common/`, `flowgym/utils.py`
   - configuration: `flowgym/config/`
   - models / networks: `flowgym/nn/`
   - training logic: `flowgym/training/`, `train*.py`
   - flow / estimators: `flowgym/flow/`
   - environments: `flowgym/environment/`
2. Search for existing patterns and helpers (reuse before adding new abstractions).
3. Check `tests/` for similar tests and existing fixtures (see `tests/conftest.py`).

**Done criteria:**
- You know *which module owns the change* and which tests are closest.
