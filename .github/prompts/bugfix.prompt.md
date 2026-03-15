---
description: Fix a bug or prevent a known regression
---

Follow the canonical rules in [copilot-instructions.md](../copilot-instructions.md).

Then follow the workflow in [docs/workflows/bugfix.md](../../docs/workflows/bugfix.md).

**Use when:** Fixing bugs or preventing known regressions.

**Output format:**
- Describe the bug and minimal reproduction
- Write a failing test first
- Implement the fix
- Show that test now passes and all gates pass
