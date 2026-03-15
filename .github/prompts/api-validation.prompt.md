---
description: Validate and integrate a new external library API
---

Follow the canonical rules in [copilot-instructions.md](../copilot-instructions.md).

Then follow the workflow in [docs/workflows/api-validation.md](../../docs/workflows/api-validation.md).

**Use when:** Integrating a new external library or API with uncertain behavior.

**Key principle:** Validate first in a scratch script before committing integration code.

**Output format:**
- Write a small validation script to confirm API behavior
- Summarize findings (1-5 bullets)
- Write failing tests defining the desired integration
- Implement minimal core change to make tests pass
- Delete the scratch script
- Confirm all gates pass
