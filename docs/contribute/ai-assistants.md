# Using AI assistants

Flow Gym is set up so AI coding assistants can pick up the same
[Contributor's guide](contributing.md) humans use, with minimal
per-tool configuration.

## How it's wired

| File | Purpose | Read by |
|---|---|---|
| `AGENTS.md` (repo root) | Canonical agent-facing instruction set; pointer to the contributor's guide | Codex CLI, GitHub Copilot, Cursor, Gemini CLI, and other tools that follow the [agents.md](https://agents.md) convention |
| `.claude/CLAUDE.md` | One-line bridge: `@../AGENTS.md` | Claude Code, which does not auto-read `AGENTS.md` |

The actual content lives in
[`docs/contribute/contributing.md`](contributing.md). `AGENTS.md` is a
short pointer; updating one place keeps every assistant in sync.

## Adding support for another tool

If the tool reads `AGENTS.md` natively, no action is needed. If it
doesn't, add a single shim file in the location that tool expects,
either using the tool's import mechanism (the way `.claude/CLAUDE.md`
does with `@../AGENTS.md`) or a one-line redirect to `AGENTS.md` and
the contributor's guide.

The rule is: wrappers stay thin pointers; canonical content lives in
the contributor's guide.
