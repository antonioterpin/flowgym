# FlowGym project documentation

This folder contains all official project documentation, rules, workflows, and guides for the FlowGym repository. All agents and contributors should follow these documents.

## 🎯 Quick start

1. **New to the project?** Start with [Architecture Guide](docs/guides/architecture.md)
2. **Starting a task?** Use the appropriate [workflow](docs/workflows/)
3. **Need a specific rule?** Check [Standards](docs/standards/)
4. **Contributing?** See [Contributing Guide](docs/guides/contributing.md)

## 📋 Standards

Non-negotiable project rules for code quality, testing, and development practices:

- [Environment & Tooling](docs/standards/environment-tooling.md) - Using uv, managing dependencies
- [Code Quality Gates](docs/standards/code-quality.md) - Ruff, BasedPyright, pre-commit
- [Linting & Formatting](docs/standards/linting-formatting.md) - Import rules, Ruff policy
- [Typing & Docstrings](docs/standards/typing-docstrings.md) - Google-style, types in signatures
- [Code Clarity](docs/standards/code-clarity.md) - Comments, readability, markdown heading capitalization, and relative markdown links
- [Exploration & Validation](docs/standards/exploration-validation.md) - Scratch scripts and design validation
- [ML & Numerical Code](docs/standards/ml-numerical.md) - JAX preference, determinism
- [Testing Policy](docs/standards/testing.md) - pytest, fixtures, best practices
- [API Design](docs/standards/api-design.md) - Avoiding over-engineering
- [Version Control](docs/standards/version-control.md) - Commit discipline
- [Code Organization](docs/standards/code-organization.md) - Structure and patterns
- [Change Scope](docs/standards/change-scope.md) - Focused, logical commits
- [Device Utilization](docs/standards/device-utilization.md) - GPU management

## 🔄 Workflows

Step-by-step procedures for common development tasks:

- [Feature Workflow](docs/workflows/feature.md) - Implement standard features (test-first approach)
- [Bugfix Workflow](docs/workflows/bugfix.md) - Identify, fix, and verify bugs
- [Refactor Workflow](docs/workflows/refactor.md) - Structural improvements without behavior change
- [API Validation Workflow](docs/workflows/api-validation.md) - Explore external APIs with scratch scripts
- [Documentation Workflow](docs/workflows/docs.md) - Documentation-only changes
- [Orientation Workflow](docs/workflows/orientation.md) - Initial repository navigation and setup

## 📖 Guides

Comprehensive guides for understanding and contributing to the project:

- [Architecture Guide](docs/guides/architecture.md) - Repository layout, module responsibilities, entry points
- [Contributing Guide](docs/guides/contributing.md) - Development workflow, testing, PR preparation
- [Agent Development](docs/guides/agent-development.md) - How to use this documentation with agents

## 👤 Agent personas

Predefined roles for different types of agentic work:

- [Implementer Agent](agents/implementer.md) - Task implementer for features and fixes
- [Reviewer Agent](agents/reviewer.md) - Code review and quality assessment

## 🔗 Cross-references

### By file type

| File Type | Location | Purpose |
|-----------|----------|---------|
| Source code | `src/flowgym/` | Core implementation |
| Tests | `tests/` | Unit and integration tests |
| Examples | `examples/` | Usage demonstrations |
| Experiments | `experiments/` | Experiment-specific scripts |
| Workflows (CI/CD) | `.github/workflows/` | GitHub Actions automation |

### Entry points

- **Main CLI**: `src/main.py`
- **Training**: `src/train.py` (RL) or `src/train_supervised.py` (supervised)
- **Evaluation**: `src/eval.py`
- **Comparison**: `src/compare.py`

### Configuration

- **Project config**: `pyproject.toml`
- **Claude Code**: `.claude/CLAUDE.md` (or `.agent/` for agent workflows)
- **GitHub Copilot**: `.github/copilot-instructions.md`
- **Pre-commit hooks**: `.pre-commit-config.yaml`

## 📊 Documentation structure

```
docs/
├── index.md (this file)
├── standards/
│   ├── environment-tooling.md
│   ├── code-quality.md
│   ├── linting-formatting.md
│   ├── typing-docstrings.md
│   ├── code-clarity.md
│   ├── exploration-validation.md
│   ├── ml-numerical.md
│   ├── testing.md
│   ├── api-design.md
│   ├── version-control.md
│   ├── code-organization.md
│   ├── change-scope.md
│   └── device-utilization.md
├── workflows/
│   ├── feature.md
│   ├── bugfix.md
│   ├── refactor.md
│   ├── orientation.md
│   ├── api-validation.md
│   └── docs.md
├── guides/
│   ├── architecture.md
│   ├── contributing.md
│   └── agent-development.md
└── agents/
    ├── implementer.md
    └── reviewer.md
```

## 🤖 Using this documentation with agents

All documentation in this folder is designed to be agent-friendly:

- **Claude Code**: See `.claude/CLAUDE.md` for configuration
- **GitHub Copilot**: See `.github/copilot-instructions.md` for setup
- **Custom Agents**: Reference this index and follow the structured workflows

Each document includes clear procedures, examples, and done criteria for agents to follow.

## ✏️ Maintaining this documentation

When you update project standards or workflows:

1. Make changes directly to the appropriate file in this folder
2. All agents (Claude Code, Copilot, etc.) will automatically use the latest version
3. The compatibility layers in `.agent/`, `.github/`, and `.claude/` reference these files

This single-source-of-truth approach keeps documentation in sync across all agentic systems.
