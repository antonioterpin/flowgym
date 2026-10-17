"""Re-render the results blocks of every experiment README.

Each ``<!-- BEGIN <file> -->`` ... ``<!-- END <file> -->`` block is rewritten
from the named results file next to the README (``run.py`` and the
experiment scripts do this after writing their results).

Usage, from the repository root:

    uv run python experiments/update_readmes.py
"""

from __future__ import annotations

from pathlib import Path

from flowgym.experiments import update_readme_results


def main() -> None:
    """Update every README under ``experiments/`` and list the changed ones."""
    for readme in sorted(Path(__file__).parent.rglob("README.md")):
        if update_readme_results(readme):
            print(f"updated {readme}")


if __name__ == "__main__":
    main()
