"""Per-frame comparison of the Table 1 routers (paper Appendix E).

For each router and seed, write to ``results.json`` the share of test frames
where its EPE is lower than (better), equal to (tie) or higher than (worse)
that of its row's best single estimator, from the per-frame summaries of a
``table_1`` run. The spread and tails are in ``table_1/results.json``.

Usage, from the repository root, after ``table_1`` (see README.md):

    uv run python experiments/art_of_piv/appendix/E_epe_statistics/stats.py
"""

import json
import statistics
from pathlib import Path

import numpy as np

from flowgym.experiments import update_readme_results

# Router and best single estimator of each row, as in table_1/exp.yaml
ROWS = {
    "models/router_dis_20hz.yaml": "models/best_single_dis_20hz.yaml",
    "models/router_dis_1khz.yaml": "models/best_single_dis_1khz.yaml",
    "models/router_openpiv_1khz.yaml": "models/best_single_openpiv.yaml",
    "models/router_openpiv_4khz.yaml": "models/best_single_openpiv.yaml",
    "models/router_raft32.yaml": "models/best_single_raft32.yaml",
}
SEEDS = ("0", "1", "2")

HERE = Path(__file__).resolve().parent
SUMMARIES = HERE.parents[1] / "table_1" / "_generated" / "summaries"


def frame_epes(model: str, seed: str) -> np.ndarray:
    """Each test frame's EPE in one run's summary, ordered by file name.

    Args:
        model: Model entry of ``table_1/exp.yaml``.
        seed: Seed of the run.

    Returns:
        EPE per frame.
    """
    summary = json.loads((SUMMARIES / model / f"seed{seed}.json").read_text())
    frames = summary["frames"]
    epe = dict(zip(frames["file"], frames["epe"], strict=True))
    return np.array([epe[f] for f in sorted(epe)])


def main() -> None:
    """Write each router's better, tie and worse shares over seeds."""
    results = {}
    for router, best in ROWS.items():
        per_seed = {}
        for seed in SEEDS:
            r, b = frame_epes(router, seed), frame_epes(best, seed)
            per_seed[seed] = {
                "better_pct": 100 * float(np.mean(r < b)),
                "tie_pct": 100 * float(np.mean(r == b)),
                "worse_pct": 100 * float(np.mean(r > b)),
            }
        results[router] = {
            name: {
                "mean": statistics.mean(v[name] for v in per_seed.values()),
                "std": statistics.pstdev(v[name] for v in per_seed.values()),
                "seeds": {s: v[name] for s, v in per_seed.items()},
            }
            for name in per_seed["0"]
        }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    update_readme_results(HERE / "README.md")


if __name__ == "__main__":
    main()
