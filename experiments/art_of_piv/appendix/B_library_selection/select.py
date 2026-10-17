"""Greedy versus exact library selection on the DIS pool (paper Appendix B).

For every library size K, pick K DIS configs minimizing the mean oracle EPE
(per tune frame, the error of the best member) with the greedy solver and
with the exact MILP of ``scripts/select_ensemble.py``, then write both
costs, the gap (in pixels and relative to the optimum) and the MILP's solve
time to ``results.json``.

Usage, from the repository root (see README.md):

    uv run --extra hf python \\
        experiments/art_of_piv/appendix/B_library_selection/select.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from synthpix.data_sources import hf_resolver

# Per-frame EPEs of every DIS config, and the tune split whose frames they
# are restricted to; both pinned as in experiments/art_of_piv/datasets.
CACHES = "hf://activefluidcontrol/piv-class1-caches@7068a49525da1d8c1b7bdf4027a3c3a74da5d02a"
TUNE = "hf://activefluidcontrol/piv-class1@ccc784ebbe86f16121bdc51ccb20b4cdf6b6a150:tune"
POOL_PREFIX = "piv-class1-dis"
SIZES = range(1, 21)
MILP_TIMEOUT_S = 300.0

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _load_select_ensemble():
    """Import ``scripts/select_ensemble.py`` (not a package) as a module.

    Returns:
        The loaded module.
    """
    path = REPO / "scripts" / "select_ensemble.py"
    spec = importlib.util.spec_from_file_location("select_ensemble", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run greedy and exact selection for every size and write results.

    Raises:
        RuntimeError: If the MILP proves no optimum for some size.
    """
    se = _load_select_ensemble()
    cache_root = hf_resolver.resolve_to_directory(CACHES)
    tune_dir = hf_resolver.resolve_to_directory(TUNE)

    E, cache_ids, _ = se._load_cache(
        cache_root,
        "epe",
        verbose=False,
        keys_from=tune_dir,
        cache_id_prefix=POOL_PREFIX,
    )
    # Configs that diverge on some frame cannot be selected.
    finite = np.isfinite(E).all(axis=1)
    E = E[finite]
    cache_ids = [c for c, keep in zip(cache_ids, finite, strict=True) if keep]
    print(f"pool: {E.shape[0]} DIS configs x {E.shape[1]} tune frames")

    def cost(selected: list[int]) -> float:
        return float(E[selected].min(axis=0).mean())

    sizes: dict[str, dict[str, object]] = {}
    for k in SIZES:
        greedy = se._greedy_select(E, k, "mean", False)
        start = time.perf_counter()
        exact, status, _ = se._exact_milp(E, k, MILP_TIMEOUT_S, False)
        seconds = time.perf_counter() - start
        # status=0 is HiGHS's proven optimum; a time-limit incumbent is not
        if exact is None or not status.startswith("status=0 "):
            raise RuntimeError(
                f"K={k}: MILP found no proven optimum ({status})"
            )
        g, e = cost(greedy), cost(exact)
        sizes[str(k)] = {
            "greedy_aepe": g,
            "exact_aepe": e,
            "gap_px": g - e,
            "gap_percent": (g - e) / e * 100.0,
            "milp_seconds": seconds,
            "milp_status": status,
            "greedy": [cache_ids[i] for i in greedy],
            "exact": sorted(cache_ids[i] for i in exact),
        }
        print(
            f"K={k:2d} greedy={g:.5f} exact={e:.5f} gap={g - e:.5f} px "
            f"({(g - e) / e * 100:.3f}%) milp={seconds:.1f}s"
        )

    results = {
        "pool_size": int(E.shape[0]),
        "dropped_nonfinite": int((~finite).sum()),
        "tune_frames": int(E.shape[1]),
        "sizes": sizes,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
