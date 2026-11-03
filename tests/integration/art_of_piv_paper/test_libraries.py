"""The ART of PIV paper's libraries, pools and their selection.

The fast tests check that the committed library and pool files hash to the
caches the paper used. The ``run_explicitly`` tests download the pinned
caches and re-run each selection (Table 1 and the A.1 libraries) with
``scripts/select_ensemble.py``
(``pytest -m run_explicitly tests/integration/art_of_piv_paper``).
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from flowgym import ALL_ESTIMATORS
from flowgym.utils import load_configuration

EXP = Path("experiments/art_of_piv")

# Greedy DIS libraries of appendix A.1; abl_n<K> is the first K of these.
ABL_HASHES = (
    "fe074ef0 e48f5064 ab441c2c 259d75c5 3cf3cb36 "
    "be1eaf6d f6cef0e8 60471ef2 551175b8 c036e3c6 "
    "b68adbad 8b9c3640 b67b6d2a 919d720b 99cf63bf "
    "7ae7658d cb42e3a5 1f2e4378 ded331fe 845842bb"
).split()

# Table 1 row: (pool, budget on mean latency in ms = 1 / nominal rate,
# best single estimator)
TABLE_1 = {
    "dis_20hz": ("dis", 50.0, "best_single_dis_20hz"),
    "dis_1khz": ("dis", 1.0, "best_single_dis_1khz"),
    "openpiv_1khz": ("openpiv", 1.0, "best_single_openpiv"),
    "openpiv_4khz": ("openpiv", 0.25, "best_single_openpiv"),
}


def _cache_id(cache_id_base: str, model: dict) -> str:
    estimator = ALL_ESTIMATORS[model["estimator"]].from_config(model["config"])
    return cache_id_base + estimator.get_cache_id_suffix(None)


def _model_cache_ids(cache_id_base: str, paths: list[str]) -> list[str]:
    return [_cache_id(cache_id_base, load_configuration(p)) for p in paths]


def _pool_cache_ids(pool: str) -> tuple[str, list[str]]:
    """Expand a pool file; a list-valued knob stands for each of its values."""
    spec = load_configuration(EXP / "pools" / f"{pool}.yaml")
    base = load_configuration(spec["base"])
    ids = []
    for entry in spec["candidates"]:
        values = [v if isinstance(v, list) else [v] for v in entry.values()]
        for combo in itertools.product(*values):
            knobs = {
                k: v for k, v in zip(entry, combo, strict=True) if v is not None
            }
            config = {**base["config"], **knobs}
            ids.append(
                _cache_id(spec["cache_id_base"], {**base, "config": config})
            )
    return spec["cache_id_base"], ids


@pytest.mark.parametrize(
    ("row", "hashes"),
    [
        *(
            (f"abl_n{k}", " ".join(ABL_HASHES[:k]))
            for k in (1, 2, 3, 4, 5, 10, 15, 20)
        ),
        (
            "dis_20hz",
            "007d9a94 86725532 9f485cd6 ad340132 b68adbad "
            "c072b2f2 c83b3862 cc366684 f37adac3 f581f19f",
        ),
        (
            "dis_1khz",
            "121533ae 37c4290e 555d463e 65957b24 6aec2a81 "
            "9215fc07 dae3d3ef dc377309 dcbc9026 ea3880db",
        ),
        (
            "openpiv_1khz",
            "056ddd95 05ed395f 36659c05 4b514659 6047fd62 "
            "8a926801 b2f7f13d c4a2b3db d1bb2725 f8badbaa",
        ),
        (
            "openpiv_4khz",
            "10a2232b 36659c05 5ffd819a 76590e9e 7bf46b30 "
            "8a926801 c4a2b3db d1bb2725 dbf97772 effb56b3",
        ),
    ],
)
def test_libraries_hit_the_paper_caches(row, hashes):
    """The merged library configs hash to the cache ids the paper used."""
    library = load_configuration(EXP / "libraries" / f"{row}.yaml")
    base = library["cache_id_base"]
    cache_ids = _model_cache_ids(base, library["estimators"])
    assert cache_ids == [f"{base}_c{h}" for h in hashes.split()]


def test_pools_hold_the_table_1_selections():
    """Every library member and best single of Table 1 is in its pool."""
    pools = {pool: _pool_cache_ids(pool) for pool in ("dis", "openpiv")}
    for _, ids in pools.values():
        assert len(set(ids)) == len(ids)
    for row, (pool, _, best_single) in TABLE_1.items():
        prefix, ids = pools[pool]
        library = load_configuration(EXP / "libraries" / f"{row}.yaml")
        models = [
            *library["estimators"],
            str(EXP / "table_1" / "models" / f"{best_single}.yaml"),
        ]
        assert set(_model_cache_ids(prefix, models)) <= set(ids)


def _select_ensemble():
    spec = importlib.util.spec_from_file_location(
        "select_ensemble", "scripts/select_ensemble.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pins() -> tuple[str, str]:
    """The pinned caches and tune split, as the experiments use them."""
    train = load_configuration(EXP / "datasets" / "train.yaml")
    tune = train["file_list"][0].replace(":train", ":tune")
    return train["caching"]["root_dir"], tune


def _select(tmp_path: Path, prefix: str, k: int, *flags: str) -> dict:
    caches, tune = _pins()
    out = tmp_path / f"{prefix}_{k}_{'_'.join(flags)}.json"
    argv = ["--cache-root", caches, "--keys-from", tune, "--K", str(k)]
    argv += ["--cache-id-prefix", prefix, "--out", str(out), *flags]
    assert _select_ensemble().main(argv) == 0
    return json.loads(out.read_text())


@pytest.mark.run_explicitly
@pytest.mark.parametrize("row", list(TABLE_1))
def test_table_1_selections_reproduce(row, tmp_path):
    """Exact 10-member library and best single (K = 1) within the budget."""
    pool, budget_ms, best_single = TABLE_1[row]
    prefix = f"piv-class1-{pool}"
    budget = ["--time-limit", str(budget_ms)]
    library = load_configuration(EXP / "libraries" / f"{row}.yaml")
    picked = _select(tmp_path, prefix, 10, *budget, "--exact")
    assert [m["cache_id"] for m in picked["exact"]["selected"]] == (
        _model_cache_ids(prefix, library["estimators"])
    )
    # K = 1 is greedy, which keeps the first cache on ties (the OpenPIV best
    # single has an exact twin without sig2noise_threshold)
    single = _select(tmp_path, prefix, 1, *budget)
    best = str(EXP / "table_1" / "models" / f"{best_single}.yaml")
    assert [m["cache_id"] for m in single["greedy"]["selected"]] == (
        _model_cache_ids(prefix, [best])
    )


@pytest.mark.run_explicitly
def test_a1_selection_reproduces(tmp_path):
    """Greedy over the whole DIS pool; abl_n<K> is the first K of 20."""
    prefix = "piv-class1-dis"
    picked = _select(tmp_path, prefix, 20)
    order = [m["cache_id"] for m in picked["greedy"]["selected"]]
    for k in (1, 2, 3, 4, 5, 10, 15, 20):
        library = load_configuration(EXP / "libraries" / f"abl_n{k}.yaml")
        assert order[:k] == _model_cache_ids(prefix, library["estimators"])


@pytest.mark.run_explicitly
@pytest.mark.parametrize("pool", ["dis", "openpiv"])
def test_pools_cover_every_cache(pool):
    """Every cached candidate with complete tune errors is in the pool."""
    caches, tune = _pins()
    se = _select_ensemble()
    prefix, ids = _pool_cache_ids(pool)
    _, cached, _ = se._load_cache(
        se._path_or_hf(caches),
        "epe",
        False,
        keys_from=se._path_or_hf(tune),
        cache_id_prefix=prefix,
    )
    assert set(cached) <= set(ids)
