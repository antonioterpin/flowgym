"""Agent execution time of the deployed and larger routers (paper Table 5).

For each library (DIS and OpenPIV at 1 kHz) and router (the deployed fast
one and the larger standard one), build the ART of PIV estimator from its
YAML, force the router to pick each member in turn (head bias), and time the
full jitted call on one 256x256 frame. The agent time is that call minus the
member alone, averaged over the members; it is written to
``agent_time.json``.

Usage, from the repository root on an idle GPU (see README.md):

    uv run python \\
        experiments/art_of_piv/appendix/A.3_agent_architecture/agent_time.py
"""

from __future__ import annotations

import json
import statistics
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from flowgym.common.base import NNEstimatorTrainableState
from flowgym.experiments import update_readme_results
from flowgym.make import make_estimator
from flowgym.types import CachePayload
from flowgym.utils import load_configuration

B, H, W = 1, 256, 256
REPETITIONS, EXECUTIONS = 5, 100
MODELS = "experiments/art_of_piv/models"
ROUTERS = {
    "deployed": f"{MODELS}/router/fast.yaml",
    "large": f"{MODELS}/router/standard.yaml",
}
LIBRARIES = {
    "dis": "experiments/art_of_piv/libraries/dis_1khz.yaml",
    "openpiv": "experiments/art_of_piv/libraries/openpiv_1khz.yaml",
}
HERE = Path(__file__).resolve().parent


def call_ms(config: dict, member: int | None = None) -> float:
    """Best per-call time (ms) of an estimator's jitted call on one frame.

    Args:
        config: Estimator config.
        member: For a router, the library member it is forced to pick.

    Returns:
        The minimum over repetitions of the mean time per call.
    """
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    img1 = jax.random.uniform(key1, (B, H, W)) * 255.0
    img2 = jax.random.uniform(key2, (B, H, W)) * 255.0
    ts, create_state, compute, _ = make_estimator(
        config, image_shape=(B, H, W), estimate_shape=(B, H, W, 2), rng=0
    )
    if member is not None:
        assert isinstance(ts, NNEstimatorTrainableState)
        ts = force_member(ts, member)
    state = create_state(img1, key1)
    payload = CachePayload(is_training=False)

    def run() -> None:
        jax.block_until_ready(compute(img2, state, ts, cache_payload=payload))

    run()  # Compile and autotune.
    times = timeit.repeat(run, number=EXECUTIONS, repeat=REPETITIONS)
    return min(times) / EXECUTIONS * 1e3


def force_member(
    ts: NNEstimatorTrainableState, j: int
) -> NNEstimatorTrainableState:
    """Bias the router's output layer so that it always picks member ``j``.

    Args:
        ts: Router trainable state.
        j: Member index.

    Returns:
        The trainable state with biased heads.
    """
    params = unfreeze(ts.params)
    for name, head in params.items():
        if not name.startswith("head"):
            continue
        last = head[max(head, key=lambda k: int(k.split("_")[-1]))]
        if name == "head":  # Single-arm head: one output per member.
            last["bias"] = jnp.full_like(last["bias"], -1e6).at[j].set(1e6)
        else:  # One head per member.
            value = 1e6 if name == f"head_{j}" else -1e6
            last["bias"] = jnp.full_like(last["bias"], value)
    return ts.replace(params=freeze(params))


def main() -> None:
    """Time both routers on both libraries and write agent_time.json."""
    print("device:", jax.devices()[0])
    results = {}
    for lib, lib_path in LIBRARIES.items():
        members = load_configuration(lib_path)["estimators"]
        member_ms = [call_ms(load_configuration(m)) for m in members]
        for router, router_path in ROUTERS.items():
            config = load_configuration(router_path)
            config["config"]["estimators_list_path"] = lib_path
            agent_ms = [
                call_ms(config, j) - ms for j, ms in enumerate(member_ms)
            ]
            results[f"{lib}_{router}_ms"] = round(statistics.fmean(agent_ms), 4)
            print(lib, router, [round(t, 4) for t in agent_ms], flush=True)
    results["device"] = jax.devices()[0].device_kind
    (HERE / "agent_time.json").write_text(json.dumps(results, indent=2) + "\n")
    update_readme_results(HERE / "README.md")


if __name__ == "__main__":
    main()
