"""Tests that exploration is disabled at eval time.

These tests pin two related guarantees:

1. ``eval.py`` rebuilds the ``CachePayload`` it forwards to the estimator with
   ``is_training=False`` (both when a payload is supplied and when it is not),
   so any estimator that branches on ``cache_payload.is_training`` sees the
   eval-time flag.
2. The ArtOfPIV router honours that flag — when ``is_training=False`` the
   exploration value is zeroed and the router falls back to ``argmax``-style
   greedy selection regardless of the configured exploration schedule.
"""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax

from eval import eval_density, eval_flow
from flowgym.flow.art_of_piv import router as art_of_piv_router
from flowgym.flow.art_of_piv.inference import estimate as inference_estimate
from flowgym.flow.art_of_piv.types import RouterOutputs
from flowgym.types import CachePayload


def _make_flow_batch():
    """Build a minimal ``SynthpixBatch`` for flow evaluation tests."""
    from synthpix import SynthpixBatch

    return SynthpixBatch(
        images1=jnp.zeros((1, 4, 4)),
        images2=jnp.zeros((1, 4, 4)),
        flow_fields=jnp.zeros((1, 4, 4, 2)),
    )


def _make_density_batch():
    """Build a minimal ``SynthpixBatch`` for density evaluation tests."""
    from synthpix import SynthpixBatch

    params = Mock()
    params.seeding_densities = jnp.zeros((1, 4, 4))
    return SynthpixBatch(
        images1=jnp.zeros((1, 4, 4)),
        images2=jnp.zeros((1, 4, 4)),
        flow_fields=jnp.zeros((1, 4, 4, 2)),
        params=params,
    )


def _flow_mocks():
    model = Mock()
    model.is_oracle.return_value = False
    model.process_metrics.return_value = {}
    dummy_state = {"estimates": jnp.zeros((1, 1, 4, 4, 2))}
    create_state_fn = Mock(return_value=dummy_state)
    compute_flow_fn = Mock(return_value=(dummy_state, {}))
    return model, Mock(), create_state_fn, compute_flow_fn


def _density_mocks():
    model = Mock()
    model.process_metrics.return_value = {}
    dummy_state = {"estimates": jnp.zeros((1, 1, 4, 4))}
    create_state_fn = Mock(return_value=dummy_state)
    compute_estimate_fn = Mock(return_value=(dummy_state, {}))
    return model, Mock(), create_state_fn, compute_estimate_fn


def test_eval_flow_disables_exploration_with_cache():
    """``eval_flow`` flips ``is_training`` to False on the supplied payload."""
    model, trained_state, create_state_fn, compute_flow_fn = _flow_mocks()
    cache_payload = CachePayload(extras={"some_key": "some_value"})

    eval_flow(
        estimator=model,
        trained_state=trained_state,
        create_state_fn=create_state_fn,
        compute_flow_fn=compute_flow_fn,
        batch=_make_flow_batch(),
        key=jax.random.PRNGKey(0),
        cache_payload=cache_payload,
    )

    assert compute_flow_fn.called
    passed = compute_flow_fn.call_args.kwargs["cache_payload"]
    assert passed is not None
    assert passed.is_training is False
    # Existing extras must be preserved when we replace the field.
    assert passed.extras.get("some_key") == "some_value"


def test_eval_flow_disables_exploration_without_cache():
    """``eval_flow`` builds a fresh ``is_training=False`` payload when None."""
    model, trained_state, create_state_fn, compute_flow_fn = _flow_mocks()

    eval_flow(
        estimator=model,
        trained_state=trained_state,
        create_state_fn=create_state_fn,
        compute_flow_fn=compute_flow_fn,
        batch=_make_flow_batch(),
        key=jax.random.PRNGKey(0),
        cache_payload=None,
    )

    assert compute_flow_fn.called
    passed = compute_flow_fn.call_args.kwargs["cache_payload"]
    assert passed is not None
    assert passed.is_training is False


def test_eval_flow_disables_exploration_on_short_circuit():
    """The short-circuit path must still leave ``is_training=False``.

    Today the short-circuit returns before any router-aware code runs, but if a
    future change makes it fall through (e.g. to recompute auxiliary metrics
    from a cached flow), the invariant "payload leaving eval has
    ``is_training=False``" must still hold.
    """
    from eval import _disable_exploration

    payload = CachePayload(
        has_precomputed_errors=True,
        epe=np.zeros((1,), dtype=np.float32),
    )
    normalized = _disable_exploration(payload)
    assert normalized.is_training is False
    assert normalized.has_precomputed_errors is True
    assert normalized.epe is not None

    # Same invariant when no payload was supplied at all.
    fresh = _disable_exploration(None)
    assert fresh.is_training is False


def test_eval_density_disables_exploration():
    """``eval_density`` flips ``is_training`` to False before estimating."""
    model, trained_state, create_state_fn, compute_estimate_fn = (
        _density_mocks()
    )

    eval_density(
        estimator=model,
        trained_state=trained_state,
        create_state_fn=create_state_fn,
        compute_estimate_fn=compute_estimate_fn,
        batch=_make_density_batch(),
        key=jax.random.PRNGKey(0),
        cache_payload=None,
    )

    assert compute_estimate_fn.called
    passed = compute_estimate_fn.call_args.kwargs["cache_payload"]
    assert passed is not None
    assert passed.is_training is False


def test_inference_estimate_disables_router_exploration_when_not_training():
    """Router sees zero exploration and goes greedy when payload says eval.

    This complements the ``eval.py`` tests above by exercising the actual
    consumer of ``cache_payload.is_training`` — ``art_of_piv.inference`` — and
    asserting that the ``run_router_fn`` it dispatches to is invoked with
    ``train=False``. We further verify that the real ``router.run_router``
    zeros its exploration rate in that mode and selects ``argmax(q_values)``.
    """
    batch_size, height, width = 2, 4, 4
    state = {
        "images": jnp.zeros((batch_size, 2, height, width), dtype=jnp.float32),
        "estimates": jnp.zeros(
            (batch_size, 2, height, width, 2), dtype=jnp.float32
        ),
        "keys": jnp.zeros((batch_size, 2), dtype=jnp.uint32),
    }
    curr = jnp.zeros((batch_size, height, width), dtype=jnp.float32)
    trainable_state = Mock()
    trainable_state.extras = {
        "steps": jnp.zeros((batch_size, 1), dtype=jnp.int32),
    }

    captured: dict[str, bool] = {}

    def fake_run_router(prev, _curr, _state, _ts, *, flow, train):
        # Record what the router was asked to do.
        captured["train"] = bool(train)
        # Use the production router with a constant schedule + epsilon_greedy
        # so the exploration-gating behaviour is exercised end-to-end.
        keys = jnp.zeros((prev.shape[0], 2), dtype=jnp.uint32)
        # Q-values where the greedy action is always index 1.
        q_values = jnp.tile(
            jnp.asarray([0.0, 1.0], dtype=jnp.float32)[None, :],
            (prev.shape[0], 1),
        )
        outputs = art_of_piv_router.run_router(
            prev,
            _curr,
            keys=keys,
            steps=_ts.extras["steps"],
            q_estimate_fn=lambda *_a, **_kw: q_values,
            schedule=optax.constant_schedule(1.0),
            explore=lambda qs, eps, key: jnp.where(
                eps > 0.0,
                jnp.zeros(qs.shape[0], dtype=jnp.int32),  # explore -> idx 0
                jnp.argmax(qs, axis=-1),  # greedy -> idx 1
            ),
            flow=flow,
            train=train,
        )
        return RouterOutputs(*outputs)

    def fake_run_subestimators(
        _state,
        _ts,
        _curr,
        choices,
        _extras,
        _qs,
        _exploration_rate,
    ):
        flows = jnp.zeros((choices.shape[0], height, width, 2))
        return flows, {"choices": choices}, {"choices": choices}

    extras = {"cache_payload": CachePayload(is_training=False)}
    _flows, out_extras, _metrics = inference_estimate(
        curr,
        state,
        trainable_state,
        extras,
        use_temporal_propagation=False,
        use_last_estimate=False,
        run_router_fn=fake_run_router,
        run_subestimators_fn=fake_run_subestimators,
    )

    assert captured["train"] is False, (
        "router must be invoked with train=False when "
        "cache_payload.is_training is False"
    )
    np.testing.assert_array_equal(
        np.asarray(out_extras["choices"]),
        np.ones(batch_size, dtype=np.int32),
    )


def test_inference_estimate_keeps_router_exploration_when_training():
    """When ``is_training=True`` the router still receives ``train=True``."""
    batch_size, height, width = 2, 4, 4
    state = {
        "images": jnp.zeros((batch_size, 2, height, width), dtype=jnp.float32),
        "estimates": jnp.zeros(
            (batch_size, 2, height, width, 2), dtype=jnp.float32
        ),
        "keys": jnp.zeros((batch_size, 2), dtype=jnp.uint32),
    }
    curr = jnp.zeros((batch_size, height, width), dtype=jnp.float32)
    trainable_state = Mock()
    trainable_state.extras = {
        "steps": jnp.zeros((batch_size, 1), dtype=jnp.int32),
    }
    captured: dict[str, bool] = {}

    def fake_run_router(prev, _curr, _state, _ts, *, flow, train):
        captured["train"] = bool(train)
        choices = jnp.zeros(prev.shape[0], dtype=jnp.int32)
        qs = jnp.zeros((prev.shape[0], 2), dtype=jnp.float32)
        exploration_rate = jnp.zeros(prev.shape[0], dtype=jnp.float32)
        estimate_keys = jnp.zeros((prev.shape[0], 2), dtype=jnp.uint32)
        return RouterOutputs(
            actions=choices,
            q_values=qs,
            exploration_value=exploration_rate,
            estimate_keys=estimate_keys,
        )

    def fake_run_subestimators(
        _state,
        _ts,
        _curr,
        choices,
        _extras,
        _qs,
        _exploration_rate,
    ):
        flows = jnp.zeros((choices.shape[0], height, width, 2))
        return flows, {"choices": choices}, {"choices": choices}

    inference_estimate(
        curr,
        state,
        trainable_state,
        {"cache_payload": CachePayload(is_training=True)},
        use_temporal_propagation=False,
        use_last_estimate=False,
        run_router_fn=fake_run_router,
        run_subestimators_fn=fake_run_subestimators,
    )

    assert captured["train"] is True
