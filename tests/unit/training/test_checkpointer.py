"""Unit tests for ``flowgym.checkpointing.Checkpointer``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flowgym.checkpointing import CheckpointConfig, Checkpointer


class _FakeManager:
    """In-memory stand-in for ``ocp.CheckpointManager``.

    Mirrors the slice of the API the Checkpointer actually exercises
    (``save``, ``best_step``, ``metrics``, ``wait_until_finished``,
    ``close``) using the same best_fn/best_mode policy orbax does, so
    the tests stay decoupled from disk I/O but still validate the
    branching that depends on orbax's native best tracking.
    """

    def __init__(self, directory, options):
        self._dir = directory
        self._options = options
        self._best_fn = options.best_fn
        self._best_mode = options.best_mode
        self._steps: list[int] = []
        self._metrics: dict[int, dict] = {}
        self.wait_calls = 0
        self.close_calls = 0
        self.save_args: list[dict] = []

    def save(self, step, args, metrics=None):
        # Match orbax: silently reject duplicate or backward steps.
        if self._steps and step <= self._steps[-1]:
            self.save_args.append(
                {"step": step, "args": args, "metrics": metrics, "ok": False}
            )
            return False
        self._steps.append(step)
        if metrics is not None:
            self._metrics[step] = dict(metrics)
        self.save_args.append(
            {"step": step, "args": args, "metrics": metrics, "ok": True}
        )
        return True

    def best_step(self):
        scored = [(self._best_fn(m), s) for s, m in self._metrics.items()]
        if not scored:
            return None
        chosen = (
            max(scored, key=lambda x: x[0])
            if self._best_mode == "max"
            else min(scored, key=lambda x: x[0])
        )
        return chosen[1]

    def metrics(self, step):
        return self._metrics.get(step)

    def latest_step(self):
        return self._steps[-1] if self._steps else None

    def wait_until_finished(self):
        self.wait_calls += 1

    def close(self):
        self.close_calls += 1


def _make(tmp_path, cfg=None, name="DummyEstimator"):
    """Build a Checkpointer backed by a ``_FakeManager`` instance."""
    model = type(name, (), {})()
    fakes: list[_FakeManager] = []

    def _ctor(directory, options=None):
        fake = _FakeManager(directory, options)
        fakes.append(fake)
        return fake

    with (
        patch("flowgym.checkpointing.ocp.CheckpointManager", side_effect=_ctor),
        patch(
            "flowgym.checkpointing.build_save_args", return_value=MagicMock()
        ),
        patch("flowgym.checkpointing.logger") as logger_mock,
    ):
        ckpt = Checkpointer(
            out_dir=tmp_path, model=model, sampler=None, config=cfg
        )
        # Hand the test the live fake so it can introspect what was saved.
        yield ckpt, fakes[0], logger_mock


# ---------------------------------------------------------------------------
# CheckpointConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wandb_upload": "evrey"},
        {"wandb_upload": ""},
        {"keep": 0},
        {"keep": -1},
        {"keep": 1.5},  # pyright: ignore[reportArgumentType]
        {"metric_key": ""},
    ],
)
def test_checkpoint_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        CheckpointConfig(**kwargs)


def test_from_configs_reads_model_block_with_legacy_fallback():
    cfg = CheckpointConfig.from_configs(
        model_config={"checkpoint": {"wandb_upload": "every", "keep": 5}},
        dataset_config={"save_only_best": True},
    )
    assert cfg.wandb_upload == "every"
    assert cfg.keep == 5
    assert cfg.save_only_best is True


def test_from_configs_model_block_overrides_legacy():
    cfg = CheckpointConfig.from_configs(
        model_config={"checkpoint": {"save_only_best": False}},
        dataset_config={"save_only_best": True},
    )
    assert cfg.save_only_best is False


# ---------------------------------------------------------------------------
# Native best tracking via orbax (best_fn + best_mode)
# ---------------------------------------------------------------------------


def test_best_metric_none_before_any_save(tmp_path):
    for ckpt, _, _ in _make(tmp_path):
        assert ckpt.best_metric is None
        assert ckpt.best_step is None


def test_best_step_tracks_lower_is_better(tmp_path):
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.5})
        ckpt.on_validation(MagicMock(), 20, {"mean_error": 0.6})
        ckpt.on_validation(MagicMock(), 30, {"mean_error": 0.3})
        assert ckpt.best_step == 30
        assert ckpt.best_metric == pytest.approx(0.3)
        # All three saves landed with the expected metric keys.
        recorded = [c["metrics"] for c in fake.save_args if c["ok"]]
        assert [m["mean_error"] for m in recorded] == [0.5, 0.6, 0.3]


def test_best_step_tracks_higher_is_better(tmp_path):
    cfg = CheckpointConfig(
        wandb_upload="every", metric_key="accuracy", higher_is_better=True
    )
    for ckpt, _, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 10, {"accuracy": 0.8})
        ckpt.on_validation(MagicMock(), 20, {"accuracy": 0.7})
        ckpt.on_validation(MagicMock(), 30, {"accuracy": 0.9})
        assert ckpt.best_step == 30
        assert ckpt.best_metric == pytest.approx(0.9)


def test_on_validation_best_ignores_missing_or_nonfinite_metric(tmp_path):
    """``best`` never saves on a missing or non-finite metric."""
    cfg = CheckpointConfig(wandb_upload="best")
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        assert ckpt.on_validation(MagicMock(), 5, {}) is None
        assert (
            ckpt.on_validation(MagicMock(), 6, {"mean_error": float("nan")})
            is None
        )
        # Neither call reached orbax and no best was registered.
        assert not any(c["ok"] for c in fake.save_args)
        assert ckpt.best_metric is None


def test_on_validation_every_saves_even_on_nonfinite_metric(tmp_path):
    """``every`` snapshots the latest model after *every* validation.

    A non-finite (or missing) metric must still produce a save/upload
    — the documented "upload after every validation" contract — but the
    non-finite value must never become the tracked best, and the save
    carries no metrics so orbax cannot rank it as best.
    """
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        out = ckpt.on_validation(MagicMock(), 6, {"mean_error": float("nan")})
        assert out is not None
        assert len(fake.save_args) == 2
        # The NaN save reached orbax with no metrics attached.
        assert fake.save_args[-1]["ok"] is True
        assert fake.save_args[-1]["metrics"] is None
        # It uploads under "latest", never "best".
        last_aliases = logger_mock.artifact.call_args.args[0]["aliases"]
        assert "latest" in last_aliases and "best" not in last_aliases
        # Best tracker still reflects the finite 0.5 from step 5.
        assert ckpt.best_metric == pytest.approx(0.5)


def test_on_validation_rolls_back_best_when_orbax_rejects_save(tmp_path):
    """A rejected save in ``on_validation`` must not advance best trackers.

    If orbax rejects the write (duplicate / backward step), nothing
    lands on disk, so ``best_metric`` must stay at its prior value and
    the unsaved-best latch must not be left set (which would otherwise
    trip a spurious later ``save_periodic`` write).
    """
    cfg = CheckpointConfig(wandb_upload="best", save_only_best=True)
    for ckpt, _fake, _ in _make(tmp_path, cfg=cfg):
        # First improvement saves cleanly at step 10.
        ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.5})
        assert ckpt.best_metric == pytest.approx(0.5)
        # A "better" metric at a backward step is rejected by orbax.
        out = ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.2})
        assert out is None
        # No new on-disk checkpoint, so the tracker stays at 0.5 and the
        # latch is not left armed.
        assert ckpt.best_metric == pytest.approx(0.5)
        assert ckpt.save_periodic(MagicMock(), step=5) is None


# ---------------------------------------------------------------------------
# wandb_upload dispatch
# ---------------------------------------------------------------------------


def test_on_validation_never_does_not_save_or_upload(tmp_path):
    cfg = CheckpointConfig(wandb_upload="never")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        out = ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.3})
        assert out is None
        assert fake.save_args == []
        logger_mock.artifact.assert_not_called()


def test_on_validation_best_uploads_only_on_improvement(tmp_path):
    cfg = CheckpointConfig(wandb_upload="best")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        assert len(fake.save_args) == 1
        assert logger_mock.artifact.call_count == 1
        assert "best" in logger_mock.artifact.call_args.args[0]["aliases"]

        ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.6})
        assert len(fake.save_args) == 1
        assert logger_mock.artifact.call_count == 1

        ckpt.on_validation(MagicMock(), 15, {"mean_error": 0.2})
        assert len(fake.save_args) == 2
        assert logger_mock.artifact.call_count == 2


def test_on_validation_every_uploads_at_each_call(tmp_path):
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, _, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.6})
        ckpt.on_validation(MagicMock(), 15, {"mean_error": 0.4})
        assert logger_mock.artifact.call_count == 3
        # Best alias migrates with orbax's view of the best.
        step5 = logger_mock.artifact.call_args_list[0].args[0]["aliases"]
        step10 = logger_mock.artifact.call_args_list[1].args[0]["aliases"]
        step15 = logger_mock.artifact.call_args_list[2].args[0]["aliases"]
        assert "best" in step5 and "latest" in step5
        assert "latest" in step10 and "best" not in step10
        assert "best" in step15 and "latest" in step15


def test_artifact_payload_uses_model_class_name(tmp_path):
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, _, logger_mock in _make(tmp_path, cfg=cfg, name="MyEstimator"):
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        payload = logger_mock.artifact.call_args.args[0]
        assert payload["name"] == "MyEstimator_checkpoint"
        assert payload["type"] == "checkpoint"
        assert "path" in payload


# ---------------------------------------------------------------------------
# Periodic / final saves
# ---------------------------------------------------------------------------


def test_save_periodic_unconditional(tmp_path):
    cfg = CheckpointConfig(save_only_best=False)
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.save_periodic(MagicMock(), step=10)
        ckpt.save_periodic(MagicMock(), step=20)
        assert len(fake.save_args) == 2
        logger_mock.artifact.assert_not_called()


def test_save_periodic_only_after_new_best(tmp_path):
    cfg = CheckpointConfig(save_only_best=True)
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        # No validation yet → skip.
        assert ckpt.save_periodic(MagicMock(), step=10) is None
        assert fake.save_args == []
        # First validation flips the latch (no save here because
        # wandb_upload=never).
        ckpt.on_validation(
            MagicMock(), step=15, val_metrics={"mean_error": 0.5}
        )
        assert ckpt.save_periodic(MagicMock(), step=20) is not None
        assert len(fake.save_args) == 1
        # Second periodic call without a new best → skip.
        assert ckpt.save_periodic(MagicMock(), step=30) is None
        assert len(fake.save_args) == 1
        # New best → next periodic save fires.
        ckpt.on_validation(
            MagicMock(), step=35, val_metrics={"mean_error": 0.3}
        )
        assert ckpt.save_periodic(MagicMock(), step=40) is not None
        assert len(fake.save_args) == 2


def test_save_only_best_with_wandb_never_skips_regressions(tmp_path):
    """Regression for codex P1 on #158.

    With ``wandb_upload="never"`` and ``save_only_best=True``,
    validation events don't reach orbax (no save → no metrics on
    disk). Without a local best-metric tracker, every subsequent
    validation would compare against ``None``, register as a "new
    best", and trip the latch — so periodic saves would fire on
    every cycle instead of only on real improvements.

    Sequence: first observation 0.5 (best), regression 0.6, regression
    0.7 — only the first periodic save should fire.
    """
    cfg = CheckpointConfig(save_only_best=True, wandb_upload="never")
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        assert ckpt.save_periodic(MagicMock(), 10) is not None
        assert len(fake.save_args) == 1

        # Regression: no new best, latch must stay down.
        ckpt.on_validation(MagicMock(), 15, {"mean_error": 0.6})
        assert ckpt.save_periodic(MagicMock(), 20) is None
        assert len(fake.save_args) == 1

        # Another regression: still no save.
        ckpt.on_validation(MagicMock(), 25, {"mean_error": 0.7})
        assert ckpt.save_periodic(MagicMock(), 30) is None
        assert len(fake.save_args) == 1

        # Genuine improvement: save fires again.
        ckpt.on_validation(MagicMock(), 35, {"mean_error": 0.4})
        assert ckpt.save_periodic(MagicMock(), 40) is not None
        assert len(fake.save_args) == 2


def test_save_periodic_carries_latest_observed_metrics(tmp_path):
    """Periodic saves attach the most recent validation metrics.

    Without this, orbax's on-disk best tracking would diverge from
    our local tracker (the in-memory ``_seen_best`` would say
    "metric=0.3 is the best" while ``best_step`` would point at a
    metricless save and the stored ``metrics`` for it would be
    empty).
    """
    cfg = CheckpointConfig(save_only_best=False)
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.3})
        ckpt.save_periodic(MagicMock(), 10)
        assert fake.save_args[-1]["metrics"] == {"mean_error": 0.3}


def test_best_metric_survives_resume(tmp_path):
    """A fresh Checkpointer on the same dir bootstraps ``best_metric``
    from orbax's persisted metrics.

    Exercises the real ``__init__`` bootstrap path: the underlying
    manager is pre-seeded with prior-run state *before* the
    Checkpointer is constructed, so the assertion observes what the
    constructor read out of orbax — not a value the test wrote in
    after the fact.
    """
    cfg = CheckpointConfig(wandb_upload="every")
    model = type("DummyEstimator", (), {})()
    fakes: list[_FakeManager] = []

    def _ctor(directory, options=None):
        fake = _FakeManager(directory, options)
        # Simulate a prior run that already wrote two checkpoints.
        fake._steps = [5, 10]
        fake._metrics = {5: {"mean_error": 0.5}, 10: {"mean_error": 0.2}}
        fakes.append(fake)
        return fake

    with (
        patch("flowgym.checkpointing.ocp.CheckpointManager", side_effect=_ctor),
        patch(
            "flowgym.checkpointing.build_save_args", return_value=MagicMock()
        ),
        patch("flowgym.checkpointing.logger"),
    ):
        ckpt2 = Checkpointer(
            out_dir=tmp_path, model=model, sampler=None, config=cfg
        )
    # Constructor read orbax's persisted best (step 10, mean_error=0.2)
    # and seeded ``_seen_best`` from it — without the test touching it.
    assert ckpt2.best_step == 10
    assert ckpt2.best_metric == pytest.approx(0.2)


def test_on_validation_save_clears_unsaved_best_latch(tmp_path):
    """A save through ``on_validation`` must clear the unsaved-best flag.

    Same-step ``save_periodic`` right after would otherwise re-fire and
    orbax would reject the duplicate-step write.
    """
    cfg = CheckpointConfig(wandb_upload="best", save_only_best=True)
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 100, {"mean_error": 0.5})
        assert len(fake.save_args) == 1
        assert ckpt.save_periodic(MagicMock(), 100) is None
        assert len(fake.save_args) == 1


def test_save_final_always_writes_and_uploads_when_enabled(tmp_path):
    cfg = CheckpointConfig(wandb_upload="best")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.save_final(MagicMock(), step=99)
        assert len(fake.save_args) == 1
        assert logger_mock.artifact.call_count == 1
        assert "final" in logger_mock.artifact.call_args.args[0]["aliases"]


def test_save_final_no_upload_when_disabled(tmp_path):
    cfg = CheckpointConfig(wandb_upload="never")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.save_final(MagicMock(), step=99)
        assert len(fake.save_args) == 1
        logger_mock.artifact.assert_not_called()


def test_save_final_forwards_val_metrics_to_orbax(tmp_path):
    """Regression for Copilot #158 review.

    If the final evaluation is the best, orbax must receive its
    metrics so ``best_step`` migrates to the final checkpoint;
    otherwise the ``best`` artifact alias would stay stuck on a
    mid-training checkpoint with worse metrics.
    """
    cfg = CheckpointConfig(wandb_upload="best")
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.5})
        ckpt.save_final(MagicMock(), step=99, val_metrics={"mean_error": 0.2})
        # Final save carries the final-eval metric, not metrics=None.
        assert fake.save_args[-1]["metrics"] == {"mean_error": 0.2}
        # Best now points at the final step.
        assert ckpt.best_step == 99
        assert ckpt.best_metric == pytest.approx(0.2)


def test_save_final_falls_back_to_last_observed(tmp_path):
    """When no val_metrics is passed, save_final reuses the most
    recent validation observation so orbax's best tracker stays
    consistent with the in-memory tracker."""
    cfg = CheckpointConfig(wandb_upload="never")
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 10, {"mean_error": 0.3})
        ckpt.save_final(MagicMock(), step=99)
        assert fake.save_args[-1]["metrics"] == {"mean_error": 0.3}


def test_orbax_rejects_duplicate_step_returns_none(tmp_path):
    """A backward ``save`` that reaches orbax reports None and warns.

    Uses ``save_final`` for the second call so the in-loop same-step
    guard on ``save_periodic`` doesn't short-circuit before orbax.
    """
    cfg = CheckpointConfig()
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        assert ckpt.save_periodic(MagicMock(), step=10) is not None
        # ``save_final`` does not guard same-step so the call reaches
        # the fake and exercises orbax's duplicate-step rejection.
        assert ckpt.save_final(MagicMock(), step=10) is None
        assert len(fake.save_args) == 2
        assert fake.save_args[1]["ok"] is False
        logger_mock.warning.assert_called()


def test_save_periodic_same_step_after_on_validation_silently_skips(tmp_path):
    """Regression for antonioterpin #158 review #4.

    When ``val_interval == save_every`` (a common 100/100 shape),
    ``on_validation`` may have already saved this exact step. Without a
    guard, the follow-up ``save_periodic`` reaches orbax, is rejected
    as a duplicate, and emits a WARNING every coincident interval.
    The guard short-circuits silently.
    """
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        ckpt.on_validation(MagicMock(), 100, {"mean_error": 0.5})
        assert len(fake.save_args) == 1
        # Same step: must short-circuit BEFORE reaching orbax, so the
        # warning path stays quiet.
        assert ckpt.save_periodic(MagicMock(), step=100) is None
        assert len(fake.save_args) == 1
        logger_mock.warning.assert_not_called()


# ---------------------------------------------------------------------------
# Lifecycle: close() / context manager / async safety
# ---------------------------------------------------------------------------


def test_close_waits_then_closes_underlying_manager(tmp_path):
    for ckpt, fake, _ in _make(tmp_path):
        assert fake.wait_calls == 0 and fake.close_calls == 0
        ckpt.close()
        assert fake.wait_calls == 1
        assert fake.close_calls == 1


def test_close_is_idempotent(tmp_path):
    for ckpt, fake, _ in _make(tmp_path):
        ckpt.close()
        ckpt.close()
        assert fake.wait_calls == 1
        assert fake.close_calls == 1


def test_context_manager_closes_on_normal_exit(tmp_path):
    for ckpt, fake, _ in _make(tmp_path):
        with ckpt:
            pass
        assert fake.wait_calls == 1
        assert fake.close_calls == 1


def test_context_manager_closes_on_exception(tmp_path):
    for ckpt, fake, _ in _make(tmp_path):
        with pytest.raises(RuntimeError):
            with ckpt:
                raise RuntimeError("boom")
        assert fake.wait_calls == 1
        assert fake.close_calls == 1


# ---------------------------------------------------------------------------
# Manager construction
# ---------------------------------------------------------------------------


def test_manager_options_reflect_config(tmp_path):
    cfg = CheckpointConfig(keep=7, metric_key="loss", higher_is_better=False)
    for _, fake, _ in _make(tmp_path, cfg=cfg):
        # _FakeManager stashes the options it was constructed with.
        # Best-mode and best_fn semantics propagate.
        assert fake._best_mode == "min"
        # The best_fn pulls our metric_key.
        assert fake._best_fn({"loss": 0.42}) == pytest.approx(0.42)


def test_manager_options_higher_is_better_sets_max(tmp_path):
    cfg = CheckpointConfig(
        wandb_upload="every", metric_key="acc", higher_is_better=True
    )
    for _, fake, _ in _make(tmp_path, cfg=cfg):
        assert fake._best_mode == "max"


def test_retention_keeps_latest_n_and_best(tmp_path):
    """Retention must keep the most-recent N (for resume) *and* the best.

    Regression for the resume-from-latest blocker: driving orbax's
    ``max_to_keep`` off ``best_fn`` alone retains best-N *by metric* and
    evicts the newest checkpoints when the metric degrades. The
    constructor must instead install a composite policy combining
    ``LatestN(keep)`` and ``BestN(1)``. ``BestN.reverse`` puts the
    better metric last so it survives: ``reverse=True`` for
    lower-is-better, ``False`` otherwise.
    """
    from orbax.checkpoint import checkpoint_managers as ocp_cm

    for higher_is_better, expected_reverse in [(False, True), (True, False)]:
        cfg = CheckpointConfig(keep=5, higher_is_better=higher_is_better)
        for _, fake, _ in _make(tmp_path, cfg=cfg):
            policy = fake._options.preservation_policy
            assert isinstance(policy, ocp_cm.AnyPreservationPolicy)
            kinds = {type(p): p for p in policy.policies}
            latest = kinds[ocp_cm.LatestN]
            best = kinds[ocp_cm.BestN]
            assert latest.n == 5
            assert best.n == 1
            assert best.reverse is expected_reverse


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def _install_order_probes(fake, logger_mock):
    """Wire instrumented ``wait_until_finished`` + ``logger.artifact``.

    Returns a list that records the call order ``"wait"``/``"artifact"``
    so tests can assert the barrier fires before the upload.
    """
    call_order: list[str] = []

    def _wait(_fake=fake, _order=call_order):
        _order.append("wait")
        _fake.wait_calls += 1

    def _artifact(*args, _order=call_order, **kwargs):
        _order.append("artifact")

    fake.wait_until_finished = _wait
    logger_mock.artifact.side_effect = _artifact
    return call_order


def test_on_validation_waits_for_save_before_upload(tmp_path):
    """Regression for antonioterpin #158 review #1.

    Saves run async (``enable_async_checkpointing=True``), so
    ``self._mngr.save`` returns before ``<step>/`` is finalized. W&B
    snapshots the directory eagerly via ``logger.artifact``; without a
    barrier we'd upload an incomplete (or not-yet-present) directory.
    The barrier must precede the upload.
    """
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        call_order = _install_order_probes(fake, logger_mock)
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        assert call_order == ["wait", "artifact"]


def test_save_final_waits_for_save_before_upload(tmp_path):
    """Same async-vs-upload race as ``on_validation``; ``save_final``
    must also flush before uploading the ``final`` artifact."""
    cfg = CheckpointConfig(wandb_upload="best")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        call_order = _install_order_probes(fake, logger_mock)
        ckpt.save_final(MagicMock(), step=99)
        assert call_order == ["wait", "artifact"]


def test_save_final_no_upload_skips_wait_barrier(tmp_path):
    """When uploads are disabled, ``save_final`` need not wait — the
    upload barrier is the only reason to block synchronously here.
    """
    cfg = CheckpointConfig(wandb_upload="never")
    for ckpt, fake, _ in _make(tmp_path, cfg=cfg):
        before = fake.wait_calls
        ckpt.save_final(MagicMock(), step=99)
        # Only the eventual close() should call wait_until_finished.
        assert fake.wait_calls == before


def test_upload_failure_does_not_raise(tmp_path):
    cfg = CheckpointConfig(wandb_upload="every")
    for ckpt, fake, logger_mock in _make(tmp_path, cfg=cfg):
        logger_mock.artifact.side_effect = RuntimeError("network down")
        # Should swallow + warn so training continues.
        ckpt.on_validation(MagicMock(), 5, {"mean_error": 0.5})
        assert len(fake.save_args) == 1
        logger_mock.warning.assert_called()
