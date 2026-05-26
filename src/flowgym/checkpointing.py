"""Checkpoint orchestration: orbax save, best-metric tracking, W&B upload."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import goggles as gg
import orbax.checkpoint as ocp

from flowgym.make import build_save_args

if TYPE_CHECKING:
    from synthpix.sampler import Sampler

    from flowgym.common.base import Estimator
    from flowgym.common.base.trainable_state import NNEstimatorTrainableState

logger = gg.get_logger(__name__, with_metrics=True)

WandbUploadMode = Literal["never", "best", "every"]

_VALID_UPLOAD_MODES = frozenset({"never", "best", "every"})


@dataclass(frozen=True)
class CheckpointConfig:
    """Per-run checkpoint policy.

    Attributes:
        save_only_best: When True, ``save_periodic`` only saves if the
            most recent validation produced a new best metric.
        wandb_upload: Controls W&B uploads triggered from validation
            events. ``"never"`` (default) skips uploads; ``"best"``
            uploads only when a new best is observed (alias ``best``);
            ``"every"`` uploads after every validation event (alias
            ``latest``, plus ``best`` when applicable).
        metric_key: Key into the validation metrics dict whose value
            drives the best-metric comparison and is persisted to
            orbax's per-step ``metrics`` so ``best_step`` survives a
            process restart.
        higher_is_better: When True, larger metric values are better.
        keep: Maximum number of on-disk checkpoint steps to retain
            (orbax ``max_to_keep``). The best step is exempt from
            eviction (orbax preserves it alongside the most recent N).
        artifact_type: W&B artifact type label.
        artifact_name: W&B artifact name; defaults to
            ``f"{model.__class__.__name__}_checkpoint"``.
    """

    save_only_best: bool = False
    wandb_upload: WandbUploadMode = "never"
    metric_key: str = "mean_error"
    higher_is_better: bool = False
    keep: int = 3
    artifact_type: str = "checkpoint"
    artifact_name: str | None = None

    def __post_init__(self) -> None:
        """Fail fast on typos or invalid values from YAML configs.

        Raises:
            ValueError: If any field carries an unsupported value (e.g.
                ``wandb_upload="evrey"`` from a typo, or ``keep <= 0``).
        """
        if self.wandb_upload not in _VALID_UPLOAD_MODES:
            raise ValueError(
                f"CheckpointConfig.wandb_upload must be one of "
                f"{sorted(_VALID_UPLOAD_MODES)}; got "
                f"{self.wandb_upload!r}."
            )
        if not isinstance(self.keep, int) or self.keep < 1:
            raise ValueError(
                f"CheckpointConfig.keep must be a positive integer; "
                f"got {self.keep!r}."
            )
        if not isinstance(self.metric_key, str) or not self.metric_key:
            raise ValueError(
                "CheckpointConfig.metric_key must be a non-empty string."
            )

    @classmethod
    def from_configs(
        cls, model_config: dict, dataset_config: dict
    ) -> CheckpointConfig:
        """Build a CheckpointConfig from the loaded run configs.

        Reads ``model_config['checkpoint']`` (preferred) and falls back
        to the legacy ``dataset_config['save_only_best']`` flag so
        existing configs keep working unchanged.

        Args:
            model_config: Resolved model configuration.
            dataset_config: Resolved dataset configuration.

        Returns:
            A CheckpointConfig ready to pass to ``train_supervised``.
        """
        spec = dict(model_config.get("checkpoint", {}) or {})
        spec.setdefault(
            "save_only_best", dataset_config.get("save_only_best", False)
        )
        return cls(**spec)


class Checkpointer:
    """Orchestrate orbax checkpoint saves and optional W&B uploads.

    Owns **one** ``ocp.CheckpointManager`` for the lifetime of the
    training run, rooted at ``out_dir/checkpoints/``. Best-step
    tracking is delegated entirely to orbax via ``CheckpointManager
    Options.best_fn`` — orbax persists per-step ``metrics`` on disk so
    ``best_step`` survives a process restart without our own sidecar.

    Saves are async by default; the long-lived manager keeps an orbax
    background thread alive across calls. Callers must invoke
    ``close()`` (or use the instance as a context manager) so the
    in-flight save drains cleanly; without it the last save can be
    lost and orbax's background commit can run into a torn-down
    thread pool at interpreter exit.
    """

    def __init__(
        self,
        out_dir: str | Path,
        model: Estimator,
        sampler: Sampler | None = None,
        config: CheckpointConfig | None = None,
    ) -> None:
        """Create the checkpointer and open a long-lived manager.

        Args:
            out_dir: Root output directory for the run; checkpoints land
                under ``out_dir/checkpoints/<step>/``.
            model: The estimator instance whose state is being saved.
                Used to extract the optimizer config and to derive the
                default W&B artifact name.
            sampler: Optional synthpix sampler whose grain state is
                serialized alongside the model state for exact resume.
            config: Checkpoint policy; defaults to ``CheckpointConfig()``
                (no upload, every-step periodic saves, ``mean_error``
                as the best-metric key).
        """
        self._cfg = config or CheckpointConfig()
        self._model = model
        self._sampler = sampler
        self._ckpt_root = Path(out_dir).resolve() / "checkpoints"
        self._ckpt_root.mkdir(parents=True, exist_ok=True)

        metric_key = self._cfg.metric_key
        higher_is_better = self._cfg.higher_is_better

        def _best_fn(metrics: Mapping[str, Any]) -> float:
            try:
                return float(metrics[metric_key])
            except (KeyError, TypeError, ValueError):
                return float("-inf" if higher_is_better else "inf")

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self._cfg.keep,
            best_fn=_best_fn,
            best_mode="max" if higher_is_better else "min",
            create=True,
            enable_async_checkpointing=True,
        )
        self._mngr = ocp.CheckpointManager(self._ckpt_root, options=options)

        self._artifact_name = (
            self._cfg.artifact_name or f"{type(model).__name__}_checkpoint"
        )
        # Local best-metric tracker. Authoritative for the
        # "is this a new best?" comparison in every mode — including
        # ``wandb_upload="never"``, where validation metrics are
        # observed but no save (and therefore no orbax metrics) happens
        # until the next periodic event. Bootstrap from any persisted
        # best so resumes pick up where they left off.
        self._seen_best: float | None = None
        seeded = self._mngr.best_step()
        if seeded is not None:
            recorded = self._mngr.metrics(seeded)
            if recorded is not None:
                try:
                    self._seen_best = float(recorded[metric_key])
                except (KeyError, TypeError, ValueError):
                    pass
        # Metrics from the most recent validation event. Passed to
        # orbax on the next ``save_periodic`` so the on-disk best
        # tracker stays consistent with the observed best even when
        # the save is deferred to a periodic event.
        self._last_observed: dict[str, float] = {}
        # Latch consumed by ``save_periodic`` when ``save_only_best``
        # is set: ``on_validation`` flips it on a real improvement; the
        # next periodic save consumes it.
        self._has_unsaved_best = False
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def best_step(self) -> int | None:
        """Step of the best checkpoint seen so far (queries orbax)."""
        return self._mngr.best_step()

    @property
    def best_metric(self) -> float | None:
        """Best metric value seen so far across all validations.

        May be more recent than ``best_step`` when validations happen
        without an accompanying save (the typical
        ``wandb_upload="never"`` + ``save_only_best=True`` shape).
        """
        return self._seen_best

    def on_validation(
        self,
        state: NNEstimatorTrainableState,
        step: int,
        val_metrics: Mapping[str, Any],
    ) -> str | None:
        """Record a validation event and act per ``wandb_upload`` mode.

        Args:
            state: The trainable state to checkpoint, if needed.
            step: Training step at which validation ran.
            val_metrics: Validation metrics produced by
                ``evaluate_batches``.

        Returns:
            The on-disk checkpoint directory path when a save happened,
            ``None`` when validation was purely observational (no
            extractable metric, or current mode does not save at
            validation time).
        """
        metric_value = self._extract_metric(val_metrics)
        if metric_value is None:
            return None
        is_new_best = self._is_better(metric_value)
        self._last_observed = {self._cfg.metric_key: metric_value}
        if is_new_best:
            self._seen_best = metric_value
            self._has_unsaved_best = True
        mode = self._cfg.wandb_upload

        will_save = mode == "every" or (mode == "best" and is_new_best)
        if not will_save:
            return None

        path = self._save(state, step, self._last_observed)
        if path is None:
            return None
        self._has_unsaved_best = False
        # Orbax may have moved the best step now; re-confirm.
        is_best_now = self._mngr.best_step() == step
        aliases = ["best"] if is_best_now else ["latest"]
        if is_best_now and mode == "every":
            aliases.append("latest")
        # Saves run async (``enable_async_checkpointing=True``), so
        # ``self._mngr.save`` returns before the ``<step>/`` directory is
        # finalized — only the ``<step>.orbax-checkpoint-tmp-*`` staging
        # dir exists until orbax's background thread commits. W&B
        # snapshots ``path`` eagerly via ``logger.artifact``; without a
        # barrier here we would upload an incomplete or not-yet-present
        # directory. Flush before handing the path on.
        self._mngr.wait_until_finished()
        self._upload(path, step, aliases=aliases)
        return path

    def save_periodic(
        self, state: NNEstimatorTrainableState, step: int
    ) -> str | None:
        """Disk-only save at a periodic ``save_every`` event.

        Honors ``save_only_best``: skips when no unsaved improvement is
        pending. Does not upload (W&B uploads are tied to validation
        events via ``on_validation``).

        Args:
            state: The trainable state to checkpoint.
            step: Current training step.

        Returns:
            The on-disk checkpoint path when a save happened, otherwise
            ``None``.
        """
        if self._cfg.save_only_best and not self._has_unsaved_best:
            return None
        # Same-step guard: when ``val_interval == save_every`` (a common
        # 100/100 shape), ``on_validation`` may have already saved this
        # exact step. Orbax would silently reject the duplicate write
        # and our wrapper would log a WARNING every coincident
        # interval. Skip the no-op here instead of generating noise.
        latest = self._mngr.latest_step()
        if latest is not None and step <= latest:
            return None
        self._has_unsaved_best = False
        return self._save(state, step, self._last_observed or None)

    def save_final(
        self,
        state: NNEstimatorTrainableState,
        step: int,
        val_metrics: Mapping[str, Any] | None = None,
    ) -> str | None:
        """End-of-training save. Always writes to disk.

        Uploads to W&B with the ``final`` alias when ``wandb_upload``
        is not ``"never"`` so post-mortem analyses always have an
        end-of-run snapshot to download from the run page.

        When ``val_metrics`` is provided, the configured ``metric_key``
        is extracted and handed to orbax so a best-on-final-validation
        outcome correctly migrates ``best_step`` to the final
        checkpoint (otherwise the ``best`` alias would stay on the
        previous best even when the final eval is the genuine best).

        Args:
            state: The final trainable state to checkpoint.
            step: Final training step.
            val_metrics: Validation metrics from the final evaluation,
                if available. ``None`` falls back to the most recent
                observed validation metrics.

        Returns:
            The on-disk checkpoint path, or ``None`` when orbax rejected
            the save (duplicate step / backward step).
        """
        metrics: Mapping[str, float] | None
        if val_metrics is not None:
            metric_value = self._extract_metric(val_metrics)
            if metric_value is not None:
                metrics = {self._cfg.metric_key: metric_value}
                # Keep the in-memory tracker in sync so a subsequent
                # query of ``best_metric`` reports the latest.
                if self._is_better(metric_value):
                    self._seen_best = metric_value
            else:
                metrics = self._last_observed or None
        else:
            metrics = self._last_observed or None
        path = self._save(state, step, metrics)
        if path is None:
            return None
        if self._cfg.wandb_upload != "never":
            # See ``on_validation``: async saves must finalize before
            # W&B snapshots the directory.
            self._mngr.wait_until_finished()
            self._upload(path, step, aliases=["final"])
        return path

    def close(self) -> None:
        """Flush async saves and tear down the manager.

        Idempotent. **Must** be called before the process exits or the
        last in-flight save can be lost and the interpreter can raise
        on shutdown (orbax tries to schedule work onto a dying thread
        pool). Use the Checkpointer as a context manager to make this
        automatic.
        """
        if self._closed:
            return
        try:
            self._mngr.wait_until_finished()
        finally:
            self._mngr.close()
            self._closed = True

    def __enter__(self) -> Checkpointer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save(
        self,
        state: NNEstimatorTrainableState,
        step: int,
        metrics: Mapping[str, float] | None,
    ) -> str | None:
        args = build_save_args(
            state, step, model=self._model, sampler=self._sampler
        )
        saved = self._mngr.save(step=step, args=args, metrics=metrics)
        if not saved:
            logger.warning(
                f"Checkpoint save at step {step} was skipped by orbax "
                "(duplicate or backward step)."
            )
            return None
        return str(self._ckpt_root / str(step))

    def _extract_metric(self, val_metrics: Mapping[str, Any]) -> float | None:
        if self._cfg.metric_key not in val_metrics:
            return None
        try:
            v = float(val_metrics[self._cfg.metric_key])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return v

    def _is_better(self, value: float) -> bool:
        current = self.best_metric
        if current is None:
            return True
        return (
            value > current if self._cfg.higher_is_better else value < current
        )

    def _upload(self, ckpt_path: str, step: int, *, aliases: list[str]) -> None:
        if not hasattr(logger, "artifact"):
            return
        try:
            logger.artifact(
                {
                    "path": ckpt_path,
                    "name": self._artifact_name,
                    "type": self._cfg.artifact_type,
                    "aliases": aliases,
                },
                step=step,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to upload checkpoint to W&B at step {step}: {exc}"
            )
