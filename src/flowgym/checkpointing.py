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
from orbax.checkpoint import checkpoint_managers as ocp_cm

from flowgym.make import build_save_args, load_model

if TYPE_CHECKING:
    from synthpix.sampler import Sampler

    from flowgym.common.base import Estimator
    from flowgym.common.base.trainable_state import (
        EstimatorTrainableState,
        NNEstimatorTrainableState,
    )

logger = gg.get_logger(__name__, with_metrics=True)

WandbUploadMode = Literal["never", "best", "every"]

_VALID_UPLOAD_MODES = frozenset({"never", "best", "every"})


@dataclass(frozen=True)
class CheckpointConfig:
    """Per-run checkpoint policy.

    Attributes:
        save_only_best: When True, ``save_periodic`` is a no-op, so the
            only checkpoints are the validation-time saves of new bests
            (plus the final one).
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
        keep: Number of most-recent on-disk checkpoint steps to retain
            for resume. The best step is preserved on top of these
            latest N (see ``Checkpointer`` for the orbax retention
            policy), so resuming from the newest checkpoint always
            works even when the metric degrades over training.
        artifact_type: W&B artifact type label.
        artifact_name: W&B artifact name; defaults to
            ``f"{model.__class__.__name__}_checkpoint"``.
        hf_repo: Optional ``<org>/<name>`` Hugging Face model repo. When
            set, ``save_final`` pushes the best checkpoint there (one
            commit per run); off by default, for maintainers who publish
            models. The repo is created private if it does not exist.
        hf_path: Folder inside ``hf_repo``; defaults to ``artifact_name``.
    """

    save_only_best: bool = False
    wandb_upload: WandbUploadMode = "never"
    metric_key: str = "mean_error"
    higher_is_better: bool = False
    keep: int = 3
    artifact_type: str = "checkpoint"
    artifact_name: str | None = None
    hf_repo: str | None = None
    hf_path: str | None = None

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
        if self.hf_repo is not None and self.hf_repo.count("/") != 1:
            raise ValueError(
                "CheckpointConfig.hf_repo must be '<org>/<name>'; got "
                f"{self.hf_repo!r}."
            )
        if self.hf_path is not None and self.hf_repo is None:
            raise ValueError("CheckpointConfig.hf_path needs hf_repo.")
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


def _upload_to_hub(repo: str, path_in_repo: str, folder: Path) -> None:
    """Push ``folder`` to ``path_in_repo`` of a Hub model repo, in one commit.

    Args:
        repo: ``<org>/<name>`` model repo; created private if missing.
        path_in_repo: Destination folder inside the repo.
        folder: Local checkpoint directory.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=f"Add {path_in_repo}",
    )


class Checkpointer:
    """Orchestrate orbax checkpoint saves and optional W&B uploads.

    Owns **one** ``ocp.CheckpointManager`` for the lifetime of the
    training run, rooted at ``out_dir/checkpoints/``. Best-step
    tracking is delegated to orbax via ``CheckpointManagerOptions
    .best_fn`` — orbax persists per-step ``metrics`` on disk so
    ``best_step`` survives a process restart without our own sidecar.

    Retention combines two orbax preservation policies so the two
    consumers never starve each other: ``LatestN(keep)`` keeps the
    most recent ``keep`` steps for *resume*, and ``BestN(1)`` keeps the
    single best-by-metric step for the ``best`` W&B alias. Driving
    ``max_to_keep`` off ``best_fn`` alone (orbax's default when only
    ``best_fn`` is set) would retain the best-N *by metric* and evict
    the newest checkpoints whenever validation degrades — silently
    breaking resume-from-latest — so the explicit composite policy is
    required.

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

        # Retain the most recent ``keep`` steps (for resume) *and* the
        # single best step (for the ``best`` alias). ``best_fn`` alone
        # would make orbax retain the best-N by metric and evict the
        # newest checkpoints when validation degrades, breaking
        # resume-from-latest; the explicit composite policy keeps both.
        # ``reverse`` puts the better metric last so ``BestN`` keeps it:
        # ascending sort (``reverse=False``) for higher-is-better,
        # descending (``reverse=True``) for lower-is-better.
        preservation_policy = ocp_cm.AnyPreservationPolicy(
            [
                ocp_cm.LatestN(n=self._cfg.keep),
                ocp_cm.BestN(
                    get_metric_fn=_best_fn,
                    reverse=not higher_is_better,
                    n=1,
                    keep_checkpoints_without_metrics=False,
                ),
            ]
        )
        options = ocp.CheckpointManagerOptions(
            best_fn=_best_fn,
            best_mode="max" if higher_is_better else "min",
            preservation_policy=preservation_policy,
            create=True,
            enable_async_checkpointing=True,
        )
        self._mngr = ocp.CheckpointManager(self._ckpt_root, options=options)

        self._artifact_name = (
            self._cfg.artifact_name or f"{type(model).__name__}_checkpoint"
        )
        # Local best-metric tracker for the "is this a new best?"
        # comparison. Bootstrap from any persisted best so resumes pick
        # up where they left off.
        self._seen_best: float | None = None
        seeded = self._mngr.best_step()
        if seeded is not None:
            recorded = self._mngr.metrics(seeded)
            if recorded is not None:
                try:
                    self._seen_best = float(recorded[metric_key])
                except (KeyError, TypeError, ValueError):
                    pass
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def best_step(self) -> int | None:
        """Step of the best checkpoint seen so far (queries orbax)."""
        return self._mngr.best_step()

    def restore_best(
        self, template: NNEstimatorTrainableState
    ) -> EstimatorTrainableState | None:
        """Load the best checkpoint's parameters into ``template``.

        Args:
            template: State with the target structure (e.g. the current one).

        Returns:
            The restored state, or ``None`` if no checkpoint has a metric.
        """
        step = self.best_step
        if step is None:
            return None
        self._mngr.wait_until_finished()
        return load_model(self._ckpt_root / str(step), template)

    @property
    def best_metric(self) -> float | None:
        """Best metric value seen so far across all validations."""
        return self._seen_best

    def on_validation(
        self,
        state: NNEstimatorTrainableState,
        step: int,
        val_metrics: Mapping[str, Any],
    ) -> str | None:
        """Record a validation event; save on a new best or in ``every`` mode.

        A new best is always saved to disk at this step, so the metric
        orbax records belongs to the exact state that was validated
        (``restore_best`` relies on this). Uploads follow
        ``wandb_upload``.

        Args:
            state: The trainable state to checkpoint, if needed.
            step: Training step at which validation ran.
            val_metrics: Validation metrics produced by
                ``evaluate_batches``.

        Returns:
            The on-disk checkpoint directory path when a save happened,
            ``None`` when validation was purely observational (no
            improvement / no extractable metric outside ``every`` mode).
        """
        metric_value = self._extract_metric(val_metrics)
        mode = self._cfg.wandb_upload
        is_new_best = metric_value is not None and self._is_better(metric_value)

        # ``every`` snapshots the latest model after *every* validation,
        # including one whose metric is missing or non-finite (e.g. a
        # NaN-out near the end) — that run state is exactly what a
        # post-mortem wants.
        if not (is_new_best or mode == "every"):
            return None

        # Attach only *this* event's metric to the save. A non-finite
        # event saves with no metrics (in ``every`` mode) so it can
        # never be mistaken for a best checkpoint.
        current_metrics = (
            {self._cfg.metric_key: metric_value}
            if metric_value is not None
            else None
        )
        path = self._save(state, step, current_metrics)
        if path is None:
            # Orbax rejected the write (duplicate / backward step), so
            # nothing landed on disk; don't advance the best tracker.
            return None
        if is_new_best:
            self._seen_best = metric_value
        if mode == "never":
            return path
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
        """Disk-only save at a periodic ``save_every`` event, for resume.

        Saves without metrics, since this state was not validated (best
        checkpoints are written by ``on_validation``). A no-op under
        ``save_only_best``. Does not upload.

        Args:
            state: The trainable state to checkpoint.
            step: Current training step.

        Returns:
            The on-disk checkpoint path when a save happened, otherwise
            ``None``.
        """
        if self._cfg.save_only_best:
            return None
        # Same-step guard: when ``val_interval == save_every`` (a common
        # 100/100 shape), ``on_validation`` may have already saved this
        # exact step. Orbax would silently reject the duplicate write
        # and our wrapper would log a WARNING every coincident
        # interval. Skip the no-op here instead of generating noise.
        latest = self._mngr.latest_step()
        if latest is not None and step <= latest:
            return None
        return self._save(state, step, None)

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
                if available. Without them the save carries no metrics.

        Returns:
            The on-disk checkpoint path (the freshly written step, or an
            already-persisted one at the same step), or ``None`` when
            orbax rejected a genuinely backward save.
        """
        metrics: Mapping[str, float] | None = None
        if val_metrics is not None:
            metric_value = self._extract_metric(val_metrics)
            if metric_value is not None:
                metrics = {self._cfg.metric_key: metric_value}
                # Keep the in-memory tracker in sync so a subsequent
                # query of ``best_metric`` reports the latest.
                if self._is_better(metric_value):
                    self._seen_best = metric_value
        # The final step may already be on disk when training ends on a
        # ``save_every`` boundary that ``save_periodic`` just wrote.
        # Orbax would reject the duplicate write (``_save`` -> ``None``),
        # dropping the ``final`` alias upload and emitting a spurious
        # warning. Reuse the existing checkpoint instead so the final
        # artifact is always tagged regardless of ``num_episodes`` /
        # ``save_every`` alignment.
        if step in self._mngr.all_steps():
            path: str | None = str(self._ckpt_root / str(step))
        else:
            path = self._save(state, step, metrics)
        if path is None:
            return None
        if self._cfg.wandb_upload != "never":
            # See ``on_validation``: async saves must finalize before
            # W&B snapshots the directory.
            self._mngr.wait_until_finished()
            self._upload(path, step, aliases=["final"])
        if self._cfg.hf_repo is not None:
            self._mngr.wait_until_finished()
            self._push_best_to_hub(step)
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

    def _push_best_to_hub(self, final_step: int) -> None:
        """Publish the best checkpoint (the final one if none is scored).

        Args:
            final_step: Step of the end-of-training save.
        """
        assert self._cfg.hf_repo is not None
        step = self._mngr.best_step()
        if step is None:
            step = final_step
        try:
            _upload_to_hub(
                self._cfg.hf_repo,
                self._cfg.hf_path or self._artifact_name,
                self._ckpt_root / str(step),
            )
        except Exception as exc:
            logger.warning(
                f"Failed to push checkpoint {step} to the Hub: {exc}"
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
