"""Checkpoint management for KiRL trainers.

HuggingFace Trainer already handles model weights, optimizer, scheduler, and
RNG state. This module adds RL-specific metadata (algo name, algo_config,
step, metrics) written alongside each checkpoint so that training can be
resumed with the correct algorithm and hyperparameters.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Trainer

from ..utils.logging import Log

logger = Log.get_logger(__name__, role="checkpoint")

_METADATA_FILE = "kirl_metadata.json"
_KIRL_VERSION = "0.1.0"

# Registry: algo name → set of required metadata keys beyond the base set.
# Base keys are always written: algo, step, metrics, kirl_version.
_ALGO_EXTRA_KEYS: dict[str, set[str]] = {
    "grpo": {"algo_config", "generation_config"},
    "ppo":  {"algo_config", "generation_config"},
    "dpo":  {"algo_config"},
    "sft":  set(),
}


class CheckpointManager:
    """Save and restore KiRL checkpoints alongside Trainer-managed state.

    Wraps a live ``Trainer`` instance. The Trainer's own ``save_model`` /
    ``_save_checkpoint`` machinery is used for weights and optimizer state;
    this class writes a ``kirl_metadata.json`` file into the same directory
    so that the algorithm, config, and metrics are always co-located with
    the checkpoint they describe.

    Args:
        trainer: The active Trainer (or subclass) instance.
        algo: Algorithm name, must be registered in ``registry.yaml``.
        algo_config: Algorithm hyperparameters dict (e.g. ``GRPOTrainer.algo_config``).
        generation_config: Generation hyperparameters dict, or ``None`` for non-RL.
        metric_for_best: Metric key used to track the best checkpoint.
        greater_is_better: Whether a higher metric value is better.
    """

    def __init__(
        self,
        trainer: Trainer,
        *,
        algo: str,
        algo_config: dict[str, Any],
        generation_config: dict[str, Any] | None = None,
        metric_for_best: str = "reward/mean",
        greater_is_better: bool = True,
    ) -> None:
        import yaml as _yaml
        _registry_path = Path(__file__).parent.parent / "config" / "registry.yaml"
        _reg: dict[str, list[str]] = _yaml.safe_load(_registry_path.read_text(encoding="utf-8"))

        if algo not in _reg.get("checkpoint", []):
            raise ValueError(
                f"Unknown checkpoint algo {algo!r}. "
                f"Registered: {_reg.get('checkpoint', [])}"
            )
        self._trainer = trainer
        self._algo = algo
        self._algo_config = algo_config
        self._generation_config = generation_config
        self._metric_for_best = metric_for_best
        self._greater_is_better = greater_is_better
        self._best_metric: float | None = None
        self._best_path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        step: int,
        metrics: dict[str, float],
    ) -> Path:
        """Save a checkpoint to ``output_dir`` and update best-checkpoint tracking.

        Calls ``trainer.save_model`` for weights + tokenizer, then writes
        ``kirl_metadata.json`` with algo, config, step, and metrics.

        Args:
            output_dir: Directory to write this checkpoint into.
            step: Current global training step.
            metrics: Dict of scalar metrics for this checkpoint.

        Returns:
            Resolved path to the checkpoint directory.
        """
        path = Path(output_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)

        self._trainer.save_model(str(path))

        metadata = self._build_metadata(step, metrics)
        (path / _METADATA_FILE).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        self._update_best(path, metrics)
        logger.info(
            "Checkpoint saved algo=%s step=%d path=%s",
            self._algo, step, path,
            extra={"event": "checkpoint.save", "phase": "checkpoint", "step": step},
        )
        return path

    def load_metadata(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        """Read ``kirl_metadata.json`` from a checkpoint directory.

        Args:
            checkpoint_dir: Path to a checkpoint directory created by ``save``.

        Returns:
            The metadata dict as written by ``save``.

        Raises:
            FileNotFoundError: If the metadata file is absent.
            ValueError: If the file exists but cannot be parsed as JSON.
        """
        meta_path = Path(checkpoint_dir) / _METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No KiRL metadata found at {meta_path}. "
                "This checkpoint may have been created by a different system."
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt metadata file at {meta_path}") from exc
        logger.debug(
            "Loaded metadata algo=%s step=%s from %s",
            meta.get("algo"), meta.get("step"), meta_path,
            extra={"event": "checkpoint.load", "phase": "checkpoint"},
        )
        return meta

    @property
    def best_checkpoint(self) -> Path | None:
        """Path to the best checkpoint seen so far, or ``None`` if none saved yet."""
        return self._best_path

    def copy_best(self, dest: str | Path) -> Path | None:
        """Copy the best checkpoint to ``dest``.

        Args:
            dest: Destination directory (created if absent).

        Returns:
            Resolved destination path, or ``None`` if no best checkpoint exists.
        """
        if self._best_path is None:
            return None
        dest = Path(dest).resolve()
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self._best_path, dest)
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, step: int, metrics: dict[str, float]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "kirl_version": _KIRL_VERSION,
            "algo": self._algo,
            "step": step,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        extra_keys = _ALGO_EXTRA_KEYS.get(self._algo, set())
        if "algo_config" in extra_keys:
            base["algo_config"] = dict(self._algo_config)
        if "generation_config" in extra_keys and self._generation_config is not None:
            base["generation_config"] = dict(self._generation_config)
        return base

    def _update_best(self, path: Path, metrics: dict[str, float]) -> None:
        value = metrics.get(self._metric_for_best)
        if value is None:
            return
        if self._best_metric is None:
            is_better = True
        elif self._greater_is_better:
            is_better = value > self._best_metric
        else:
            is_better = value < self._best_metric
        if is_better:
            self._best_metric = value
            self._best_path = path
            logger.info(
                "New best checkpoint metric=%s value=%.4f path=%s",
                self._metric_for_best, value, path,
                extra={"event": "checkpoint.best_update", "phase": "checkpoint"},
            )
