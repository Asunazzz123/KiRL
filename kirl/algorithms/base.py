"""Shared Trainer infrastructure for supervised and post-training algorithms."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Literal

import torch
from accelerate.utils import gather_object
from transformers import Trainer

from ..utils.logging import Log

_DISTRIBUTED_PLATFORM = {
    "FSDP"
}

_BASIC_TRAINER = {
    "GRPOTrainer",
    "PPOTrainer",
    "DPOTrainer"
}



@dataclass
class RolloutData:
    """
    Rollout 产物对象
    """
    rewards: torch.Tensor
    log_probs: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor = None

    def to(self,device):
        return RolloutData(
            rewards=self.rewards.to(device),
            log_probs=self.log_probs.to(device),
            response_mask=self.response_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            values = self.values.to(device) if self.values is not None else None
        )


class _BaseTrainer(Trainer):
    """Common infrastructure, with the training loop owned by Transformers.

    Subclasses implement their objective in ``compute_loss`` and may override
    ``_prepare_inputs`` or ``training_step`` for rollout-based algorithms. Keep
    optimizer, scheduler, checkpoint and distributed preparation in Trainer;
    models passed here must not already be wrapped by KiRL's FSDPBackend.

    ``compute_loss`` should accept ``**kwargs`` for Transformers compatibility.
    Custom losses that do not consume ``num_items_in_batch`` should set
    ``self.model_accepts_loss_kwargs = False`` after initialization so Trainer
    handles gradient accumulation scaling. Do not scale such losses again.
    """

    _name: ClassVar[str] = "Base"
    _tag_names: ClassVar[tuple[str, ...]] = ("kirl",)
    _paper: ClassVar[dict[str, str]] = {}

    def __init__(self, *args: Any, run_id: str | None = None, probe=None, **kwargs: Any) -> None:
        # Allocate before Trainer initialization so subclass hooks can use it.
        self._metrics: dict[str, dict[str, tuple[float, float]]] = {
            "train": {},
            "eval": {},
        }
        self._evaluating = False
        super().__init__(*args, **kwargs)
        # A supplied Probe remains owned/closed by the caller. Reuse its run ID
        # without installing another handler or generating another experiment ID.
        probe_context = getattr(probe, "logging_context", {})
        self._logging_context = {
            "run_id": run_id if run_id is not None else probe_context.get("run_id", Log.get_context()["run_id"]),
            "role": "trainer",
            "algorithm": self._name.lower(),
            "rank": self.accelerator.process_index,
        }
        self.logger = Log.get_logger(__name__, **self._logging_context)
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(list(self._tag_names))
        self._telemetry()

    def _telemetry(self) -> dict[str, Any]:
        """Return runtime metadata and log it locally on the main process."""
        backend = "none"
        if self.is_deepspeed_enabled:
            backend = "deepspeed"
        elif self.is_fsdp_enabled:
            backend = "fsdp"
        elif self.accelerator.num_processes > 1:
            backend = "ddp"
        metadata = {
            "trainer": type(self).__name__,
            "algorithm": self._name,
            "distributed": backend,
            "world_size": self.accelerator.num_processes,
            "device": self.accelerator.device.type,
        }
        if self.is_world_process_zero():
            self.logger.info(
                "Trainer runtime: %s", metadata,
                extra={"event": "trainer.initialized", "phase": "init", "step": self.state.global_step},
            )
        return metadata

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Attach update context without changing Trainer's loss/backward logic."""
        with Log.bind(**self._logging_context, phase="update", step=self.state.global_step):
            return super().training_step(*args, **kwargs)

    def store_metrics(
        self,
        metrics: Mapping[str, float | torch.Tensor],
        *,
        mode: Literal["train", "eval"] | None = None,
        weight: float = 1.0,
    ) -> None:
        """Buffer local scalar means until the next Trainer logging event.

        Pass the number of contributing examples/tokens as ``weight`` for a
        sample/token-weighted mean. The default averages calls equally. Record
        local values on every rank; do not pass already globally reduced values
        with local weights. Tensor inputs are detached and never retain graphs.
        Evaluation sampler padding is not removed here; callers must exclude
        duplicate examples when computing their metrics and weights.
        """
        mode = mode if mode is not None else ("train" if self.model.training else "eval")
        if mode not in self._metrics:
            raise ValueError("mode must be 'train' or 'eval'")
        if not isinstance(weight, Real) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be a finite positive number")

        # Validate the whole update before mutating the accumulation window.
        scalars = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Metric {name!r} must be scalar")
                value = value.detach().item()
            if not isinstance(value, Real):
                raise TypeError(f"Metric {name!r} must be a real number")
            scalars[name] = float(value)

        for name, value in scalars.items():
            total, count = self._metrics[mode].get(name, (0.0, 0.0))
            self._metrics[mode][name] = (total + value * weight, count + weight)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Keep evaluation metrics separate, including custom metric prefixes."""
        previous = self._evaluating
        self._evaluating = True
        self._metrics["eval"].clear()
        try:
            with Log.bind(**self._logging_context, phase="eval", step=self.state.global_step):
                return super().evaluate(*args, **kwargs)
        finally:
            self._evaluating = previous
            self._metrics["eval"].clear()

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Merge weighted algorithm metrics into Trainer's logging pipeline.

        All ranks must call this in the same order, including ranks with no
        buffered metrics. Object gathering permits different metric keys across
        ranks. Extra arguments pass through to the installed Trainer version.
        """
        mode = "eval" if self._evaluating or any(name.startswith("eval_") for name in logs) else "train"
        totals: dict[str, tuple[float, float]] = {}
        for rank_metrics in gather_object([self._metrics[mode]]):
            for name, (value, weight) in rank_metrics.items():
                total, count = totals.get(name, (0.0, 0.0))
                totals[name] = (total + value, count + weight)

        merged = {}
        for name, (total, count) in totals.items():
            key = f"eval_{name}" if mode == "eval" and not name.startswith("eval_") else name
            merged[key] = total / count
        # Preserve Trainer's loss, timing and other standard metrics on collision.
        merged.update(logs)
        with Log.bind(**self._logging_context, phase="eval" if mode == "eval" else "update",
                      step=self.state.global_step, event=f"{mode}.metrics"):
            super().log(merged, *args, **kwargs)
        self._metrics[mode].clear()

    def model_card(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        tags: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return model-card metadata without writing files or publishing it."""
        extra_tags = [tags] if isinstance(tags, str) else list(tags or ())
        config = getattr(self.model, "config", None)
        return {
            "model_name": model_name or getattr(self, "hub_model_id", None),
            "base_model": getattr(config, "_name_or_path", None),
            "dataset_name": dataset_name,
            "trainer_name": self._name,
            "tags": sorted(set(self._tag_names).union(extra_tags)),
            "paper": dict(self._paper),
        }
