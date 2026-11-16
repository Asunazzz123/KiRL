"""Shared Trainer infrastructure for supervised and post-training algorithms."""
from __future__ import annotations

import torch
import math

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Literal, TypeAlias
from accelerate.utils import gather_object
from transformers import (
    Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM,
    AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
)

from ..utils.logging import Log
from ..datasets import BaseDataset
from ..profiler import NullProbe, Probe

RewardFunc: TypeAlias = Callable[..., "list[float | None]"]


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


class RolloutWeightSyncCallback(TrainerCallback):
    """
    Sync policy weights to the rollout backend after each optimizer update.

    Runs at Trainer's step-end boundary, after gradient accumulation and the
    optimizer step. Skips synchronization when the accelerator reports that
    the optimizer update was skipped, and records the transfer under the
    ``weight_sync`` profiling stage.

    All participating ranks must enter this callback when policy parameter
    gathering requires distributed collectives, such as FSDP all-gather.

    Args:
        trainer: Trainer providing the accelerator, profiling probe, rollout
            backend, and policy parameter iterator. Register an instance with
            ``trainer.add_callback(RolloutWeightSyncCallback(trainer))``.
    """
    def __init__(self,trainer:_BaseTrainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer.accelerator.optimizer_step_was_skipped:
            return control

        with self.trainer._probe.stage("weight_sync"):
            self.trainer._generation_backend.update_weight(
                self.trainer._iter_policy_params()
            )
        return control



class _BaseTrainer(Trainer):
    """
    Common infrastructure, with the training loop owned by Transformers.
    """

    _name: ClassVar[str] = "Base"
    _tag_names: ClassVar[tuple[str, ...]] = ("kirl",)
    _paper: ClassVar[dict[str, str]] = {}
    _cls_logger: ClassVar[Any] = None

    @classmethod
    def _get_cls_logger(cls):
        """Lazy per-class logger for classmethods (e.g. ``_validate_config``).

        Picks up ``cls._name`` dynamically so each subclass gets its own
        algorithm tag without hardcoding. Cached on the subclass after first
        call; instance logger (``self.logger``) supersedes this once ``__init__``
        runs because it carries ``run_id`` and ``rank`` too.
        """
        if cls._cls_logger is None:
            cls._cls_logger = Log.get_logger(
                __name__, algorithm=cls._name.lower(), role="trainer"
            )
        return cls._cls_logger

    def __init__(self, *args: Any, run_id: str | None = None, probe: Probe | NullProbe | None = None , **kwargs: Any) -> None:
        # Allocate before Trainer initialization so subclass hooks can use it.
        self._metrics: dict[str, dict[str, tuple[float, float]]] = {
            "train": {},
            "eval": {},
        }
        # Probe Hook
        self._probe: Probe | NullProbe = (
            probe if probe is not None else NullProbe()
        )
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
        self.reference_model: PreTrainedModel | None = None
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

    def _prepare_reference_policy(self) -> None:
        """Create a frozen copy of self.model for KL / implicit-reward computation.

        Non-distributed and DDP: deepcopy the unwrapped model, freeze parameters,
        and move to the accelerator device.

        FSDP / DeepSpeed: deepcopy is unsafe on sharded flat params; raises
        NotImplementedError until the distributed path is implemented.

        Callers that do not need a reference model (e.g. beta=0) should
        simply skip calling this method; self.reference_model stays None.
        """
        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise NotImplementedError(
                "Reference policy under FSDP/DeepSpeed is not yet implemented. "
                "Override _prepare_reference_policy_sharded, or set beta=0 to skip."
            )
        import copy
        # Unwrap to get the base PreTrainedModel before accelerator wrapping.
        unwrapped = self.accelerator.unwrap_model(self.model)
        ref = copy.deepcopy(unwrapped)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        ref = ref.to(self.accelerator.device)
        self.reference_model = ref
        self.logger.info(
            "Reference policy ready (deepcopy, frozen)",
            extra={"event": "reference_policy.ready", "phase": "init", "step": 0},
        )

    def _iter_policy_params(self):
        """Yield (name, tensor) pairs of the full, named policy parameters.

        Under FSDP, summons complete params across ranks before yielding and
        releases the all-gather on exit. Under DDP and single-GPU the unwrapped
        model's named_parameters() already holds complete weights.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.is_fsdp_enabled:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with FSDP.summon_full_params(unwrapped, writeback=False):
                yield from unwrapped.named_parameters()
        else:
            yield from unwrapped.named_parameters()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Attach update context without changing Trainer's loss/backward logic."""
        with Log.bind(**self._logging_context, phase="update", step=self.state.global_step):
            with self._probe.stage("train.microstep"):
                return super().training_step(*args, **kwargs)

    def store_metrics(
        self,
        metrics: Mapping[str, float | torch.Tensor],
        *,
        mode: Literal["train", "eval"] | None = None,
        weight: float = 1.0,
    ) -> None:
        """
        Buffer local scalar means until the next Trainer logging event.
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
        """
        Merge weighted algorithm metrics into Trainer's logging pipeline.
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
            for name, value in merged.items():
                if type(value) in (int, float):
                    self._probe.metric(name, value, labels={"mode": mode})
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

    @staticmethod
    def _resolve_model(
            model: str | PreTrainedModel,
            *,
            model_init_kwargs: dict | None = None
        ) -> PreTrainedModel:
            """
            Resolve
            """
            if isinstance(model,str):
                assert model.strip(), "model 字段不能为空字符串!"
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    **(model_init_kwargs or {})
                )
            elif isinstance(model,PreTrainedModel):
                if model_init_kwargs is not None:
                    raise ValueError("传入模型不能再设置model_init_kwargs")
            else:
                raise ValueError("model 必须是字符串或者PreTrainedModel实例")
            return model

    @classmethod
    def _validate_config(
        cls, args: TrainingArguments | None = None,
    ) -> TrainingArguments | None:
        """Validate common Trainer arguments without initializing runtime state."""
        if args is not None and not isinstance(args, TrainingArguments):
            raise TypeError("args must be TrainingArguments or None")
        return args

    @staticmethod
    def _require_mapping(name: str, value: Any) -> None:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping or None")

    @staticmethod
    def _require_int(name: str, value: Any, minimum: int) -> int:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    @staticmethod
    def _require_number(name: str, value: Any) -> Real:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _normalize_rewards(
        reward_fn: "RewardFunc | list[RewardFunc] | None",
    ) -> "list[RewardFunc]":
        """Normalize reward_fn input to a non-empty list of callables."""
        if reward_fn is None:
            raise ValueError("reward_fn is required; pass at least one reward function")
        funcs: list[RewardFunc] = [reward_fn] if callable(reward_fn) else list(reward_fn)
        if not funcs:
            raise ValueError("reward_fn must contain at least one function")
        for i, fn in enumerate(funcs):
            if not callable(fn):
                raise TypeError(f"reward_fn[{i}] must be callable")
        return funcs

    @classmethod
    def _validate_generation_config(cls, config: Mapping[str, Any]) -> None:
        """
        Validate common text generation fields, when explicitly requested.
        """
        if not isinstance(config, Mapping):
            raise TypeError("generation_config must be a non-None mapping with resolved defaults")
        cls._require_int("max_new_tokens", config["max_new_tokens"], 1)
        if cls._require_number("temperature", config["temperature"]) <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 < cls._require_number("top_p", config["top_p"]) <= 1:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if type(config["do_sample"]) is not bool:
            raise TypeError("do_sample must be a boolean")


    @classmethod
    def _resolve_tokenizer(
        cls,
        model: str | PreTrainedModel,
        processing_class: PreTrainedTokenizerBase | None = None,
        *,
        tokenizer_init_kwargs: Mapping[str, Any] | None = None,
        padding_side: str | None = None,
        require_padding: bool = False,
    ) -> PreTrainedTokenizerBase:
        """Resolve a text tokenizer; explicit padding settings mutate it in place.

        Loading options are independent of model loading options. A requested
        padding side takes precedence over tokenizer_init_kwargs. Reusing EOS
        for padding requires callers to preserve the tokenizer's attention mask.
        """
        cls._require_mapping("tokenizer_init_kwargs", tokenizer_init_kwargs)
        if padding_side not in (None, "left", "right"):
            raise ValueError("padding_side must be 'left', 'right' or None")
        if type(require_padding) is not bool:
            raise TypeError("require_padding must be a boolean")

        if processing_class is not None:
            if not isinstance(processing_class, PreTrainedTokenizerBase):
                raise TypeError("processing_class must be a text PreTrainedTokenizerBase")
            if tokenizer_init_kwargs is not None:
                raise ValueError("An instantiated tokenizer cannot use tokenizer_init_kwargs")
            tokenizer = processing_class
        else:
            if isinstance(model, str):
                source = model
            elif isinstance(model, PreTrainedModel):
                source = getattr(model.config, "_name_or_path", None)
            else:
                raise TypeError("model must be a string or PreTrainedModel")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Cannot infer tokenizer source; pass processing_class explicitly")
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                **dict(tokenizer_init_kwargs if tokenizer_init_kwargs is not None else {}),
            )

        if padding_side is not None:
            tokenizer.padding_side = padding_side
        if require_padding and tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires a pad token or an EOS token to reuse")
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                raise ValueError("Tokenizer could not resolve pad_token_id from EOS")
        return tokenizer

    @staticmethod
    def _validate_dataset(dataset: Any) -> None:
        """Raise TypeError if dataset is neither a BaseDataset nor None."""
        if dataset is not None and not isinstance(dataset, BaseDataset):
            raise TypeError(
                f"train_dataset must be a BaseDataset instance or None, "
                f"got {type(dataset).__name__!r}"
            )
