"""Supervised Fine-Tuning trainer."""

from collections.abc import Mapping
from typing import Any

import torch
from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase

from .base import _BaseTrainer
from ..datasets import BaseDataset
from ..utils.logging import Log
from ..profiler.probe import Probe, NullProbe

class SFTTrainer(_BaseTrainer):
    """Supervised fine-tuning trainer built on ``_BaseTrainer``.

    Uses the HuggingFace model's built-in cross-entropy loss (via
    ``outputs.loss``) over labelled token sequences.  No rollout, no
    reference policy, no reward functions are required.
    """

    _name: str = "SFT"
    _ALGO_DEFAULTS: dict[str, Any] = {
        "max_seq_length": 2048,
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        train_dataset: BaseDataset | None = None,
        eval_dataset: BaseDataset | None = None,
        *,
        probe: Probe | NullProbe | None = None,
        args: TrainingArguments | None = None,
        algo_config: Mapping[str, Any] | None = None,
        model_init_kwargs: dict[str, Any] | None = None,
        tokenizer_init_kwargs: Mapping[str, Any] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        optimizer_cls_and_kwargs=None,
        data_collator=None,
    ) -> None:
        args = self._validate_config(args, algo_config)
        self.algo_config = {**self._ALGO_DEFAULTS, **(algo_config or {})}
        model_source = model
        model = self._resolve_model(model, model_init_kwargs=model_init_kwargs)
        tokenizer = self._resolve_tokenizer(
            model_source,
            processing_class=processing_class,
            tokenizer_init_kwargs=tokenizer_init_kwargs,
            # Right-padding for SFT: labels align with right-padded input_ids.
            padding_side="right",
            require_padding=True,
        )
        self._validate_dataset(train_dataset)
        if data_collator is None:
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,   # FSDP rank pad
            )
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            **({"optimizer_cls_and_kwargs": optimizer_cls_and_kwargs} if optimizer_cls_and_kwargs else {}),
            probe=probe
        )

    @classmethod
    def _validate_config(
        cls,
        args: TrainingArguments | None = None,
        algo_config: Mapping[str, Any] | None = None,
    ) -> TrainingArguments | None:
        """Validate SFT-specific config before loading weights.

        Args:
            args: HuggingFace ``TrainingArguments``, or ``None``.
            algo_config: Optional SFT hyperparameters mapping.

        Returns:
            Validated ``args``, passed through unchanged.
        """
        args = super()._validate_config(args)
        cls._require_mapping("algo_config", algo_config)
        if algo_config is not None:
            merged = {**cls._ALGO_DEFAULTS, **algo_config}
            cls._require_int("max_seq_length", merged["max_seq_length"], 1)
        return args

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Compute cross-entropy loss over labelled token sequences.

        Expects ``inputs`` to contain ``input_ids``, ``attention_mask``, and
        ``labels`` (padding positions set to ``-100`` so they are ignored by
        ``cross_entropy``). The HuggingFace model computes the loss internally
        when ``labels`` are passed.

        Args:
            model: The active (possibly wrapped) policy model.
            inputs: Collated batch dict with ``input_ids``, ``attention_mask``,
                and ``labels``.
            return_outputs: Unused; kept for Trainer API compatibility.

        Returns:
            Scalar loss tensor with gradient attached.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        self.store_metrics({"sft/loss": loss})
        return loss
