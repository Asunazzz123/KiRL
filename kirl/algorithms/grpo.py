"""GRPO algorithm implementation."""

import warnings

from collections.abc import Mapping
from typing import Any
from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase
from .base import _BaseTrainer, RewardFunc
from ..engine.inference.base import GenerationBackend, GenerationConfig

from ..datasets import BaseDataset
from ..utils.logging import Log
logger = Log.get_logger(__name__, algorithm="grpo", role="trainer")





class GRPOTrainer(_BaseTrainer):
    _ALGO_DEFAULTS = {
        "group_size": 4,
        "beta": 0.1,
        "cliprange": 0.2,
        "loss_normalization": "token",
        "advantage_normalization": "group",
    }
    _GENERATION_DEFAULTS = {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_new_tokens": 256,
        "do_sample": True,
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_fn: RewardFunc | list[RewardFunc] | None = None,
        train_dataset: BaseDataset | None = None,
        eval_dataset: BaseDataset | None = None,
        *,
        args: TrainingArguments | None = None,
        generation_backend: GenerationBackend | None = None,
        algo_config: Mapping[str, Any] | None = None,
        generation_config: Mapping[str, Any] | None = None,
        model_init_kwargs: dict[str,Any] | None = None,
        tokenizer_init_kwargs: Mapping[str, Any] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        data_collator=None
    ):
        args = self._validate_config(args, algo_config, generation_config)
        self.algo_config = {**self._ALGO_DEFAULTS, **(algo_config if algo_config is not None else {})}
        self.generation_config = {
            **self._GENERATION_DEFAULTS,
            **(generation_config if generation_config is not None else {}),
        }
        model_source = model
        model = self._resolve_model(
            model,
            model_init_kwargs=model_init_kwargs
        )
        tokenizer = self._resolve_tokenizer(
            model_source,
            processing_class=processing_class,
            tokenizer_init_kwargs=tokenizer_init_kwargs,
            padding_side="left",
            require_padding=True,
        )
        self.reward_funcs = self._normalize_rewards(reward_fn)
        self._validate_dataset(train_dataset)
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator
        )
        self._prepare_reference_policy()
        self._setup_generation(backend=generation_backend)
        self._initialize_rollout_state()


    def _setup_generation(
        self,
        backend: GenerationBackend | None = None
    ) -> None :
        gc = self.generation_config
        self._generation_config = GenerationConfig(
            num_return_sequences=self.algo_config["group_size"],
            max_new_tokens=gc["max_new_tokens"],
            do_sample=gc["do_sample"],
            temperature=gc["temperature"],
            top_p=gc["top_p"],
            return_logprobs=True,
        )
        self._generation_backend: GenerationBackend | None = backend

    @classmethod
    def _validate_config(
        cls,
        args: TrainingArguments | None = None,
        algo_config: Mapping[str, Any] | None = None,
        generation_config: Mapping[str, Any] | None = None,
    ) -> TrainingArguments | None:
        """
        Validate configuration before loading weights, preserve Trainer args.
        """
        args = super()._validate_config(args)
        cls._require_mapping("algo_config", algo_config)
        cls._require_mapping("generation_config", generation_config)

        algo = {**cls._ALGO_DEFAULTS, **(algo_config if algo_config is not None else {})}
        generation = {
            **cls._GENERATION_DEFAULTS,
            **(generation_config if generation_config is not None else {}),
        }
        if "temperature" in algo:
            raise ValueError("Set temperature only in generation_config, not algo_config")
        if "name" in algo and algo["name"] != "grpo":
            raise ValueError("algo_config.name must be 'grpo'")

        cls._validate_generation_config(generation)

        # A single response cannot provide a useful within-group baseline.
        cls._require_int("group_size", algo["group_size"], 2)
        if cls._require_number("beta", algo["beta"]) < 0:
            raise ValueError("beta must be >= 0")
        # Match compute_grpo_policy_loss's current clip_eps contract.
        if not 0 <= cls._require_number("cliprange", algo["cliprange"]) < 1:
            raise ValueError("cliprange must satisfy 0 <= cliprange < 1")
        if not generation["do_sample"]:
            warnings.warn(
                "do_sample=False can produce identical responses within a GRPO group, "
                "leaving no reward differences to learn from. Sampling parameters "
                "temperature and top_p may be ignored by the generation backend.",
                UserWarning,
                stacklevel=2,
            )
        # Both reductions exist in core_algo; Trainer wiring remains unfinished.
        if algo["loss_normalization"] not in ("token", "sequence"):
            raise ValueError("loss_normalization must be 'token' or 'sequence'")
        if algo["advantage_normalization"] != "group":
            raise NotImplementedError(
                f"advantage_normalization={algo['advantage_normalization']!r} is not yet supported"
            )
        return args

    

    def grpo_loss():
        ...