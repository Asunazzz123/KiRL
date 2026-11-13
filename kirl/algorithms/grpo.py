"""GRPO algorithm implementation."""
import torch
import warnings

from collections.abc import Mapping
from typing import Any
from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase

from .core_algo import compute_grpo_policy_loss
from .base import _BaseTrainer, RewardFunc, RolloutData
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

    def _initialize_rollout_state(self) -> None:
        """Set up per-step and cross-step rollout buffers used by the RL loop.

        Validates that a generation backend is available and pre-allocates the
        containers that ``compute_loss`` fills each step:

        - ``self._current_rollout``: the in-flight ``RolloutData`` for the
          current training step, replaced each rollout.
        - ``self._advantage_stats``: running mean/std of rewards across steps,
          reserved for future advantage normalization variants. Unused while
          ``advantage_normalization == "group"`` but kept as a stable hook.

        Raises:
            RuntimeError: If no ``generation_backend`` was provided. GRPO
                requires online sampling; a missing backend is detected here
                rather than deep inside ``compute_loss``.
        """
        if self._generation_backend is None:
            raise RuntimeError(
                "GRPOTrainer requires a generation_backend for online rollout. "
                "Pass one via the generation_backend constructor argument."
            )
        self._current_rollout: RolloutData | None = None
        self._advantage_stats: dict[str, float] = {"mean": 0.0, "std": 1.0, "count": 0.0}
        logger.info(
            "Rollout state initialized (group_size=%d, beta=%.4f)",
            self.algo_config["group_size"], self.algo_config["beta"],
            extra={"event": "grpo.rollout_state.init", "phase": "init", "step": 0},
        )

    def grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        ref_log_probs: torch.Tensor | None = None,
        *,
        clip_range: float,
        beta: float,
        loss_normalize: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute GRPO clipped surrogate loss with optional KL penalty.

        Delegates the clipped surrogate to ``compute_grpo_policy_loss`` and
        adds the reference-policy KL term when ``beta > 0``.

        Args:
            log_probs: Current policy log-probs, shape ``[B, T]``.
            old_log_probs: Rollout log-probs (detached), shape ``[B, T]``.
            advantages: Group-normalised advantages, shape ``[B]``.
            response_mask: Boolean mask over response tokens, shape ``[B, T]``.
            ref_log_probs: Reference policy log-probs, shape ``[B, T]``.
                Required when ``beta > 0``, ignored otherwise.
            clip_range: PPO clip epsilon.
            beta: KL penalty coefficient. Pass ``0`` to skip KL entirely.
            loss_normalize: ``"token"`` or ``"sequence"`` reduction mode.

        Returns:
            A ``(loss, metadata)`` pair. ``metadata`` contains per-token
            diagnostic scalars suitable for ``store_metrics``.
        """
        reduction = "token_mean" if loss_normalize == "token" else "sequence_mean"
        loss, metadata = compute_grpo_policy_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            clip_eps=clip_range,
            reduction=reduction,
        )
        if beta > 0:
            if ref_log_probs is None:
                raise ValueError("ref_log_probs is required when beta > 0")
            mask = response_mask.bool()
            kl = torch.where(mask, log_probs.float() - ref_log_probs.detach().float(), 0.0)
            kl_per_seq = kl.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            kl_loss = beta * kl_per_seq.mean()
            loss = loss + kl_loss
            metadata["policy/kl"] = kl_loss.detach()
        return loss, metadata

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Run a Trainer step and sync updated weights to the rollout backend."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._generation_backend.update_weight(self._iter_policy_params())
        return loss

    def _run_rollout(
        self,
        prompts: list[str],
        solutions: list[str],
    ) -> tuple[RolloutData, torch.Tensor, torch.Tensor]:
        """Generate group responses, score them, and return normalised rollout data.

        Args:
            prompts: Decoded prompt strings for the current batch, length ``B``.
            solutions: Reference answers aligned with ``prompts``, length ``B``.

        Returns:
            A ``RolloutData`` instance on CPU with tensors of shape
            ``[B * G, T]`` (log_probs, response_mask, old_log_probs) and
            ``[B * G]`` (rewards), where ``G`` is ``group_size`` and ``T`` is
            the padded completion length.
        """
        G = self.algo_config["group_size"]
        tokenizer = self.processing_class

        # Tokenize prompts; left-pad so generation appends to the right.
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        prompt_token_ids: list[list[int]] = enc["input_ids"].tolist()

        # Generate G completions per prompt → list of B*G GenerationOutput.
        outputs = self._generation_backend.generate(prompt_token_ids, self._generation_config)

        # Regroup: (B, G) of completion token ids and rollout log-probs.
        B = len(prompts)
        completions_by_prompt: list[list[list[int]]] = [[] for _ in range(B)]
        logprobs_by_prompt: list[list[tuple[float, ...]]] = [[] for _ in range(B)]
        for out in outputs:
            completions_by_prompt[out.prompt_index].append(list(out.completion_token_ids))
            logprobs_by_prompt[out.prompt_index].append(out.log_probs or ())

        # Pad completions to uniform length T across the whole batch.
        pad_id = tokenizer.pad_token_id
        T = max(
            len(ids)
            for group in completions_by_prompt
            for ids in group
        )
        padded_ids: list[list[int]] = []
        response_masks: list[list[int]] = []
        old_lp_rows: list[list[float]] = []
        for group_ids, group_lps in zip(completions_by_prompt, logprobs_by_prompt):
            for ids, lps in zip(group_ids, group_lps):
                pad_len = T - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                response_masks.append([1] * len(ids) + [0] * pad_len)
                old_lp_rows.append(list(lps) + [0.0] * pad_len)

        # Flatten completion texts for reward functions: B*G strings.
        flat_texts: list[str] = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for group in completions_by_prompt
            for ids in group
        ]
        # Each solution repeated G times to align with flat_texts.
        flat_solutions: list[str] = [sol for sol in solutions for _ in range(G)]

        # Aggregate rewards across all reward functions (mean, ignore None).
        raw_rewards: list[list[float | None]] = [
            fn(flat_texts, flat_solutions) for fn in self.reward_funcs
        ]
        agg_rewards: list[float] = []
        for i in range(B * G):
            vals = [r[i] for r in raw_rewards if r[i] is not None]
            agg_rewards.append(sum(vals) / len(vals) if vals else 0.0)

        # Group-normalise advantages: shape [B*G] with values broadcast back.
        eps = 1e-8
        reward_t = torch.tensor(agg_rewards, dtype=torch.float32)
        advantages = torch.zeros_like(reward_t)
        for b in range(B):
            sl = slice(b * G, (b + 1) * G)
            grp = reward_t[sl]
            advantages[sl] = (grp - grp.mean()) / (grp.std(unbiased=False) + eps)

        return RolloutData(
            rewards=reward_t,
            log_probs=torch.zeros(B * G, T),  # filled by compute_loss forward pass
            response_mask=torch.tensor(response_masks, dtype=torch.float32),
            old_log_probs=torch.tensor(old_lp_rows, dtype=torch.float32),
        ), advantages, torch.tensor(padded_ids, dtype=torch.long)

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run one GRPO step: rollout → advantage → policy loss + KL.

        Args:
            model: The active (possibly wrapped) policy model.
            inputs: Batch dict from the data collator; must contain
                ``"prompt"`` (list of str) and ``"solution"`` (list of str).
            return_outputs: Unused; kept for Trainer API compatibility.

        Returns:
            Scalar loss tensor with gradient attached.
        """
        prompts: list[str] = inputs["prompt"]
        solutions: list[str] = inputs["solution"]

        # --- rollout (no grad) ---
        with torch.no_grad():
            rollout, advantages, completion_ids = self._run_rollout(prompts, solutions)

        device = self.accelerator.device
        completion_ids = completion_ids.to(device)
        response_mask = rollout.response_mask.to(device)
        old_log_probs = rollout.old_log_probs.to(device)
        advantages = advantages.to(device)

        # Encode prompts again to build full input_ids for the policy forward.
        tokenizer = self.processing_class
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        prompt_ids = enc["input_ids"].to(device)       # [B, P]
        prompt_mask = enc["attention_mask"].to(device)  # [B, P]
        G = self.algo_config["group_size"]
        B = len(prompts)

        # Repeat each prompt G times to align with completions.
        prompt_ids = prompt_ids.repeat_interleave(G, dim=0)    # [B*G, P]
        prompt_mask = prompt_mask.repeat_interleave(G, dim=0)  # [B*G, P]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)      # [B*G, P+T]
        attn_mask = torch.cat([prompt_mask, response_mask.long()], dim=1)
        P = prompt_ids.shape[1]

        # --- policy forward ---
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        # Shift: predict token t from context up to t-1; slice response tokens only.
        shift_logits = logits[:, P - 1:-1, :]          # [B*G, T, V]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        log_probs = log_probs.gather(
            2, completion_ids.unsqueeze(-1)
        ).squeeze(-1)                                   # [B*G, T]

        # --- reference forward (KL penalty) ---
        ref_log_probs: torch.Tensor | None = None
        beta = self.algo_config["beta"]
        if beta > 0 and self.reference_model is not None:
            with torch.no_grad():
                ref_logits = self.reference_model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits
                shift_ref = ref_logits[:, P - 1:-1, :]
                ref_lp = torch.nn.functional.log_softmax(shift_ref, dim=-1)
                ref_log_probs = ref_lp.gather(
                    2, completion_ids.unsqueeze(-1)
                ).squeeze(-1)

        # --- loss ---
        loss, metadata = self.grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            response_mask=response_mask,
            ref_log_probs=ref_log_probs,
            clip_range=self.algo_config["cliprange"],
            beta=beta,
            loss_normalize=self.algo_config["loss_normalization"],
        )

        # Reward mean/std diagnostics.
        with torch.no_grad():
            metadata["reward/mean"] = rollout.rewards.mean()
            metadata["reward/std"] = rollout.rewards.std(unbiased=False)

        self.store_metrics(metadata)
        return loss
