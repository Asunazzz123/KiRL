"""Two-process acceptance tests for GRPO under FSDP (reference policy + weight sync).

Needs two CUDA devices (Accelerate falls back to DDP on CPU, so CPU runs skip):
    torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_fsdp_reference.py
On macOS, if rendezvous hangs, add ``--master_addr=127.0.0.1`` and drop ``--standalone``.

Each FSDP version trains a tiny random Llama for one optimizer step. All
collectives happen inside ``_run``; tests only assert on the returned plain
data, so a failing assertion on one rank cannot hang the other ranks.
"""

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.distributed.tensor import DTensor
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainingArguments,
)

from kirl.algorithms.grpo import GRPOTrainer
from kirl.datasets import BaseDataset
from kirl.engine.inference.base import GenerationBackend, GenerationConfig, GenerationOutput

pytestmark = [
    pytest.mark.skipif("RANK" not in os.environ, reason="needs torchrun: --nproc_per_node=2 -m pytest ..."),
    # Accelerate enables FSDP only on accelerator devices; on CPU it silently uses DDP.
    pytest.mark.skipif(
        torch.cuda.device_count() < int(os.environ.get("WORLD_SIZE", "0")),
        reason="FSDP needs one CUDA device per rank (CPU runs fall back to DDP)",
    ),
]

FSDP_VERSIONS = (1, 2)
GROUP_SIZE = 2
WORDS = tuple(f"w{i}" for i in range(24))
PROBE_PROMPT = "w1 w2 w3 w4"

# ---- builders -------------------------------------------------------------
def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>")


def _model() -> LlamaForCausalLM:
    """Same seed on every rank, so all ranks start from identical weights."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(WORDS) + 2, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=False,
        pad_token_id=0, eos_token_id=1, bos_token_id=1,
    )
    return LlamaForCausalLM(config)


class RecordingBackend(GenerationBackend):
    """Deterministic rollout; records what ``update_weight`` receives."""

    def __init__(self) -> None:
        self.syncs: list[dict[str, tuple[tuple[int, ...], bool]]] = []

    def generate(self, prompt_token_ids, config: GenerationConfig) -> list[GenerationOutput]:
        # Different lengths per group member -> different rewards -> nonzero advantages.
        return [
            GenerationOutput(prompt_index=i, completion_token_ids=tuple(range(2, 4 + g)),
                             log_probs=(0.0,) * (2 + g))
            for i in range(len(prompt_token_ids)) for g in range(config.num_return_sequences)
        ]

    def update_weight(self, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
        # Consume inside the call: the iterator may hold an all-gather context open.
        self.syncs.append({name: (tuple(t.shape), isinstance(t, DTensor)) for name, t in named_params})


def _length_reward(texts: list[str], solutions: list[str]) -> list[float]:
    return [float(len(text.split())) for text in texts]


# ---- observation ------------------------------------------------------------
def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _shards(module: torch.nn.Module) -> list[torch.Tensor]:
    return [_local(p).detach().clone() for p in module.parameters()]


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@dataclass
class RunResult:
    error: str | None = None
    fsdp_enabled: bool | None = None
    max_logit_diff: float | None = None
    ref_local_numel: int | None = None
    ref_requires_grad: bool | None = None
    ref_unchanged: bool | None = None
    policy_changed: bool | None = None
    syncs: list[dict[str, tuple[tuple[int, ...], bool]]] = field(default_factory=list)


class ReferenceProbe(TrainerCallback):
    """Runs on every rank; forwards here are collectives under FSDP."""

    def __init__(self, result: RunResult) -> None:
        self.result = result
        self.trainer: GRPOTrainer | None = None
        self._ref0: list[torch.Tensor] = []
        self._policy0: list[torch.Tensor] = []

    def on_train_begin(self, args, state, control, **kwargs):
        # The policy is wrapped by now (Trainer prepares it before this event).
        trainer, ref = self.trainer, self.trainer.reference_model
        ids = trainer.processing_class(PROBE_PROMPT, return_tensors="pt").input_ids
        ids = ids.to(trainer.accelerator.device)
        with torch.no_grad():
            policy_logits = trainer.model_wrapped(input_ids=ids).logits
            ref_logits = ref(input_ids=ids).logits
        self.result.max_logit_diff = (policy_logits - ref_logits).abs().max().item()
        self.result.ref_local_numel = sum(_local(p).numel() for p in ref.parameters())
        self.result.ref_requires_grad = any(p.requires_grad for p in ref.parameters())
        self._ref0, self._policy0 = _shards(ref), _shards(trainer.model_wrapped)

    def on_train_end(self, args, state, control, **kwargs):
        self.result.ref_unchanged = _same(self._ref0, _shards(self.trainer.reference_model))
        self.result.policy_changed = not _same(self._policy0, _shards(self.trainer.model_wrapped))


# ---- one training run per FSDP version ---------------------------------------
_RESULTS: dict[int, RunResult] = {}


def _run(version: int, output_dir: str) -> RunResult:
    # Trainer reuses Accelerate's singleton state; each version needs a fresh plugin.
    AcceleratorState._reset_state()
    GradientState._reset_state()
    result, backend = RunResult(), RecordingBackend()
    probe = ReferenceProbe(result)
    args = TrainingArguments(
        output_dir=output_dir, max_steps=1, per_device_train_batch_size=1,
        learning_rate=1e-2, report_to="none", save_strategy="no", logging_strategy="no",
        remove_unused_columns=False,  # keep "prompt"/"solution" for compute_loss
        seed=0,
        fsdp="full_shard",
        fsdp_config={"version": version, "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]},
    )
    samples = [{"prompt": "w1 w2", "solution": "w3"}, {"prompt": "w4 w5", "solution": "w6"}]
    try:
        trainer = GRPOTrainer(
            _model(), reward_fn=_length_reward, train_dataset=BaseDataset(samples),
            args=args, generation_backend=backend, processing_class=_tokenizer(),
            algo_config={"group_size": GROUP_SIZE, "beta": 0.1},
            generation_config={"max_new_tokens": 4},
        )
        result.fsdp_enabled = trainer.is_fsdp_enabled
        probe.trainer = trainer
        trainer.add_callback(probe)
        trainer.train()
    except Exception as exc:  # noqa: BLE001 -- recorded so every rank reaches the same asserts
        result.error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    result.syncs = backend.syncs
    return result


@pytest.fixture(params=FSDP_VERSIONS, ids=lambda v: f"fsdp{v}")
def run(request, tmp_path_factory) -> RunResult:
    version = request.param
    if version not in _RESULTS:
        _RESULTS[version] = _run(version, str(tmp_path_factory.mktemp(f"fsdp{version}")))
    return _RESULTS[version]


def _expected_params() -> dict[str, tuple[int, ...]]:
    return {name: tuple(p.shape) for name, p in _model().named_parameters()}


def _assert_ran_under_fsdp(run: RunResult) -> None:
    assert run.error is None, run.error
    # Guards against a silent DDP fallback making every check below pass vacuously.
    assert run.fsdp_enabled, "Trainer did not enable FSDP"


# ---- acceptance criteria ----------------------------------------------------------
def test_reference_matches_policy_at_step0(run):
    _assert_ran_under_fsdp(run)
    assert run.max_logit_diff is not None and run.max_logit_diff < 1e-5


def test_reference_is_frozen(run):
    _assert_ran_under_fsdp(run)
    assert run.policy_changed, "sanity: the optimizer step did not change the policy"
    assert run.ref_requires_grad is False
    assert run.ref_unchanged


def test_reference_is_sharded(run):
    _assert_ran_under_fsdp(run)
    total = sum(math.prod(shape) for shape in _expected_params().values())
    world = int(os.environ["WORLD_SIZE"])
    # Allow ~10% for FSDP padding; an unsharded copy would be ~world x larger.
    assert run.ref_local_numel * world <= total * 1.1, (run.ref_local_numel, total)


def test_weight_sync_yields_full_unprefixed_params(run):
    _assert_ran_under_fsdp(run)
    assert len(run.syncs) == 1, "expected exactly one sync per optimizer step"
    synced = run.syncs[0]
    assert not [n for n, (_, is_dtensor) in synced.items() if is_dtensor], "sharded DTensors synced"
    assert {n: shape for n, (shape, _) in synced.items()} == _expected_params()
