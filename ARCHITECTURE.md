# KiRL Architecture

## Design Intent

KiRL is an educational RL training engine built on top of HuggingFace Transformers
`Trainer`. It explores:
- How to decouple algorithm logic (GRPO/PPO/DPO policy losses) from training infrastructure
- How to separate online rollout generation (vLLM) from gradient-based policy updates
- How FSDP/DDP sharding interacts with weight synchronization to an inference backend

**Goals:**
- Clean separation between algorithm, data, config, and inference layers
- Pluggable algorithms and datasets via a name-based registry (no decorators, just a YAML whitelist + factory)
- Multi-GPU training (FSDP/DDP via Accelerate) with TP-based inference (vLLM)

**Non-goals:**
- Production-scale performance optimization (verl already does this)
- A custom training loop — the loop is owned by Transformers `Trainer`
- Complex distributed strategies beyond FSDP/DDP for training and TP for inference

**Anti-goals:**
- Monolithic design where adding a new algorithm requires modifying engine core
- Hard-coded dataset loading logic in training scripts

---

## Architecture Overview

```
Hydra CLI / YAML config
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  config/   resolve_train_config(cfg) → TrainConfig  │
│  schema.py + adapters.py + registry.yaml            │
└───────────┬─────────────────────────────────────────┘
            │ TrainConfig { resolved, training_args,
            │               algo_config, generation, rollout, ... }
   ┌────────┼────────────────────────────────────────┐
   │        │                                        │
   ▼        ▼                                        ▼
┌──────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│ datasets/    │  │ algorithms/          │  │ engine/inference/  │
│ BaseDataset  │  │ _BaseTrainer(Trainer)│  │ vLLMEngine         │
│ RLDataCollator│ │ GRPOTrainer          │  │ GenerationBackend  │
│ build_dataset│  │ core_algo.py (losses)│  │ (ABC)              │
└──────────────┘  └──────────────────────┘  └────────────────────┘
                         │                          ▲
                         │  generation_backend ─────┘
                         │  update_weight ──────────►
                         ▼
┌─────────────────────────────────────────────────────┐
│  distributed/  FSDPBackend + FSDPConfig (FSDP/FSDP2)│
│  profiler/     TorchProfilerAdapter                 │
│  utils/logging Log                                  │
└─────────────────────────────────────────────────────┘
```

The training loop is owned by Transformers `Trainer`; KiRL customizes it by
subclassing `_BaseTrainer(Trainer)` and overriding `compute_loss` /
`training_step`. Rollout generation is delegated to a `GenerationBackend`
(vLLM) that runs without gradients and is synchronized after each optimizer
step.

---

## Layer 1: Config (`kirl/config/`)

```
schema.py          Hydra-compatible dataclasses (AlgoSchema, DatasetSchema, …)
adapters.py        resolve_train_config() — validates + normalizes → TrainConfig
registry.yaml      whitelist of allowed names per dimension
```

`resolve_train_config(cfg)` merges YAML into the structured schema, validates
all numeric/enum constraints, resolves all paths to absolute, and returns a
`TrainConfig` dataclass that the pipeline uses to construct runtime objects.
No model or device is initialized here.

`TrainConfig` fields:

| Field | Type | Contents |
|---|---|---|
| `resolved` | `dict` | Full merged config including `dataset`, `reward`, etc. |
| `training_args` | `dict` | kwargs for `TrainingArguments` constructor |
| `algo_config` | `dict` | GRPO/PPO/DPO hyperparameters |
| `generation_config` | `dict` | Raw generation params dict |
| `generation` | `GenerationConfig` | Typed backend-agnostic sampling params |
| `rollout` | `VLLMBackendConfig` | vLLM engine constructor args |
| `profiler` | `ProfilerSettings` | Profiler config |

Adding a new pluggable name: add it to `registry.yaml` only. `adapters.py`
reads the YAML at import time; no Python changes are needed.

---

## Layer 2: Dataset (`kirl/datasets/`)

```
base.py        BaseDataset(torch.Dataset) + RLDataCollator
factory.py     build_dataset(cfg, split) — name → loader
gsm8k.py       load_gsm8k(path, prompt_template_path)
math.py        (stub)
custom.py      (stub)
```

**`BaseDataset`** stores samples as `list[dict]` and exposes the standard
`__len__` / `__getitem__` interface. Each sample must be a `{str: Any}` dict.

**`RLDataCollator`** replaces the HF default collator, which would try to
stack string fields into tensors. It groups each field across the batch into a
plain `list`; tensor-valued fields are `torch.stack`-ed normally.

**`build_dataset(cfg, split)`** maps `cfg["name"]` through `_LOADERS` to the
appropriate loader, then applies cross-cutting options (`max_samples`, `seed`).
Returns `None` when the split path is not configured (eval split is optional).

Standard sample keys produced by all loaders:

| Key | Type | Description |
|---|---|---|
| `sample_id` | `str` | Unique sample identifier |
| `prompt` | `str` | Input prompt (with optional template applied) |
| `solution` | `str` | Reference answer for reward computation |

---

## Layer 3: Algorithm (`kirl/algorithms/`)

```
base.py        _BaseTrainer(Trainer) — shared infrastructure
grpo.py        GRPOTrainer(_BaseTrainer)
core_algo.py   compute_grpo_policy_loss, compute_ppo_policy_loss, compute_dpo_loss
reward.py      math_accuracy_reward
```

### `_BaseTrainer`

Provides infrastructure shared across all RL trainers:

| Method | Role |
|---|---|
| `_resolve_model` | `str \| PreTrainedModel → PreTrainedModel` |
| `_resolve_tokenizer` | Load + configure padding |
| `_prepare_reference_policy` | Prepare the frozen KL reference through Accelerate (single device, DDP or FSDP); DeepSpeed reference preparation is unsupported |
| `_normalize_rewards` | `fn \| list[fn] → list[fn]` |
| `_validate_dataset` | type check |
| `_validate_config` | validate `TrainingArguments` |
| `_iter_policy_params` | `yield (name, tensor)` FSDP-aware |
| `store_metrics` | weighted accumulation buffer |
| `log` | merge + cross-rank gather into Trainer pipeline |
| `evaluate` | mode-switching wrapper |
| `training_step` | `Log.bind` context wrapper |

Policy export keeps each gathered tensor alive until the inference backend consumes it. FSDP1 uses `summon_full_params`, FSDP2 calls `DTensor.full_tensor()` per parameter, and ZeRO-3 uses `GatheredParameters`. DDP and single-device training yield the unwrapped parameters directly.

### `GRPOTrainer`

A Trainer update may accumulate several microbatches:

```
training_step (overridden)
  └─ super().training_step          Trainer forward/backward for one microbatch
       └─ compute_loss
            ├─ _run_rollout          tokenize → generate(G/prompt) →
            │                        reward_funcs → group advantage normalise
            ├─ policy forward        input_ids=[prompt‖completion] → log_probs [B*G,T]
            ├─ reference forward     ref_log_probs [B*G,T]  (beta > 0 only)
            └─ grpo_loss             clip surrogate + KL → (loss, metadata)
RolloutWeightSyncCallback.on_step_end (after a successful optimizer update)
  └─ _iter_policy_params → update_weight
```

**`_run_rollout`** returns `(RolloutData, advantages, completion_ids)`:
- Tokenizes prompts, calls `GenerationBackend.generate` for `G` completions/prompt
- Regroups `GenerationOutput` by `prompt_index`, right-pads to uniform length `T`
- Decodes texts, calls each reward function, mean-aggregates (None-safe)
- Group-normalises advantages: `A_i = (r_i − μ) / (σ + ε)` within each group of G

**`grpo_loss`** delegates clipped surrogate to `compute_grpo_policy_loss` in
`core_algo.py` and adds the KL term when `beta > 0`:

```
reduction  =  "token_mean" | "sequence_mean"   (from loss_normalization)
loss       =  clip_surrogate + beta * mean(KL per sequence)
```

### `core_algo.py`

Pure tensor functions with no trainer state:

| Function | Use |
|---|---|
| `compute_grpo_policy_loss` | GRPO clipped surrogate, returns `(loss, metadata)` |
| `compute_dr_grpo_policy_loss` | Dr.GRPO variant (length-normalised, no KL) |
| `compute_ppo_policy_loss` | PPO-Clip actor loss |
| `compute_ppo_value_loss` | PPO clipped critic loss |
| `compute_dpo_loss` | Sigmoid DPO loss |

---

## Layer 4: Inference Backend (`kirl/engine/inference/`)

```
base.py         GenerationConfig, GenerationOutput, GenerationBackend(ABC)
vllm_engine.py  vLLMEngine(GenerationBackend)
vllm_config.py  VLLMBackendConfig
```

**`GenerationBackend`** ABC exposes a single method:

```python
def generate(
    self,
    prompt_token_ids: Sequence[Sequence[int]],
    config: GenerationConfig,
) -> list[GenerationOutput]:
```

**`vLLMEngine`** wraps a `vllm.LLM` instance. Key methods:

| Method | Role |
|---|---|
| `from_config(config)` | `VLLMBackendConfig → vLLMEngine` |
| `generate(...)` | Batched rollout, extracts per-token logprobs |
| `update_weight(named_params)` | In-place weight sync via `model_runner.load_weights` |
| `start_client(config, host, port)` | Launch vLLM OpenAI server subprocess, poll `/health` |
| `client()` | Return active `openai.OpenAI` handle |
| `close_client()` | Terminate server, release handle |

Two deployment modes:

```
Embedded (default)        Server (client mode)
────────────────          ────────────────────
LLM + Trainer             vLLM subprocess  ◄──► Trainer via OpenAI API
same process              separate process
update_weight → direct    update_weight → HTTP (not yet implemented)
from_config               start_client
```

---

## Layer 5: Distributed (`kirl/distributed/`)

```
fsdp_config.py    FSDPConfig — FSDP/FSDP2 hyperparameters + validation
fsdp_backend.py   FSDPBackend — model wrapping, broadcast, optimizer offload
```

**`FSDPBackend.prepare_mode(model, transformer_layer_cls)`** wraps the model:

- **FSDP1**: `transformer_auto_wrap_policy` + `ShardingStrategy` + `MixedPrecision`;
  `use_orig_params=True` preserves parameter names through `summon_full_params`.
- **FSDP2**: Manual `fully_shard` per transformer block (reversed order, innermost first),
  then root; uses `MixedPrecisionPolicy`.

Sharding unit is determined by `transformer_layer_cls` (e.g. `LlamaDecoderLayer`);
the backend validates that at least one child module matches before wrapping.

FSDP can offload model and gradient shards through its CPU offload policy. Separately, `HybridDeviceOptimizer` in `kirl/engine/optim.py` places optimizer subgroups on CPU or the parameter device and delegates updates to the selected algorithm. Trainer-managed FSDP wrapping runs through Accelerate.

## Training and rollout placement

Today each participating training rank enters the weight-export callback in the same order. The embedded vLLM backend synchronously loads the resulting parameter stream into its driver model. Configuring inference TP alone does not establish weight delivery to every worker of a separately placed replica.

The planned thin runtime will schedule a training group and a pool of rollout replicas independently. A completed update will publish a policy version; each replica will acknowledge it before serving requests for that version. A replayable trajectory will carry token IDs, action masks, behavior logprobs and that version. OPD will score student actions with a frozen teacher, while an agent sandbox will supply tool observations between actions. See the [roadmap](TODO.md) for implementation and acceptance tasks.

---

## Configuration Flow

```
configs/train.yaml + selected config groups
        │
        ▼ (Hydra compose)
resolve_train_config(cfg)
        │
        ├─ TrainConfig.training_args  → TrainingArguments(...)
        ├─ TrainConfig.algo_config    → GRPOTrainer(algo_config=...)
        ├─ TrainConfig.optimizer_cls_and_kwargs → Trainer optimizer selection
        ├─ TrainConfig.generation     → generation backend sampling settings
        ├─ TrainConfig.rollout        → vLLMEngine.from_config(...)
        └─ TrainConfig.resolved["dataset"] → build_dataset(cfg, split)
```

---

## Extending KiRL

### Add a new algorithm
1. Add name to `registry.yaml` under `algo:`
2. Implement pure loss function in `core_algo.py`
3. Subclass `_BaseTrainer`, override `compute_loss` (and `_run_rollout` if online)

### Add a new dataset
1. Add name to `registry.yaml` under `dataset:`
2. Write a loader function `load_<name>(path, **kwargs) → BaseDataset`
3. Add a `lambda cfg, split: ...` entry to `_LOADERS` in `factory.py`

### Add a new reward function
1. Add name to `registry.yaml` under `reward:`
2. Implement `fn(completions, solution, ...) -> list[float | None]` in `reward.py`

---

## Constraints

1. **Python 3.11+** (union types, `typing.Self`)
2. **PyTorch 2.2+** (FSDP2 `fully_shard`, `torch.compile` support)
3. **Single-node multi-GPU focus** for initial validation
4. **Hydra for configuration** — `register_configs()` must be called before compose
5. **No custom CUDA kernels** — keep implementation readable
6. **HuggingFace Trainer owns the loop** — KiRL hooks in via `compute_loss` / `training_step`

---

## References

- **verl**: Main reference for algorithm implementations and distributed patterns
- **HuggingFace TRL**: Reference for `compute_loss` integration and reward shaping
- **PyTorch FSDP docs**: https://pytorch.org/docs/stable/fsdp.html
- **vLLM docs**: https://docs.vllm.ai/en/stable/

