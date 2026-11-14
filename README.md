# KiRL

An educational RL training engine for understanding bottom-level design patterns in reinforcement learning systems.

## Design Goals

- **Educational Focus**: Learn how RL training engines work under the hood
- **Clean Architecture**: Decouple algorithm, data, and infrastructure layers
- **Multi-Algorithm Support**: Easily switch between PPO, GRPO, DPO through configuration
- **Multi-GPU Training**: FSDP/DDP support with minimal boilerplate

## Project Structure

```
KiRL/
├── kirl/
│   ├── algorithms/              # Pluggable RL algorithms
│   │   ├── base.py             # _BaseTrainer: shared Trainer infrastructure
│   │   ├── core_algo.py        # Loss kernels: GRPO, PPO, DPO, Dr.GRPO
│   │   ├── grpo.py             # GRPOTrainer (scaffold; loss not yet wired)
│   │   ├── ppo.py              # PPOTrainer (stub)
│   │   ├── dpo.py              # DPOTrainer (stub)
│   │   ├── sft.py              # SFTTrainer
│   │   └── reward.py           # math_accuracy_reward
│   ├── config/                  # Hydra schema + config adapter
│   │   ├── schema.py           # TrainSchema dataclass (OmegaConf structured)
│   │   ├── adapters.py         # resolve_train_config: validate + build TrainConfig
│   │   └── registry.yaml       # Pluggable name whitelists (algo/engine/dataset/…)
│   ├── datasets/                # Pluggable datasets
│   │   ├── base.py             # BaseDataset (torch.utils.data.Dataset)
│   │   ├── factory.py          # DatasetFactory registry
│   │   ├── gsm8k.py            # GSM8K JSONL loader (raw text; no tokenization yet)
│   │   ├── math.py             # MATH dataset (stub)
│   │   └── custom.py           # Custom dataset loader
│   ├── distributed/             # Distributed utilities
│   │   ├── context.py          # DistributedContext: torchrun, barrier, all_reduce…
│   │   ├── fsdp_config.py      # FSDPConfig dataclass
│   │   └── fsdp_backend.py     # FSDPBackend: FSDP1/FSDP2 wrapping + probe hooks
│   ├── engine/                  # Engine core
│   │   ├── checkpoint.py       # Checkpoint management (stub)
│   │   └── inference/
│   │       ├── base.py         # GenerationBackend ABC, GenerationConfig, GenerationOutput
│   │       ├── vllm_config.py  # VLLMBackendConfig dataclass
│   │       └── vllm_engine.py  # vLLMEngine: generate, weight_sync, server mode
│   ├── pipeline/                # Training/inference pipeline
│   │   ├── train.py            # Training loop (stub)
│   │   ├── evaluate.py         # Evaluation loop
│   │   └── rewards.py          # Reward function exports
│   ├── profiler/                # Non-blocking observation layer
│   │   ├── basic_profiler.py   # ProfilerConfig (off/basic/sampled/trace)
│   │   ├── fsdp_profiler.py    # TorchProfilerAdapter → chrome trace
│   │   ├── vllm_profiler.py    # VLLMProfilerAdapter → vLLM start/stop_profile
│   │   ├── probe.py            # Probe / NullProbe / create_probe
│   │   ├── records.py          # MetricRecord, StageRecord, TraceArtifact
│   │   ├── sink.py             # JsonlSink: bounded queue + background writer
│   │   ├── report.py           # build_report: offline RunReport from JSONL
│   │   └── settings.py         # ProfilerSettings.from_mapping (Hydra adapter)
│   └── utils/                   # Logging, helpers, determinism
├── configs/                     # Hydra configurations
│   ├── algo/                   # Algorithm configs (grpo.yaml, …)
│   ├── dataset/                # Dataset configs
│   ├── engine/                 # Engine configs (fsdp.yaml, ddp.yaml)
│   ├── rollout/                # Rollout configs (vllm.yaml)
│   └── train.yaml              # Main training config (defaults list)
├── examples/                    # Example training scripts
├── tests/                       # Unit tests
├── ARCHITECTURE.md              # Detailed architecture documentation
└── TODO.md                      # Implementation roadmap
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets hydra-core wandb
```

### Training Example

```bash
# Single GPU training
python examples/train_grpo_gsm8k.py

# Multi-GPU training with torchrun
torchrun --nproc_per_node=2 examples/train_grpo_gsm8k.py

# Modify config via command line
python examples/train_grpo_gsm8k.py algo=ppo dataset=math
```

## Architecture Highlights

### Configuration Layer

`resolve_train_config` is the boundary between Hydra and the training pipeline. It merges `TrainSchema` over the user's YAML, resolves all `${...}` interpolations, runs semantic validation (numeric bounds, cross-field constraints, enum whitelists from `registry.yaml`), resolves relative paths, and constructs three strongly-typed objects — `training_args`, `GenerationConfig`, `VLLMBackendConfig` — that the pipeline can trust without further checking.

### Profiler

A five-layer, non-blocking observation system:

```
ProfilerSettings (Hydra adapter)
    ├── ProfilerConfig      →  Probe / NullProbe
    ├── TorchProfilerConfig →  TorchProfilerAdapter  (chrome trace)
    └── VLLMProfilerConfig  →  VLLMProfilerAdapter   (injected engine)
                                     ↓
                              Probe.iteration() / .stage() / .metric()
                                     ↓
                              JsonlSink  (bounded queue, background writer)
                                     ↓
                              records.jsonl + status.json
                                     ↓
                              build_report() → RunReport (JSON)
```

`create_probe()` returns `NullProbe` for unselected ranks — same interface, zero overhead. Both `FSDPBackend` and `vLLMEngine` accept a `probe=` kwarg and wrap their critical sections (`fsdp.wrap`, `rollout`) with `probe.stage(...)`.

### Distributed

`DistributedContext` wraps torchrun process group primitives (`barrier`, `all_reduce`, `all_gather`, `broadcast`, `root_call`). `FSDPBackend` supports both FSDP1 (`FullyShardedDataParallel` + `transformer_auto_wrap_policy`) and FSDP2 (`fully_shard` + `init_device_mesh`), selected via `FSDPConfig.mode`.

### Inference

`vLLMEngine` runs in two modes: embedded (`from_config` → `LLM(...)`) or server (`start_client` launches `vllm.entrypoints.openai.api_server` as a subprocess and connects an OpenAI-compatible client). `generate()` returns `list[GenerationOutput]` with per-token logprobs, consumed directly by the GRPO rollout loop. `update_weight()` syncs updated policy weights into the running engine without reloading.

### Algorithm Loss Kernels

`core_algo.py` contains standalone, unit-testable loss functions that have no dependency on `Trainer` or distributed state:

- `compute_grpo_policy_loss` — PPO-clip with per-sequence or per-token normalization
- `compute_dr_grpo_policy_loss` — Dr. GRPO variant (length-normalized denominator)
- `compute_ppo_policy_loss` / `compute_ppo_value_loss` — PPO actor + critic losses
- `compute_dpo_loss` — sigmoid DPO over preference pairs

## Key Features

### Multi-GPU Support
- **FSDP** (Fully Sharded Data Parallel): Shard model parameters across GPUs
- **DDP** (Distributed Data Parallel): Replicate model on each GPU
- Switch via config: `engine: fsdp` or `engine: ddp`

### Flexible Configuration
All hyperparameters managed through Hydra configs:
```yaml
# configs/train.yaml
defaults:
  - algo: grpo
  - dataset: gsm8k
  - engine: fsdp

num_steps: 1000
batch_size: 8
learning_rate: 1.0e-5
```

### Deterministic Training
Set seed for reproducible results:
```yaml
seed: 42
deterministic: true
```

## Adding New Components

KiRL uses a YAML registry (`kirl/config/registry.yaml`) as the single source of
truth for supported pluggable names. `resolve_train_config` reads it at import
time, so the validation whitelist stays in sync automatically — no Python changes
are needed in `adapters.py` when adding a new name.

```yaml
# kirl/config/registry.yaml
algo:    [grpo, ppo, dpo]       # add your new algo name here
engine:  [ddp, fsdp]
dataset: [gsm8k, math]          # add your new dataset name here
rollout: [vllm]
reward:  [gsm8k_reward, math_reward, external_model]
```

### New Algorithm

1. Add the algorithm name to `registry.yaml` under `algo`.
2. Create `kirl/algorithms/your_algo.py` and implement the trainer, inheriting
   from `_BaseTrainer`.  Optionally register a config sentinel via the
   `algorithms` registry:

   ```python
   # kirl/algorithms/your_algo.py
   from dataclasses import dataclass
   from kirl.algorithms import register_algo

   @register_algo("your_algo")
   @dataclass(frozen=True)
   class YourAlgoConfig:
       name: str = "your_algo"
       cliprange: float = 0.2
       # ... algo-specific fields
   ```

3. Add a Hydra config to `configs/algo/your_algo.yaml` and reference it in the
   experiment defaults.

### New Dataset

1. Add the dataset name to `registry.yaml` under `dataset`.
2. Create `kirl/datasets/your_dataset.py` and implement the dataset class.
3. Add a Hydra config to `configs/dataset/your_dataset.yaml`.

## Implementation Status

| Module | Status |
|---|---|
| Config schema + `resolve_train_config` | ✅ Complete |
| `DistributedContext` | ✅ Complete |
| `FSDPBackend` (FSDP1 + FSDP2) | ✅ Complete |
| `vLLMEngine` (embedded + server mode) | ✅ Complete |
| Profiler (`Probe`, `JsonlSink`, `build_report`) | ✅ Complete |
| Loss kernels (`core_algo.py`) | ✅ Complete |
| `math_accuracy_reward` | ✅ Complete |
| GSM8K dataset (raw text loader) | 🟡 Partial — tokenization / collate_fn missing |
| `GRPOTrainer` | 🟡 Scaffold — `compute_loss` / rollout loop not wired |
| `SFTTrainer` | 🟡 In progress |
| Training loop (`pipeline/train.py`) | 🔴 Stub |
| `gsm8k_reward` | 🔴 Missing |
| Checkpoint management | 🔴 Stub |
| MATH dataset | 🔴 Stub |
| PPO / DPO trainers | 🔴 Stub |

See [TODO.md](TODO.md) for the full milestone roadmap.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture design
- [TODO.md](TODO.md) - Implementation roadmap with milestones

## References

- **verl**: Reference for algorithm implementations
- **PyTorch FSDP**: Distributed training patterns
- **Transformers**: Checkpoint management

## License

MIT License - Educational project for learning RL systems design.
