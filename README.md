# TinyEngine

An educational RL training engine for understanding bottom-level design patterns in reinforcement learning systems.

## Design Goals

- **Educational Focus**: Learn how RL training engines work under the hood
- **Clean Architecture**: Decouple algorithm, data, and infrastructure layers
- **Multi-Algorithm Support**: Easily switch between PPO, GRPO, DPO through configuration
- **Multi-GPU Training**: FSDP/DDP support with minimal boilerplate

## Project Structure

```
TinyEngine/
├── tiny_engine/
│   ├── algorithms/         # Pluggable RL algorithms
│   │   ├── base.py        # Base algorithm interface
│   │   ├── grpo.py        # GRPO implementation
│   │   ├── ppo.py         # PPO implementation
│   │   └── dpo.py         # DPO implementation
│   ├── datasets/          # Pluggable datasets
│   │   ├── base.py        # Base dataset interface
│   │   ├── gsm8k.py       # GSM8K dataset
│   │   ├── math.py        # MATH dataset
│   │   └── custom.py      # Custom dataset loader
│   ├── engine/            # Engine core
│   │   ├── tiny_engine.py # Main engine implementation
│   │   └── checkpoint.py  # Checkpoint management
│   ├── pipeline/          # Training/inference pipeline
│   │   ├── train.py       # Training loop
│   │   ├── evaluate.py    # Evaluation loop
│   │   └── rewards.py     # Reward functions
│   ├── distributed/       # Distributed utilities
│   │   ├── context.py     # Distributed context
│   │   └── fsdp_config.py # FSDP configuration
│   └── utils/             # Logging, helpers
├── configs/               # Hydra configurations
│   ├── algo/             # Algorithm configs
│   ├── dataset/          # Dataset configs
│   ├── engine/           # Engine configs
│   └── train.yaml        # Main training config
├── examples/             # Example training scripts
├── tests/                # Unit tests
├── ARCHITECTURE.md       # Detailed architecture documentation
└── TODO.md              # Implementation roadmap
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

### Profiling and reports

The optional profiler provides global/rank sampling, PyTorch training traces, an injected vLLM client adapter, bounded JSONL output, and offline JSON reports. 

This example runs a small CPU training workload; it does not validate distributed FSDP or vLLM inference.

### Algorithm Module
Algorithms are pluggable through a registry pattern:
```python
@register_algorithm("grpo")
class GRPOAlgorithm(BaseAlgorithm):
    def compute_advantages(self, rollout_data, config):
        # GRPO-specific advantage estimation
        ...

    def compute_loss(self, advantages, log_probs, old_log_probs, config):
        # GRPO policy gradient loss
        ...
```

### Dataset Module
Datasets encapsulate their own loading and parsing logic:
```python
@register_dataset("gsm8k")
class GSM8KDataset(BaseDataset):
    def load_raw_data(self, split):
        # Load GSM8K from disk or HuggingFace
        ...

    def parse_sample(self, raw_sample):
        # Parse into model input format
        ...
```

### Engine Core
The engine manages model lifecycle and distributed training:
- FSDP/DDP wrapping
- Forward/backward/optimizer step
- Generation (rollout)
- Checkpoint save/load

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

### New Algorithm
1. Create `tiny_engine/algorithms/your_algo.py`
2. Inherit from `BaseAlgorithm` and implement:
   - `compute_advantages()`
   - `compute_loss()`
3. Register with `@register_algorithm("your_algo")`
4. Add config to `configs/algo/your_algo.yaml`

### New Dataset
1. Create `tiny_engine/datasets/your_dataset.py`
2. Inherit from `BaseDataset` and implement:
   - `load_raw_data()`
   - `parse_sample()`
   - (Optional) `collate_fn()`, `get_shard()`
3. Register with `@register_dataset("your_dataset")`
4. Add config to `configs/dataset/your_dataset.yaml`

## Implementation Status

See [TODO.md](TODO.md) for detailed implementation roadmap.

**Current Phase**: Project structure and documentation (Phase 0)

**Next Milestone**: Minimal working engine on single GPU (Phase 1)

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture design
- [TODO.md](TODO.md) - Implementation roadmap with milestones

## References

- **verl**: Reference for algorithm implementations
- **PyTorch FSDP**: Distributed training patterns
- **Transformers**: Checkpoint management

## License

MIT License - Educational project for learning RL systems design.
