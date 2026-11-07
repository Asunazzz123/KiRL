# KiRL Architecture

## Design Intent

KiRL is an educational RL training engine designed to help understand:
- Bottom-level design patterns in RL training systems
- Module optimization opportunities
- How to decouple algorithm logic from training/inference infrastructure

**Goals:**
- Clean separation between algorithm, data, and infrastructure layers
- Support multiple RL algorithms (PPO, GRPO, DPO) through pluggable modules
- Multi-GPU training and inference with minimal boilerplate

**Non-goals:**
- Production-scale performance optimization (verl already does this)
- Complex distributed strategies beyond FSDP/DDP
- Auto-tuning or hyperparameter search

**Anti-goals:**
- Monolithic design where adding a new algorithm requires modifying engine core
- Hard-coded dataset loading logic in training scripts

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training/Inference Pipeline               │
│  (Orchestrates algorithm + data + engine for train/eval)    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Algorithm   │ │   Dataset    │ │    Engine    │
│   Module     │ │   Module     │ │     Core     │
└──────────────┘ └──────────────┘ └──────────────┘
│                │                │
│ Pluggable     │ Pluggable      │ Manages:
│ RL algos:     │ datasets:      │ - Model loading
│ - GRPO        │ - GSM8K        │ - FSDP/DDP
│ - PPO         │ - MATH         │ - Gradient sync
│ - DPO         │ - Custom       │ - Checkpointing
│                │                │
│ Interface:    │ Interface:     │ Interface:
│ - compute_    │ - load_raw_    │ - forward()
│   advantages  │   data()       │ - backward()
│ - compute_    │ - parse_       │ - generate()
│   loss()      │   sample()     │ - save/load()
│                │ - collate_fn() │
│                │ - get_shard()  │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## Layer 1: Algorithm Module

### Design Principle
**Algorithms should be pure functions that compute loss given rollout data.**

### Base Interface
```python
class BaseAlgorithm:
    algo: str = "base"  # Algorithm identifier for registry
    
    def compute_advantages(
        self, 
        rollout_data: RolloutData,
        config: AlgoConfig
    ) -> torch.Tensor:
        """Compute advantages from rollout data.
        
        Args:
            rollout_data: Contains rewards, log_probs, values, masks
            config: Algorithm-specific hyperparameters
            
        Returns:
            advantages: Shape [batch_size, seq_len]
        """
        raise NotImplementedError
    
    def compute_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        config: AlgoConfig
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute policy loss and metrics.
        
        Returns:
            loss: Scalar tensor for backward()
            metrics: Dict of loggable metrics
        """
        raise NotImplementedError
```

### Registry Pattern
```python
ALGO_REGISTRY: dict[str, type[BaseAlgorithm]] = {}

@register_algorithm("grpo")
class GRPOAlgorithm(BaseAlgorithm):
    algo = "grpo"
    
    def compute_advantages(self, rollout_data, config):
        # Group-based advantage estimation
        ...
    
    def compute_loss(self, advantages, log_probs, old_log_probs, config):
        # GRPO policy gradient loss
        ...
```

### Key Decisions
- **Algorithms DO NOT hold model parameters** (engine manages those)
- **Algorithms DO hold their hyperparameters** (read from `config.grpo.*`)
- **Algorithms are stateless** (no internal buffers; all state in rollout_data)

---

## Layer 2: Dataset Module

### Design Principle
**Each dataset encapsulates its own loading and parsing logic, exposing a uniform interface to the engine.**

### Base Interface
```python
class BaseDataset:
    name: str = "base"
    
    def load_raw_data(self, split: str) -> list[dict]:
        """Load raw dataset from disk/HF.
        
        Args:
            split: "train" | "validation" | "test"
            
        Returns:
            List of raw samples (JSON-like dicts)
        """
        raise NotImplementedError
    
    def parse_sample(self, raw_sample: dict) -> dict[str, Any]:
        """Parse one raw sample into model input format.
        
        Returns:
            Dict with keys like "prompt", "question", "ground_truth"
        """
        raise NotImplementedError
    
    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Batch collation for DataLoader.
        
        Returns:
            Dict with "input_ids", "attention_mask", etc.
        """
        raise NotImplementedError
    
    def get_shard(
        self, 
        data: list[dict], 
        rank: int, 
        world_size: int
    ) -> list[dict]:
        """Shard data for distributed training.
        
        Default: rank-based round-robin sharding
        Can override for custom sharding strategies
        """
        return data[rank::world_size]
```

### Registry Pattern
```python
DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}

@register_dataset("gsm8k")
class GSM8KDataset(BaseDataset):
    name = "gsm8k"
    
    def load_raw_data(self, split):
        return load_gsm8k_json(split)
    
    def parse_sample(self, raw_sample):
        return {
            "prompt": self.template.format(question=raw_sample["question"]),
            "ground_truth": raw_sample["answer"]
        }
```

### Key Decisions
- **Dataset handles its own format** (JSON, HF datasets, CSV, etc.)
- **Engine calls uniform interface** (`load_raw_data` → `parse_sample` → `collate_fn`)
- **Sharding is handled by dataset** (via `get_shard`), but engine provides default implementation

---

## Layer 3: Engine Core

### Design Principle
**Engine manages model lifecycle, distributed communication, and gradient updates, but remains algorithm-agnostic.**

### Core Responsibilities
1. **Model Management**
   - Load model from HF/local checkpoint
   - Wrap with FSDP/DDP for distributed training
   - Handle mixed-precision (bfloat16/float32)
   - Apply gradient checkpointing

2. **Distributed Communication**
   - Initialize NCCL/Gloo process group
   - Coordinate across ranks (all_reduce, all_gather, broadcast)
   - Handle rank-specific initialization (seed, device placement)

3. **Training Primitives**
   - `forward(input_ids, attention_mask)` → logits
   - `backward(loss)` → gradient computation + sync
   - `step()` → optimizer step + lr scheduling
   - `generate(input_ids, gen_config)` → rollout generation

4. **Checkpoint Management**
   - Save full model + optimizer state
   - Save best checkpoint by metric
   - Resume from checkpoint

### Interface
```python
class KiRL:
    def __init__(
        self,
        model_factory: Callable,
        fsdp_config: FSDPConfig,
        context: DistributedContext
    ):
        self.model = self._init_model(model_factory, fsdp_config)
        self.optimizer = self._init_optimizer()
        self.context = context
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        ...
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient accumulation."""
        ...
    
    def step(self):
        """Optimizer step with gradient clipping."""
        ...
    
    def generate(
        self, 
        input_ids: torch.Tensor,
        gen_config: GenerationConfig
    ) -> RolloutResult:
        """Generate responses for RL rollout."""
        ...
    
    def save_checkpoint(self, path: Path, metadata: dict):
        """Save model + optimizer state."""
        ...
    
    def load_checkpoint(self, path: Path) -> dict:
        """Load checkpoint and return metadata."""
        ...
```

### Key Decisions
- **Engine does NOT know about algorithms** (receives loss, computes gradients)
- **Engine does NOT know about datasets** (receives batched tensors)
- **FSDP vs DDP is a configuration choice**, not hard-coded

---

## Layer 4: Training/Inference Pipeline

### Design Principle
**Pipeline orchestrates the three modules (algorithm + data + engine) to execute training/evaluation loops.**

### Training Loop Pseudocode
```python
def train_loop(config):
    # Initialize modules
    algo = ALGO_REGISTRY[config.algo](config.algo_config)
    dataset = DATASET_REGISTRY[config.dataset](config.data_config)
    engine = KiRL(model_factory, fsdp_config, context)
    
    # Prepare data
    train_data = dataset.load_raw_data("train")
    local_data = dataset.get_shard(train_data, rank, world_size)
    
    for step in range(num_steps):
        # 1. Sample prompts
        prompts = sample_batch(local_data, batch_size)
        
        # 2. Generate rollouts (inference mode)
        rollout = engine.generate(prompts, gen_config)
        
        # 3. Compute rewards (external reward model)
        rewards = reward_fn(rollout.responses, prompts)
        
        # 4. Algorithm computes advantages and loss
        advantages = algo.compute_advantages(
            RolloutData(rewards, rollout.log_probs, ...), 
            config.algo_config
        )
        loss, metrics = algo.compute_loss(
            advantages, 
            rollout.log_probs, 
            rollout.old_log_probs,
            config.algo_config
        )
        
        # 5. Engine performs backward and optimizer step
        engine.backward(loss)
        if (step + 1) % grad_accum_steps == 0:
            engine.step()
        
        # 6. Logging and checkpointing
        if step % eval_interval == 0:
            metrics = evaluate(engine, dataset, algo, config)
            log_metrics(metrics)
            engine.save_checkpoint(f"step_{step}.pt", metrics)
```

### Inference/Evaluation Loop
```python
def evaluate(engine, dataset, algo, config):
    eval_data = dataset.load_raw_data("validation")
    local_data = dataset.get_shard(eval_data, rank, world_size)
    
    all_rewards = []
    for batch in batches(local_data):
        rollout = engine.generate(batch, gen_config)
        rewards = reward_fn(rollout.responses, batch)
        all_rewards.extend(rewards)
    
    # Aggregate across ranks
    mean_reward = context.all_reduce(sum(all_rewards)) / len(eval_data)
    return {"mean_reward": mean_reward}
```

---

## Multi-GPU Scheduling

### Data Parallelism
- Each rank processes a different shard of the dataset
- Sharding done by `dataset.get_shard(rank, world_size)`
- All ranks must have same local batch size (padding handled in `get_shard`)

### Model Parallelism (FSDP)
- Model parameters sharded across GPUs
- Forward: all-gather parameters, compute, discard
- Backward: all-gather parameters, compute grads, reduce-scatter
- Optimizer step: each rank updates its shard

### Communication Pattern
```
Rank 0: data[0::4]  ──┐
Rank 1: data[1::4]  ──┼─→ FSDP forward/backward ──→ all-reduce gradients
Rank 2: data[2::4]  ──┤
Rank 3: data[3::4]  ──┘
```

### Key Implementation Notes
1. **NCCL backend required** for efficient GPU-GPU communication
2. **Determinism**: Set `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(True)`
3. **Gradient accumulation**: Only sync gradients every N steps to reduce communication overhead
4. **Mixed precision**: Use `torch.autocast` with bfloat16 for forward/backward, keep optimizer in fp32

---

## Critical Design Choices

### 1. Why Registry Pattern Over Inheritance-Only?
- **Pro**: Easy to add new algorithms without modifying engine code
- **Pro**: Clear separation of concerns (algorithm logic vs. infrastructure)
- **Con**: Slightly more boilerplate (decorators)

**Decision**: Registry + Base Class (best of both worlds)

### 2. Why Dataset Handles Sharding?
- **Pro**: Some datasets need custom sharding (e.g., grouped by task)
- **Pro**: Engine stays simple (just calls `get_shard`)
- **Con**: Every dataset must implement `get_shard`

**Decision**: Provide default implementation in `BaseDataset`, allow override

### 3. Why Separate `compute_advantages` and `compute_loss`?
- **Pro**: Some algorithms (GRPO) compute advantages offline, others (PPO) inline
- **Pro**: Easier to debug (can inspect advantages before loss)
- **Con**: Two methods to implement per algorithm

**Decision**: Keep separate for flexibility

### 4. Why Not Use Existing Libraries (e.g., Accelerate, DeepSpeed)?
- **Pro**: Would reduce boilerplate
- **Con**: Hides implementation details (defeats educational purpose)

**Decision**: Implement FSDP wrapper manually, reference verl/Transformers for best practices

---

## Optimization Opportunities (Educational Focus)

After building the baseline, KiRL should make it easy to explore:

1. **Gradient Accumulation Scheduling**
   - How many microbatches before sync?
   - Trade-off: memory vs. communication overhead

2. **Mixed Precision Strategies**
   - Autocast vs. manual casting
   - When to upcast to fp32 (e.g., loss computation, normalization)

3. **Checkpoint Strategies**
   - Gradient checkpointing: memory vs. recomputation
   - Selective activation checkpointing (transformer blocks only)

4. **Communication Overlap**
   - Overlap all-reduce with backward pass
   - Bucket gradients for fewer communication calls

5. **Dataset Preprocessing**
   - Pre-tokenize vs. tokenize-on-the-fly
   - Caching strategies for large datasets

6. **Generation Optimization**
   - vLLM integration for faster rollout generation
   - Separate generation pool vs. inline generation

---

## Constraints

1. **Python 3.10+** (for match/case, better type hints)
2. **PyTorch 2.0+** (for FSDP2, torch.compile support)
3. **Single-node multi-GPU focus** (8 GPUs max)
4. **Hydra for configuration** (match verl conventions)
5. **No custom CUDA kernels** (keep implementation readable)

---

## Success Criteria

KiRL is successful if:
- [ ] A new algorithm can be added in <100 lines (just the algo class)
- [ ] A new dataset can be added in <150 lines (just the dataset class)
- [ ] Training loop code is <300 lines (pipeline orchestration)
- [ ] Switching from DDP to FSDP requires changing 1 config line
- [ ] Multi-GPU training matches single-GPU results (determinism check)
- [ ] Code is readable by someone learning RL systems (extensive comments)

---

## References

- **verl**: Main reference for algorithm implementations and best practices
- **assignment5/fsdp_grpo.py**: Reference for FSDP wrapper and data sharding
- **PyTorch FSDP tutorial**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **Transformers Trainer**: Reference for checkpoint management and logging
