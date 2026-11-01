# TinyEngine Implementation TODO

## Project Setup

- [ ] Initialize project structure
  ```
  TinyEngine/
  ├── tiny_engine/
  │   ├── algorithms/         # Algorithm module
  │   ├── datasets/           # Dataset module
  │   ├── engine/             # Engine core
  │   ├── pipeline/           # Training/inference pipeline
  │   ├── distributed/        # Distributed utilities
  │   └── utils/              # Logging, config, helpers
  ├── configs/                # Hydra configs
  ├── examples/               # Example training scripts
  ├── tests/                  # Unit tests
  ├── ARCHITECTURE.md
  ├── TODO.md
  └── README.md
  ```

- [ ] Set up package dependencies
  - PyTorch >= 2.0
  - Hydra-core
  - transformers
  - datasets (HuggingFace)
  - wandb (optional logging)

- [ ] Configure Hydra config structure
  - `configs/algo/grpo.yaml`, `configs/algo/ppo.yaml`
  - `configs/dataset/gsm8k.yaml`, `configs/dataset/math.yaml`
  - `configs/engine/fsdp.yaml`, `configs/engine/ddp.yaml`
  - `configs/train.yaml` (main config)

---

## Phase 1: Distributed Context and Engine Core

### 1.1 Distributed Context (Priority: HIGH)

**File**: `tiny_engine/distributed/context.py`

- [ ] Implement `DistributedContext` class
  ```python
  class DistributedContext:
      rank: int
      local_rank: int
      world_size: int
      device: torch.device
      
      @classmethod
      def from_torchrun(cls) -> DistributedContext
      
      def barrier(self)
      def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor
      def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]
      def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor
      def root_call(self, fn: Callable) -> Any
      def agree(self, key: str, value: Any)  # Sanity check across ranks
  ```

- [ ] Add determinism utilities
  - Set CUDA seeds
  - Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - Set `PYTHONHASHSEED`

**Testing**:
- [ ] Test with `torchrun --nproc_per_node=2` on single node
- [ ] Verify all_reduce/all_gather correctness
- [ ] Test rank-0-only operations (`root_call`)

### 1.2 FSDP Configuration (Priority: HIGH)

**File**: `tiny_engine/distributed/fsdp_config.py`

- [ ] Implement `FSDPConfig` dataclass
  ```python
  @dataclass
  class FSDPConfig:
      compute_dtype: torch.dtype = torch.bfloat16
      param_dtype: torch.dtype = torch.float32
      learning_rate: float = 1e-5
      weight_decay: float = 0.0
      max_grad_norm: float = 1.0
      gradient_checkpointing: bool = False
      sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD
      cpu_offload: bool = False
  ```

- [ ] Add FSDP wrapping utilities
  - Auto-wrap policy for transformer blocks
  - Mixed precision policy

### 1.3 TinyEngine Core (Priority: HIGH)

**File**: `tiny_engine/engine/tiny_engine.py`

- [ ] Implement model initialization
  ```python
  def _init_model(
      self, 
      model_factory: Callable, 
      fsdp_config: FSDPConfig
  ) -> nn.Module:
      # Load model
      # Apply FSDP wrapping
      # Enable gradient checkpointing if requested
      ...
  ```

- [ ] Implement forward/backward
  ```python
  def forward(
      self, 
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      labels: Optional[torch.Tensor] = None
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
      # Forward pass with autocast
      # Return (logits, loss) if labels provided
      ...
  
  def backward(self, loss: torch.Tensor):
      # Backward with gradient accumulation handling
      ...
  ```

- [ ] Implement optimizer step
  ```python
  def step(self):
      # Unscale gradients (if using GradScaler)
      # Clip gradients
      # Optimizer step
      # Zero grad
      # LR scheduler step
      ...
  ```

- [ ] Implement generation
  ```python
  def generate(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      generation_config: GenerationConfig
  ) -> RolloutResult:
      # Use model.generate() for sampling
      # Return RolloutResult(response_ids, response_mask, log_probs)
      ...
  ```

**Testing**:
- [ ] Single-GPU forward/backward correctness
- [ ] Multi-GPU gradient synchronization (compare with DDP baseline)
- [ ] Generation output determinism (same seed → same output)
- [ ] Memory profiling (gradient checkpointing effect)

### 1.4 Checkpoint Management (Priority: MEDIUM)

**File**: `tiny_engine/engine/checkpoint.py`

- [ ] Implement save/load for FSDP
  ```python
  def save_checkpoint(
      engine: TinyEngine,
      path: Path,
      metadata: dict,
      save_optimizer: bool = True
  ):
      # Use FSDP state_dict with full_state_dict on rank 0
      # Save metadata separately
      ...
  
  def load_checkpoint(
      engine: TinyEngine,
      path: Path
  ) -> dict:
      # Load with FSDP-aware loading
      # Return metadata
      ...
  ```

- [ ] Implement best checkpoint tracker
  ```python
  class BestCheckpointTracker:
      def consider(self, step: int, metric: float, metadata: dict):
          # Save if metric improves
          ...
  ```

**Testing**:
- [ ] Save → load → verify parameter equivalence
- [ ] Multi-GPU: rank-0 saves, all ranks load correctly
- [ ] Optimizer state resumption (verify training continues smoothly)

---

## Phase 2: Algorithm Module

### 2.1 Base Algorithm Interface (Priority: HIGH)

**File**: `tiny_engine/algorithms/base.py`

- [ ] Define `RolloutData` dataclass
  ```python
  @dataclass
  class RolloutData:
      rewards: torch.Tensor          # [batch, seq_len]
      log_probs: torch.Tensor        # [batch, seq_len]
      values: Optional[torch.Tensor] # [batch, seq_len] for PPO
      response_mask: torch.Tensor    # [batch, seq_len]
      old_log_probs: torch.Tensor    # [batch, seq_len]
  ```

- [ ] Define `BaseAlgorithm` interface
  ```python
  class BaseAlgorithm(ABC):
      algo: str = "base"
      
      @abstractmethod
      def compute_advantages(
          self, 
          rollout_data: RolloutData,
          config: DictConfig
      ) -> torch.Tensor:
          ...
      
      @abstractmethod
      def compute_loss(
          self,
          advantages: torch.Tensor,
          log_probs: torch.Tensor,
          old_log_probs: torch.Tensor,
          response_mask: torch.Tensor,
          config: DictConfig
      ) -> tuple[torch.Tensor, dict[str, Any]]:
          ...
  ```

- [ ] Implement algorithm registry
  ```python
  ALGO_REGISTRY: dict[str, type[BaseAlgorithm]] = {}
  
  def register_algorithm(name: str):
      def decorator(cls):
          ALGO_REGISTRY[name] = cls
          return cls
      return decorator
  ```

### 2.2 GRPO Implementation (Priority: HIGH)

**File**: `tiny_engine/algorithms/grpo.py`

- [ ] Implement GRPO advantage estimation
  - Group-based normalization
  - Handle multiple responses per prompt
  - Reference: `verl/verl/trainer/ppo/core_algos.py` (GRPO estimator)

- [ ] Implement GRPO loss
  - Policy gradient with importance weighting
  - Optional clipping (if `cliprange` provided)
  - Return metrics: policy_loss, mean_advantages, clipped_fraction

**Testing**:
- [ ] Verify advantages sum to zero within groups
- [ ] Loss decreases on synthetic data (rewards = [0, 1, 0, 1, ...])
- [ ] Compare with verl GRPO on small-scale test case

### 2.3 PPO Implementation (Priority: MEDIUM)

**File**: `tiny_engine/algorithms/ppo.py`

- [ ] Implement GAE (Generalized Advantage Estimation)
  - Reference: `verl/verl/trainer/ppo/core_algos.py`
  - Compute advantages using value function and rewards

- [ ] Implement clipped PPO loss
  - Ratio clipping
  - Value loss (if critic enabled)
  - Entropy bonus

**Testing**:
- [ ] Verify advantage computation with synthetic returns
- [ ] Loss clipping behavior (ratio should stay within [1-ε, 1+ε])

### 2.4 DPO Implementation (Priority: LOW)

**File**: `tiny_engine/algorithms/dpo.py`

- [ ] Implement DPO loss (preference-based)
  - No advantage computation (direct loss on preferred/rejected pairs)
  - Reference: HuggingFace TRL implementation

---

## Phase 3: Dataset Module

### 3.1 Base Dataset Interface (Priority: HIGH)

**File**: `tiny_engine/datasets/base.py`

- [ ] Define `BaseDataset` interface
  ```python
  class BaseDataset(ABC):
      name: str = "base"
      
      @abstractmethod
      def load_raw_data(self, split: str) -> list[dict]:
          ...
      
      @abstractmethod
      def parse_sample(self, raw_sample: dict) -> dict[str, Any]:
          ...
      
      def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
          # Default tokenization + padding
          ...
      
      def get_shard(
          self, 
          data: list[dict], 
          rank: int, 
          world_size: int
      ) -> list[dict]:
          # Default: rank-based round-robin
          indices = list(range(rank, len(data), world_size))
          # Pad to ensure all ranks have same count
          count = math.ceil(len(data) / world_size)
          indices += [0] * (count - len(indices))
          return [data[i] for i in indices]
  ```

- [ ] Implement dataset registry
  ```python
  DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}
  
  def register_dataset(name: str):
      ...
  ```

### 3.2 GSM8K Dataset (Priority: HIGH)

**File**: `tiny_engine/datasets/gsm8k.py`

- [ ] Load GSM8K from HuggingFace or local JSON
- [ ] Parse format: `{"question": "...", "answer": "..."}`
- [ ] Implement prompt template loading
  - Support configurable templates

**Testing**:
- [ ] Verify data loading on train/test splits
- [ ] Check sharding: all ranks receive disjoint subsets
- [ ] Tokenization output shape correctness

### 3.3 MATH Dataset (Priority: MEDIUM)

**File**: `tiny_engine/datasets/math.py`

- [ ] Load MATH dataset (HuggingFace)
- [ ] Parse problem + solution format
- [ ] Handle multiple difficulty levels

### 3.4 Custom Dataset Support (Priority: LOW)

**File**: `tiny_engine/datasets/custom.py`

- [ ] Generic dataset loader for user-provided JSONL files
- [ ] Configurable field mapping (e.g., "question" → "input", "answer" → "target")

---

## Phase 4: Training Pipeline

### 4.1 Training Loop (Priority: HIGH)

**File**: `tiny_engine/pipeline/train.py`

- [ ] Implement main training loop
  ```python
  def train_loop(config: DictConfig):
      # 1. Initialize distributed context
      ctx = DistributedContext.from_torchrun()
      
      # 2. Load modules
      algo = ALGO_REGISTRY[config.algo.name](config.algo)
      dataset = DATASET_REGISTRY[config.dataset.name](config.dataset)
      engine = TinyEngine(model_factory, config.engine, ctx)
      
      # 3. Prepare data
      train_data = dataset.load_raw_data("train")
      local_data = dataset.get_shard(train_data, ctx.rank, ctx.world_size)
      
      # 4. Training loop
      for step in range(config.num_steps):
          # Sample batch
          batch = sample_batch(local_data, config.batch_size)
          
          # Generate rollouts
          rollout = engine.generate(batch, config.generation)
          
          # Compute rewards (external reward function)
          rewards = compute_rewards(rollout, batch, config.reward)
          
          # Algorithm: advantages + loss
          rollout_data = RolloutData(rewards, rollout.log_probs, ...)
          advantages = algo.compute_advantages(rollout_data, config.algo)
          loss, metrics = algo.compute_loss(advantages, ...)
          
          # Backward + step
          engine.backward(loss)
          if (step + 1) % config.gradient_accumulation_steps == 0:
              engine.step()
          
          # Logging
          if step % config.log_interval == 0:
              log_metrics(metrics, step)
          
          # Evaluation
          if step % config.eval_interval == 0:
              eval_metrics = evaluate(engine, dataset, algo, config)
              save_checkpoint(engine, step, eval_metrics)
  ```

**Testing**:
- [ ] Single-GPU: full training run on 100 samples
- [ ] Multi-GPU: verify loss curves match single-GPU (determinism)
- [ ] Gradient accumulation: same final loss with different accumulation steps

### 4.2 Evaluation Loop (Priority: HIGH)

**File**: `tiny_engine/pipeline/evaluate.py`

- [ ] Implement evaluation loop
  ```python
  def evaluate(
      engine: TinyEngine,
      dataset: BaseDataset,
      config: DictConfig
  ) -> dict[str, float]:
      eval_data = dataset.load_raw_data("validation")
      local_data = dataset.get_shard(eval_data, rank, world_size)
      
      all_rewards = []
      for batch in batches(local_data):
          rollout = engine.generate(batch, gen_config)
          rewards = compute_rewards(rollout, batch, config.reward)
          all_rewards.extend(rewards)
      
      # Aggregate across ranks
      total_reward = ctx.all_reduce(torch.tensor(sum(all_rewards)))
      mean_reward = total_reward / len(eval_data)
      
      return {"mean_reward": mean_reward.item()}
  ```

**Testing**:
- [ ] Verify metric aggregation across ranks
- [ ] Check for data leakage (eval data not in train)

### 4.3 Reward Function Module (Priority: MEDIUM)

**File**: `tiny_engine/pipeline/rewards.py`

- [ ] Implement reward function interface
  ```python
  class RewardFunction(ABC):
      @abstractmethod
      def compute(
          self, 
          response: str, 
          ground_truth: Any
      ) -> dict[str, float]:
          # Return {"reward": float, "answer_reward": float, ...}
          ...
  ```

- [ ] Implement GSM8K reward function
  - Extract boxed answer
  - Compare with ground truth
  - Format reward (0/1 based on format validity)

- [ ] Support external reward models (optional)
  - Load HF reward model
  - Batch scoring for efficiency

---

## Phase 5: Configuration and Utilities

### 5.1 Hydra Configuration (Priority: MEDIUM)

**Files**: `configs/*.yaml`

- [ ] Define base config structure
  ```yaml
  # configs/train.yaml
  defaults:
    - algo: grpo
    - dataset: gsm8k
    - engine: fsdp
  
  num_steps: 1000
  batch_size: 8
  gradient_accumulation_steps: 4
  log_interval: 10
  eval_interval: 100
  seed: 42
  output_dir: ./outputs
  ```

- [ ] Create algorithm configs
  - `configs/algo/grpo.yaml`: `beta`, `temperature`, etc.
  - `configs/algo/ppo.yaml`: `clip_ratio`, `gae_lambda`, etc.

- [ ] Create dataset configs
  - `configs/dataset/gsm8k.yaml`: paths, templates
  - `configs/dataset/math.yaml`: difficulty filter, etc.

- [ ] Create engine configs
  - `configs/engine/fsdp.yaml`: FSDP settings
  - `configs/engine/ddp.yaml`: DDP settings

### 5.2 Logging and Metrics (Priority: MEDIUM)

**File**: `tiny_engine/utils/logging.py`

- [ ] Implement metric logger
  - Console logging (rank 0 only)
  - JSON logging (append to `metrics.jsonl`)
  - Optional: WandB integration

- [ ] Track key metrics
  - Loss, advantages, KL divergence
  - Generation stats (response length, temperature)
  - Training speed (samples/sec, tokens/sec)
  - GPU memory usage

### 5.3 Utilities (Priority: LOW)

**File**: `tiny_engine/utils/helpers.py`

- [ ] Sampling utilities
  - `sample_batch(data, batch_size)` with optional replacement
  - `sample_temperature_schedule(step, total_steps)` for annealing

- [ ] Tokenization helpers
  - Batch tokenization with padding
  - Extract generated tokens from full sequence

- [ ] Gradient analysis tools
  - Gradient norm logging
  - Per-layer gradient statistics

---

## Phase 6: Testing and Validation

### 6.1 Unit Tests (Priority: MEDIUM)

**Directory**: `tests/`

- [ ] Test distributed context
  - Mock multi-rank environment
  - Verify all_reduce correctness

- [ ] Test algorithms
  - GRPO: group normalization, loss computation
  - PPO: GAE computation, clipping behavior

- [ ] Test datasets
  - Sharding correctness (no overlap, full coverage)
  - Tokenization output shapes

- [ ] Test engine
  - Forward/backward numerical correctness
  - Generation determinism

### 6.2 Integration Tests (Priority: HIGH)

- [ ] End-to-end training on toy dataset (10 samples)
  - Verify loss decreases
  - Check checkpoint save/load
  - Validate multi-GPU equivalence

- [ ] Reproduction test: GRPO on GSM8K subset
  - Compare final metrics with verl baseline
  - Verify determinism (same seed → same result)

### 6.3 Performance Benchmarks (Priority: LOW)

- [ ] Measure throughput (samples/sec) on 1/2/4/8 GPUs
- [ ] Compare FSDP vs DDP training speed
- [ ] Profile memory usage with/without gradient checkpointing

---

## Phase 7: Documentation and Examples

### 7.1 README (Priority: HIGH)

**File**: `README.md`

- [ ] Project overview and design goals
- [ ] Installation instructions
- [ ] Quick start example
  ```bash
  torchrun --nproc_per_node=2 examples/train_grpo_gsm8k.py
  ```
- [ ] Configuration guide (how to add new algo/dataset)
- [ ] Architecture reference (link to ARCHITECTURE.md)

### 7.2 Example Scripts (Priority: MEDIUM)

**Directory**: `examples/`

- [ ] `train_grpo_gsm8k.py`: Full GRPO training script
- [ ] `train_ppo_math.py`: PPO training example
- [ ] `evaluate_model.py`: Standalone evaluation script
- [ ] `export_checkpoint.py`: Convert FSDP checkpoint to HF format

### 7.3 Inline Documentation (Priority: MEDIUM)

- [ ] Add docstrings to all public methods
- [ ] Comment complex sections (e.g., FSDP wrapping, gradient sync)
- [ ] Explain design choices in comments (why not use library X)

---

## Critical Notes for Implementation

### Multi-GPU Synchronization Points
1. **Generation**: All ranks must generate same number of batches (use `ctx.agree()`)
2. **Loss computation**: Reduce loss across ranks before backward
3. **Evaluation**: Aggregate metrics with `all_reduce` before logging
4. **Checkpointing**: Only rank 0 writes, but all ranks must call `save_checkpoint`

### Determinism Requirements
- Set all seeds before model init: `torch.manual_seed(seed + rank)`
- Use `torch.use_deterministic_algorithms(True)`
- For generation: seed = `base_seed + rollout_id * 1009 + rank * 104729`
- Disable dropout during RL training (inference mode for policy)

### Memory Optimization Strategies
1. **Gradient Checkpointing**: Trade recomputation for memory
   - Enable for layers with large activations (transformer blocks)
   - Cost: ~30% slower training, ~50% less memory
2. **Mixed Precision**: Use bfloat16 for forward/backward, fp32 for optimizer
   - Avoid loss scaling (bfloat16 has better range than fp16)
3. **Gradient Accumulation**: Reduce memory by processing smaller microbatches
   - Only sync gradients every N steps
4. **FSDP CPU Offload**: Move unused parameters to CPU (last resort, very slow)

### Common Pitfalls to Avoid
1. **Uneven Batch Sizes**: All ranks must process same batch size, pad if needed
2. **Hanging on Collectives**: Ensure all ranks call `all_reduce`/`barrier` at same points
3. **OOM During Generation**: Use smaller generation batch size than training batch size
4. **KL Divergence Explosion**: Clip PPO ratio or use adaptive KL penalty
5. **Reward Hacking**: Normalize rewards per-prompt or use advantage normalization

### Reference Implementations
- **FSDP Setup**: CS336 `assignment5/training/backends/fsdp_grpo.py`
- **GRPO Algorithm**: verl `verl/trainer/ppo/core_algos.py` (GRPO estimator)
- **Data Sharding**: CS336 `fsdp_grpo.py:69-73` (rank-based sharding)
- **Checkpoint**: verl `verl/checkpoint_engine/base.py`

---

## Milestone Checklist

### Milestone 1: Minimal Working Engine (Week 1-2)
- [x] ARCHITECTURE.md completed
- [x] TODO.md completed
- [ ] DistributedContext + TinyEngine (forward/backward/step only)
- [ ] Single algorithm (GRPO) + single dataset (GSM8K)
- [ ] Single-GPU training loop (no FSDP yet)
- [ ] **Validation**: Train on 100 GSM8K samples, loss decreases

### Milestone 2: Multi-GPU Training (Week 3-4)
- [ ] FSDP integration
- [ ] Multi-GPU data sharding
- [ ] Checkpoint save/load
- [ ] **Validation**: 2-GPU training matches 1-GPU loss curve

### Milestone 3: Algorithm Flexibility (Week 5-6)
- [ ] PPO implementation
- [ ] Algorithm registry fully functional
- [ ] Easy switching via config (`algo: grpo` → `algo: ppo`)
- [ ] **Validation**: Both GRPO and PPO converge on GSM8K

### Milestone 4: Dataset Flexibility (Week 7-8)
- [ ] MATH dataset support
- [ ] Dataset registry fully functional
- [ ] Custom dataset loader
- [ ] **Validation**: Train on MATH with minimal config changes

### Milestone 5: Production-Ready (Week 9-10)
- [ ] Full test coverage (unit + integration)
- [ ] Documentation complete
- [ ] Example scripts working
- [ ] Performance benchmarks documented
- [ ] **Validation**: Reproduce verl GRPO results on GSM8K subset

---

## Priority Summary

**Must Have (P0)**:
- Distributed context and FSDP engine
- GRPO algorithm
- GSM8K dataset
- Basic training pipeline
- Checkpoint management

**Should Have (P1)**:
- PPO algorithm
- Multi-dataset support
- Comprehensive logging
- Unit tests

**Nice to Have (P2)**:
- DPO algorithm
- Custom dataset loader
- WandB integration
- Performance profiling tools
- vLLM integration for faster generation

---

## Post-Implementation: Learning Exercises

After completing the baseline, use TinyEngine to explore:

1. **Gradient Accumulation Trade-offs**
   - Run with 1/2/4/8 accumulation steps
   - Measure: throughput, memory, convergence speed
   - Question: When does accumulation hurt convergence?

2. **FSDP Sharding Strategies**
   - Compare FULL_SHARD vs SHARD_GRAD_OP vs NO_SHARD
   - Measure: memory, communication time, throughput
   - Question: What's the optimal strategy for 7B models on 8 GPUs?

3. **Generation Backend Comparison**
   - Baseline: HF `model.generate()`
   - Alternative: Integrate vLLM for generation
   - Measure: tokens/sec, memory overhead
   - Question: When is vLLM worth the complexity?

4. **Advantage Normalization**
   - GRPO: per-group normalization
   - PPO: global normalization vs per-prompt
   - Measure: final accuracy, training stability
   - Question: Which normalization works best for code generation tasks?

5. **Reward Shaping**
   - Baseline: binary correctness reward
   - Alternative: partial credit, format bonus, length penalty
   - Measure: accuracy, overfitting to format
   - Question: How much does reward design matter vs algorithm choice?
