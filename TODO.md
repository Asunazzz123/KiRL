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
  - Separate identical model initialization seeds from rank-specific training/sampling seeds
  - Establish CUDA determinism settings before the relevant runtime is initialized
  - Configure Python hash seeding before interpreter startup when required

- [ ] Define process-group ownership, initialization validation, timeout policy, and cleanup on exit
- [ ] Distinguish rank-0-only local work from operations requiring collective participation

**Testing**:
- [ ] Test with `torchrun --nproc_per_node=2` on single node
- [ ] Verify all_reduce/all_gather correctness
- [ ] Test rank-0-only operations (`root_call`)

### 1.2 FSDP Configuration and Backend (Priority: HIGH)

**Goal**: Establish a dedicated FSDP backend that exposes a small, stable interface to TinyEngine. Keep backend-specific behavior out of the training orchestration code.

**Planned responsibility boundaries** (new paths are planned, not yet implemented):

| Owner | Responsibility |
| --- | --- |
| `distributed/context.py` | Process identity, device binding, process-group lifecycle, and shared communication helpers |
| `distributed/fsdp_config.py` | Configuration values, defaults, and validation; no model construction or distributed initialization |
| `distributed/fsdp_backend.py` (planned) | FSDP1/FSDP2 selection, wrapping, precision policies, gradient synchronization, global clipping, and backend-specific state access |
| `engine/tiny_engine.py` | Model/optimizer ownership and orchestration of initialization, forward, backward, update, and evaluation |
| `engine/checkpoint.py` | Checkpoint contents, persistence, metadata, retention, and recovery, using the backend's state interface |
| `pipeline/` and training entry scripts | Data flow, algorithm loss, training schedule, logging setup, and process cleanup |

Start with one backend module. Separate FSDP1/FSDP2 implementations later only if their complexity warrants it. The backend must not depend on TinyEngine or on a particular RL algorithm.

#### 1.2.1 Configuration Contract

- [ ] Complete the dataclass fields and initialization-time validation
- [ ] Distinguish backend selection from sharding behavior
  - FSDP1: explicitly supported sharding strategies
  - FSDP2: parameter resharding behavior and mesh requirements
  - Reject incompatible fields rather than silently ignoring user input
- [ ] Define model storage dtype, compute dtype, gradient reduction dtype, and relevant buffer dtype semantics
- [ ] Define ownership of optimizer settings separately from FSDP settings, even if initially stored in one configuration object
- [ ] Convert YAML values into validated configuration values at the configuration boundary
- [ ] Define supported PyTorch versions and detect selected-backend capabilities before model allocation
- [ ] Keep importing configuration independent of optional backend APIs
- [ ] Reject requested features that are not implemented; a configuration flag alone does not establish support

#### 1.2.2 Model Preparation and Wrapping

- [ ] Define the backend preparation contract: input model, configuration, distributed context, and selected block types; output the prepared model
- [ ] Document whether preparation mutates or replaces the model and require callers to use the returned model
- [ ] Validate device/backend compatibility before entering distributed model preparation
- [ ] Establish a minimal single-node topology; defer hybrid and multi-dimensional meshes
- [ ] Select Transformer blocks explicitly and report when no supported blocks are found
- [ ] Preserve shared/tied parameter ownership when selecting wrapping boundaries
- [ ] Support FSDP1 wrapping and FSDP2 child-before-parent sharding through the same preparation boundary
- [ ] Guarantee consistent initial parameters and buffers across ranks
  - Separate identical model initialization from rank-specific training randomness
  - Define consistency requirements for random initialization and pretrained loading
- [ ] Define mixed precision and optional gradient checkpointing preparation order
- [ ] Require optimizer construction after model preparation
- [ ] Record the effective backend, precision, wrapping targets, and topology in startup diagnostics

#### 1.2.3 Runtime Services Exposed to TinyEngine

- [ ] Expose gradient synchronization control for accumulation, with a documented all-rank participation contract
- [ ] Expose global gradient norm/clipping behavior appropriate to each backend
- [ ] Define distributed-safe overflow detection and scaling requirements for FP16 training
- [ ] Define how train/eval transitions preserve valid sharding state
- [ ] Expose backend-aware model and optimizer state access for checkpoint save/load
- [ ] Document which operations are collective and must be called in the same order on all participating ranks
- [ ] Keep algorithm loss, data sampling, checkpoint paths, and checkpoint retention outside this module

#### 1.2.4 Baseline Scope and Acceptance

- [ ] Select one backend as the first acceptance target; validate the other independently before advertising dual-backend support
- [ ] Start with a small dense model, a single node, and GPU-resident model/optimizer state
- [ ] Validate FP32 first, then BF16 where supported; accept FP16 only with verified scaling and overflow handling
- [ ] Verify configuration rejection and supported block selection without requiring a GPU
- [ ] On two GPUs, verify successful preparation, consistent initial weights, and a complete forward/backward/update
- [ ] Compare loss, global gradient norm, and one-step parameter updates with a single-device reference using the same global batch and declared tolerances
- [ ] Keep static checks, CPU tests, and real CUDA/NCCL acceptance results distinct

**Deferred extensions**: CPU/activation offload, meta-device loading, hybrid sharding, prefetch tuning, LoRA, quantization, sequence parallelism, and external rollout weight synchronization. Add separate acceptance tasks before enabling any of these features.

### 1.3 TinyEngine Core (Priority: HIGH)

**File**: `tiny_engine/engine/tiny_engine.py`

**Dependency**: Use the backend contract from 1.2. TinyEngine owns training state and invokes backend services; it does not duplicate FSDP wrapping, collective clipping, or state conversion logic.

- [ ] Define model initialization and ownership
  - Load the model, apply requested model preparation, and delegate distributed wrapping
  - Construct optimizer and optional scheduler only after backend preparation
  - Keep model initialization separate from algorithm and dataset construction
- [ ] Define forward/backward contracts
  - Specify accepted batch fields and returned outputs
  - Preserve the computation graph required by the algorithm loss
  - Keep precision behavior consistent with the backend policy
  - Define who normalizes loss across ranks and micro-batches; avoid double normalization
- [ ] Implement gradient accumulation orchestration
  - Define micro-step versus optimizer-step counters
  - Handle incomplete accumulation windows and unequal effective token counts
  - Require compatible collective schedules across ranks
  - Establish correctness with normal synchronization before adding deferred-sync optimization
- [ ] Implement the optimizer update lifecycle
  - Undo scaling before clipping when FP16 scaling is enabled
  - Delegate global clipping to the backend
  - Ensure all ranks agree whether an update proceeds or is skipped for non-finite gradients
  - Define gradient clearing and scheduler advancement relative to successful updates
  - Return update status, gradient norm, and learning rate for logging
- [ ] Define train/eval transitions and forward-only execution
  - Separate module training mode from gradient recording
  - Restore the previous mode and valid backend state after evaluation
- [ ] Define a basic generation contract
  - Specify response token IDs, masks, and the meaning of returned log probabilities
  - Keep gradient-free rollout outputs distinct from differentiable training log probabilities
  - Coordinate generation completion across ranks when forward passes involve collectives
  - Validate compatibility with the selected FSDP backend before using model generation APIs
  - Defer external inference-engine integration to a separate generation/weight-sync boundary

**Testing**:
- [ ] Single-device forward/backward/update correctness on a small deterministic model
- [ ] Two-GPU updates match a single-device or DDP reference for the same global batch within declared tolerances
- [ ] Accumulated micro-batches match the unsplit batch, including unequal effective token counts
- [ ] Non-finite gradients produce consistent skip/fail behavior and scheduler state on all ranks
- [ ] Evaluation does not accumulate gradients and training resumes correctly afterward
- [ ] Generation with different per-rank EOS times completes without collective hangs
- [ ] Record memory effects of gradient checkpointing on actual target hardware

### 1.4 Checkpoint Management (Priority: MEDIUM)

**File**: `tiny_engine/engine/checkpoint.py`

- [ ] Separate resumable training checkpoints from model-only export
- [ ] Define checkpoint contents and ownership
  - Model and optimizer state through the backend interface
  - Scheduler, scaler, successful update count, and accumulation position where applicable
  - Per-rank RNG state and pipeline-provided data progress for reproducible continuation
  - Model/configuration metadata and supported topology/version information
- [ ] Save baseline checkpoints at completed update boundaries; reject unsupported mid-accumulation resume
- [ ] Define collective participation independently from which ranks write files
  - All required ranks participate in state collection/restoration
  - Distinguish full exports from per-rank/distributed checkpoint storage
- [ ] Define compatibility checks for restore; initially require the validated backend and world size
- [ ] Publish a checkpoint as complete only after all required state is saved successfully
- [ ] Preserve the previous valid checkpoint if saving is interrupted
- [ ] Define model-only loading behavior so stale optimizer state is not mistaken for a valid resume
- [ ] Track the best checkpoint using an explicit validation metric, direction, and retention policy

**Testing**:
- [ ] Save and reload on two GPUs; verify model and optimizer state equivalence
- [ ] Compare the next update after restore with uninterrupted training on the same next batch
- [ ] Verify scheduler/scaler, RNG, and data progress recovery when those capabilities are enabled
- [ ] Reject incomplete checkpoints and unsupported topology/configuration changes clearly
- [ ] Verify exported model weights load independently of the training engine

**Implementation order**: configuration validation → backend preparation → one-step engine training → mixed precision/global clipping → gradient accumulation → checkpoint recovery → distributed generation. Each stage requires its own acceptance evidence; adding a configuration option does not mark the corresponding runtime task complete.

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
  - Keep backend-specific options consistent with the validated configuration contract in 1.2
  - Document model storage, compute, and gradient reduction precision independently
  - Expose only runtime features with completed acceptance checks

### 5.2 Logging and Metrics (Priority: MEDIUM)

**File**: `tiny_engine/utils/logging.py`

- [ ] Implement metric logger
  - Configure logging in each training process entry point; modules only acquire named loggers
  - Aggregate routine metrics on rank 0; retain rank-tagged startup/error diagnostics from every rank
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

### Profiler implementation status (2026-11-03)

- [x] Add validated global, Torch, and vLLM profiler settings with a plain-mapping entry point
- [x] Add per-process Probe/NullProbe, rank/step sampling, coarse host timings, and trace windows
- [x] Add a bounded background JSONL sink with drop/error/drain status
- [x] Add a PyTorch adapter and an injected synchronous vLLM client adapter
- [x] Add backend-independent metric/stage/artifact records and offline JSON report generation
- [x] Add local CPU trace and fake-vLLM contract tests; optional OmegaConf integration is dependency-gated
- [x] Add a CPU-only runnable smoke example and profiler integration guide
- [ ] Wire probes into the real training and inference pipeline once those modules exist
- [ ] Validate actual CUDA/NCCL FSDP traces and actual vLLM metrics/traces on the target software versions
- [ ] Measure off/basic/sampled/trace overhead on representative training and inference workloads
- [ ] Add AsyncLLM/HTTP adapters, cross-node artifact collection, and trace-completion verification
- [ ] Add windowed rate/quantile analysis and additional report renderers when required

Local CPU tests and fake engine calls do not establish GPU correctness, real vLLM compatibility, or a performance overhead budget.

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

- [ ] Complete the FSDP acceptance sequence from 1.2–1.4
  - Validate each advertised backend and precision mode independently
  - Use identical global batches and documented numerical tolerances for reference comparisons
  - Verify accumulation, non-finite update handling, and checkpoint continuation
  - Record hardware, software versions, world size, and actual commands/results
  - Do not count configuration-only or mocked tests as CUDA/NCCL acceptance

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
1. **Generation**: Coordinate batch schedules and per-step completion while model execution requires collectives; agreeing on batch counts alone is insufficient.
2. **Loss computation**: Backpropagate each rank's differentiable local contribution with the intended global normalization. Backend gradient synchronization handles aggregation; metric reduction is a separate operation.
3. **Evaluation**: Aggregate metric numerators and denominators across ranks, excluding padded/duplicate examples as appropriate.
4. **Checkpointing**: All ranks required by the state operation participate. File-writing ownership depends on whether the format is full or sharded.
5. **Updates**: Keep collective order, accumulation boundaries, and update/skip decisions consistent across ranks.

### Determinism Requirements
- Initialize the same model weights across ranks, or synchronize a common initialized state before training.
- Treat rank-specific sampling/training randomness separately from model initialization; capture per-rank RNG state for resume.
- Declare deterministic settings and numerical tolerances for the tested hardware/software configuration.
- Define the RL policy's dropout behavior explicitly; evaluation mode and disabled gradient recording are separate decisions.

### Memory Optimization Strategies
1. **Gradient Checkpointing**: Trade recomputation for memory
   - Select supported model regions and verify correctness after preparation
   - Measure the actual throughput/memory trade-off rather than assuming fixed percentages
2. **Mixed Precision**: Use bfloat16 for forward/backward, fp32 for optimizer
   - Validate hardware support and explicitly define parameter/reduction precision
   - FP16 support additionally requires validated gradient scaling and overflow handling
3. **Gradient Accumulation**: Reduce memory by processing smaller microbatches
   - Preserve the intended global loss weighting and optimizer update frequency
   - Deferred synchronization is optional and can increase retained gradient memory
4. **FSDP CPU Offload**: Defer until the GPU-resident baseline passes; validate compatibility with accumulation and measure transfer costs

### Common Pitfalls to Avoid
1. **Uneven Batches**: Keep collective schedules compatible and normalize by actual sample/token counts; exclude artificial padding from loss and metrics
2. **Hanging on Collectives**: Ensure all ranks call `all_reduce`/`barrier` at same points
3. **OOM During Generation**: Use smaller generation batch size than training batch size
4. **KL Divergence Explosion**: Clip PPO ratio or use adaptive KL penalty
5. **Reward Hacking**: Normalize rewards per-prompt or use advantage normalization

### Reference Implementations
- **FSDP Backend Lifecycle**: verl `verl/workers/engine/fsdp/transformer_impl.py`
- **Wrapping and State Utilities**: verl `verl/utils/fsdp_utils.py`
- **Mesh and Strategy Selection**: verl `verl/workers/engine/fsdp/utils.py`
- **GRPO Algorithm**: verl `verl/trainer/ppo/core_algos.py` (GRPO estimator)
- **Checkpoint**: verl `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- Use these as responsibility and correctness references; review version requirements before adapting backend APIs.

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
- [ ] Dedicated FSDP backend connected to TinyEngine through the boundaries in 1.2
- [ ] First backend/precision combination passes small-model one-step acceptance
- [ ] Multi-GPU data sharding
- [ ] Correct gradient accumulation and global clipping
- [ ] Checkpoint save/load and next-update continuation
- [ ] **Validation**: Compare two-GPU loss, gradient norm, and updates with a one-GPU reference using the same global batch and declared tolerances
- [ ] Accept additional backend/precision combinations separately before exposing them as supported

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
