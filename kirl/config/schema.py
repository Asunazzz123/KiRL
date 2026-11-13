"""Serializable experiment schema. No model or distributed runtime is created here."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class AlgoSchema:
    name: str = "grpo"
    group_size: int = 4
    beta: float = 0.1
    cliprange: float = 0.2
    loss_normalization: str = "token"
    advantage_normalization: str = "group"


@dataclass
class DatasetSchema:
    name: str = "gsm8k"
    train_path: str = MISSING
    eval_path: str | None = None
    prompt_template_path: str | None = None
    max_samples: int | None = None
    shuffle: bool = True
    seed: int = 42
    difficulty_filter: int | None = None
    subject_filter: str | None = None


@dataclass
class EngineSchema:
    name: str = "fsdp"
    model_id: str = MISSING
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"
    sharding_strategy: str | None = None
    cpu_offload: bool = False
    gradient_checkpointing: bool = False
    find_unused_parameters: bool = False
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100


@dataclass
class GenerationSchema:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = -1
    do_sample: bool = True
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False


@dataclass
class RolloutSchema:
    backend: str = "vllm"
    gpu_memory_utilization: float = MISSING
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enable_sleep_mode: bool = False
    enforce_eager: bool = False
    trust_remote_code: bool = False


@dataclass
class RewardSchema:
    type: str = "gsm8k_reward"
    format_weight: float = 0.1


@dataclass
class TrainSchema:
    algo: AlgoSchema = field(default_factory=AlgoSchema)
    dataset: DatasetSchema = field(default_factory=DatasetSchema)
    engine: EngineSchema = field(default_factory=EngineSchema)
    generation: GenerationSchema = field(default_factory=GenerationSchema)
    rollout: RolloutSchema = field(default_factory=RolloutSchema)
    reward: RewardSchema = field(default_factory=RewardSchema)
    # Validated against the existing profiler dataclasses by the adapter.
    profiler: dict[str, Any] = field(default_factory=dict)
    num_steps: int = 1000
    batch_size: int = 8  # Training sequences per device, per microbatch.
    gradient_accumulation_steps: int = 4
    prompts_per_batch: int = 32  # Global prompt count per rollout.
    log_interval: int = 10
    eval_interval: int = 100
    eval_examples: int = 500
    output_dir: str = "./outputs"
    save_interval: int = 500
    best_checkpoint_dir: str = "checkpoints/best"  # Relative to this run's output directory.
    resume_from_checkpoint: str | None = None
    seed: int = 42
    deterministic: bool = True


def register_configs() -> None:
    """Register before Hydra composes the defaults list."""
    ConfigStore.instance().store(name="train_schema", node=TrainSchema)
