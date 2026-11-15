"""vLLM-specific backend and profiler configuration."""

from dataclasses import dataclass

from ...profiler.vllm_profiler import VLLMProfilerConfig  # re-exported; listed in __all__
from .base import GenerationConfig  # re-exported for backward compatibility

__all__ = ["VLLMBackendConfig", "GenerationConfig", "VLLMProfilerConfig"]


@dataclass(frozen=True)
class VLLMBackendConfig:
    model: str
    gpu_memory_utilization: float  # 根据训推部署显式设置

    tokenizer: str | None = None
    revision: str | None = None
    dtype: str = "auto"

    tensor_parallel_size: int = 1
    max_model_len: int | None = None

    seed: int = 0
    trust_remote_code: bool = False

    enable_sleep_mode: bool = False
    enforce_eager: bool = False

    # 性能调优，可暂时沿用 vLLM 默认值
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None



