"""vllm inference backend"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class GenerationConfig:
    num_return_sequences: int = 1
    max_new_tokens: int = 256

    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1

    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    seed: int | None = None
    return_logprobs: bool = False


class VLLMProfilerConfig:
    ...


