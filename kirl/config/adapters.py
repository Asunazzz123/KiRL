"""Validate an experiment and adapt it without loading models or initializing devices."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from kirl.engine.inference.base import GenerationConfig
from kirl.engine.inference.vllm_config import VLLMBackendConfig
from kirl.profiler.settings import ProfilerSettings

from .schema import TrainSchema

# ---------------------------------------------------------------------------
# Module registry — single source of truth for supported pluggable names.
# To extend the system, add a name to config/registry.yaml and write the
# corresponding implementation; nothing in this file needs to change.
# ---------------------------------------------------------------------------
_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"
_REG: dict[str, list[str]] = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TrainConfig:
    """Resolved experiment plus parameters for the future training pipeline.

    training_args are constructor kwargs, not a TrainingArguments instance:
    constructing that instance belongs to the runtime/device initialization phase.
    """

    resolved: dict[str, Any]
    training_args: dict[str, Any]
    algo_config: dict[str, Any]
    generation_config: dict[str, Any]
    generation: GenerationConfig
    rollout: VLLMBackendConfig
    profiler: ProfilerSettings


def _number(name: str, value: Any, *, minimum: float = 0, inclusive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (not inclusive and value == minimum):
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum}")


def _choice(name: str, value: Any, choices: tuple) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _path(value: str, base_dir: Path) -> str:
    if not value.strip():
        raise ValueError("Configured paths must not be empty")
    path = Path(value).expanduser()
    return str((base_dir / path if not path.is_absolute() else path).resolve())


def resolve_train_config(
    cfg: DictConfig | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainConfig:
    """
    Resolve CLI/YAML values; input paths are relative to base_dir (default cwd).
    """
    source = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(dict(cfg))
    validated = OmegaConf.merge(OmegaConf.structured(TrainSchema), source)
    values = OmegaConf.to_container(validated, resolve=True, throw_on_missing=True)
    algo, engine = values["algo"], values["engine"]
    generation, rollout = values["generation"], values["rollout"]
    dataset = values["dataset"]

    # 新增算法组件时，在这里设置
    _choice("algo.name", algo["name"], tuple(_REG["algo"]))
    _choice("engine.name", engine["name"], tuple(_REG["engine"]))
    _choice("dataset.name", dataset["name"], tuple(_REG["dataset"]))
    _choice("rollout.backend", rollout["backend"], tuple(_REG["rollout"]))

    
    for name in ("num_steps", "batch_size", "gradient_accumulation_steps", "prompts_per_batch",
                 "log_interval", "eval_interval", "eval_examples", "save_interval"):
        _number(name, values[name])
    _number("seed", values["seed"], inclusive=True)
    _number("dataset.seed", dataset["seed"], inclusive=True)
    _number("algo.group_size", algo["group_size"], minimum=2, inclusive=True)
    _number("algo.beta", algo["beta"], inclusive=True)
    _number("algo.cliprange", algo["cliprange"], inclusive=True)
    if algo["cliprange"] >= 1:
        raise ValueError("algo.cliprange must be < 1")
    _choice("algo.loss_normalization", algo["loss_normalization"], ("token", "sequence"))
    _choice("algo.advantage_normalization", algo["advantage_normalization"], ("group",))
    _number("generation.max_new_tokens", generation["max_new_tokens"])
    _number("generation.temperature", generation["temperature"])
    _number("generation.top_p", generation["top_p"])
    if generation["top_p"] > 1:
        raise ValueError("generation.top_p must be <= 1")
    if generation["top_k"] != -1:
        _number("generation.top_k", generation["top_k"])
    for token_id in generation["stop_token_ids"]:
        _number("generation.stop_token_ids", token_id, inclusive=True)
    for name in ("learning_rate", "max_grad_norm"):
        _number(f"engine.{name}", engine[name])
    for name in ("weight_decay", "warmup_steps"):
        _number(f"engine.{name}", engine[name], inclusive=True)
    for name in ("compute_dtype", "param_dtype"):
        _choice(f"engine.{name}", engine[name], ("float16", "bfloat16", "float32"))
    if not engine["model_id"].strip():
        raise ValueError("engine.model_id must not be empty")
    _choice("engine.optimizer", engine["optimizer"], ("adamw", "sgd"))
    _choice("engine.lr_scheduler", engine["lr_scheduler"], ("cosine", "linear", "constant"))
    if engine["name"] == "fsdp":
        _choice("engine.sharding_strategy", engine["sharding_strategy"],
                ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"))
    elif engine["cpu_offload"] or engine["sharding_strategy"] is not None:
        raise ValueError("DDP cannot use FSDP sharding_strategy or cpu_offload")
    _number("rollout.gpu_memory_utilization", rollout["gpu_memory_utilization"])
    if rollout["gpu_memory_utilization"] > 1:
        raise ValueError("rollout.gpu_memory_utilization must be <= 1")
    _number("rollout.tensor_parallel_size", rollout["tensor_parallel_size"])
    for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if rollout[name] is not None:
            _number(f"rollout.{name}", rollout[name])
    if rollout["max_model_len"] is not None and generation["max_new_tokens"] >= rollout["max_model_len"]:
        raise ValueError("rollout.max_model_len must leave room for prompt tokens")
    if dataset["max_samples"] is not None:
        _number("dataset.max_samples", dataset["max_samples"])
    if dataset["difficulty_filter"] is not None:
        _choice("dataset.difficulty_filter", dataset["difficulty_filter"], (1, 2, 3, 4, 5))
    _choice("reward.type", values["reward"]["type"], tuple(_REG["reward"]))
    _number("reward.format_weight", values["reward"]["format_weight"], inclusive=True)

    root = Path(base_dir or Path.cwd()).expanduser().resolve()
    for name in ("train_path", "eval_path", "prompt_template_path"):
        if dataset[name] is not None:
            dataset[name] = _path(dataset[name], root)
    if values["resume_from_checkpoint"] is not None:
        values["resume_from_checkpoint"] = _path(values["resume_from_checkpoint"], root)
    values["output_dir"] = _path(str(output_dir) if output_dir is not None else values["output_dir"], root)
    run_dir = Path(values["output_dir"])
    # Relative output children use the run directory; explicit absolute paths survive.
    values["best_checkpoint_dir"] = _path(values["best_checkpoint_dir"], run_dir)
    profiler_mapping = values["profiler"]
    global_profiler = profiler_mapping.setdefault("global", {})
    global_profiler["output_dir"] = _path(global_profiler.get("output_dir", "profile"), run_dir)
    vllm_profiler = profiler_mapping.get("vllm", {})
    if vllm_profiler.get("torch_profiler_dir"):
        vllm_profiler["torch_profiler_dir"] = _path(vllm_profiler["torch_profiler_dir"], root)
    profiler = ProfilerSettings.from_mapping(profiler_mapping)

    # Leave TrainingArguments construction to the pipeline, including device setup.
    training_args = {
        "output_dir": values["output_dir"], "max_steps": values["num_steps"],
        "per_device_train_batch_size": values["batch_size"],
        "gradient_accumulation_steps": values["gradient_accumulation_steps"],
        "learning_rate": engine["learning_rate"], "weight_decay": engine["weight_decay"],
        "max_grad_norm": engine["max_grad_norm"], "warmup_steps": engine["warmup_steps"],
        "lr_scheduler_type": engine["lr_scheduler"],
        "optim": {"adamw": "adamw_torch", "sgd": "sgd"}[engine["optimizer"]],
        "gradient_checkpointing": engine["gradient_checkpointing"],
        "bf16": engine["compute_dtype"] == "bfloat16", "fp16": engine["compute_dtype"] == "float16",
        "logging_steps": values["log_interval"], "eval_steps": values["eval_interval"],
        "save_steps": values["save_interval"], "seed": values["seed"],
        "data_seed": dataset["seed"], "full_determinism": values["deterministic"],
    }
    if engine["name"] == "fsdp":
        strategy = {"FULL_SHARD": "full_shard", "SHARD_GRAD_OP": "shard_grad_op", "NO_SHARD": "no_shard"}
        training_args["fsdp"] = strategy[engine["sharding_strategy"]] + (" offload" if engine["cpu_offload"] else "")
    else:
        training_args["ddp_find_unused_parameters"] = engine["find_unused_parameters"]
    generation_config = dict(generation)
    generation_config["stop"] = tuple(generation_config["stop"])
    generation_config["stop_token_ids"] = tuple(generation_config["stop_token_ids"])
    generation_output = GenerationConfig(
        **generation_config, num_return_sequences=algo["group_size"], seed=values["seed"], return_logprobs=True
    )
    rollout_config = VLLMBackendConfig(
        **{key: value for key, value in rollout.items() if key != "backend"},
        model=engine["model_id"], dtype=engine["compute_dtype"], seed=values["seed"],
    )
    return TrainConfig(values, training_args, dict(algo), generation_config, generation_output, rollout_config, profiler)
