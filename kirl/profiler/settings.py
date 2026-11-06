"""Construct the three configuration layers from a plain/Hydra mapping."""
from dataclasses import dataclass, field

from .basic_profiler import ProfilerConfig
from .fsdp_profiler import TorchProfilerConfig
from .vllm_profiler import VLLMProfilerConfig


@dataclass(frozen=True)
class ProfilerSettings:
    global_config: ProfilerConfig = field(default_factory=ProfilerConfig)
    torch: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)
    vllm: VLLMProfilerConfig = field(default_factory=VLLMProfilerConfig)

    @classmethod
    def from_mapping(cls, mapping):
        unknown = set(mapping) - {"global", "torch", "vllm"}
        if unknown:
            raise ValueError(f"Unknown profiler sections: {sorted(unknown)}")
        global_config = dict(mapping.get("global", {}))
        for name in ("ranks", "trace_iterations"):
            if name in global_config:
                global_config[name] = tuple(global_config[name])
        torch_config = dict(mapping.get("torch", {}))
        if "activities" in torch_config:
            torch_config["activities"] = tuple(torch_config["activities"])
        return cls(ProfilerConfig(**global_config), TorchProfilerConfig(**torch_config),
                   VLLMProfilerConfig(**dict(mapping.get("vllm", {}))))
