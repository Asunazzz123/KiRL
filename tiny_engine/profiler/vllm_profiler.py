"""Optional vLLM integration: inject an existing engine, never load a model here."""
import inspect
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

from .records import MetricRecord, TraceArtifact, cpu_value


@dataclass(frozen=True)
class VLLMProfilerConfig:
    profiler: Literal["torch"] | None = None
    torch_profiler_dir: str = ""
    torch_profiler_with_stack: bool = False
    torch_profiler_record_shapes: bool = False
    torch_profiler_with_memory: bool = False
    ignore_frontend: bool = True
    delay_iterations: int = 0
    max_iterations: int = 10

    def __post_init__(self):
        if self.profiler not in (None, "torch"):
            raise ValueError("This adapter currently supports only the torch vLLM profiler")
        if self.profiler == "torch" and not self.torch_profiler_dir:
            raise ValueError("torch_profiler_dir is required when profiling is enabled")
        for name in ("delay_iterations", "max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("torch_profiler_with_stack", "torch_profiler_record_shapes", "torch_profiler_with_memory", "ignore_frontend"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")

    def to_backend_config(self) -> dict:
        """Pass to LLM(profiler_config=..., disable_log_stats=False) at construction."""
        result = asdict(self)
        if self.torch_profiler_dir:
            result["torch_profiler_dir"] = str(Path(self.torch_profiler_dir).expanduser().resolve())
        return result


vLLMProfilerConfig = VLLMProfilerConfig  # Backward-compatible scaffold name.


class VLLMProfilerAdapter:
    name = "vllm"

    def __init__(self, engine, config: VLLMProfilerConfig):
        self.engine = engine
        self.config = config
        self._active = False
        for name in ("start_profile", "stop_profile"):
            method = getattr(engine, name, None)
            if not callable(method):
                raise TypeError(f"Injected engine must expose {name}()")
            if inspect.iscoroutinefunction(method):
                raise TypeError("AsyncLLM requires an async/RPC bridge; this adapter accepts synchronous clients")

    def start_capture(self, context):
        if self.config.profiler is None:
            raise RuntimeError("vLLM tracing is disabled in its engine configuration")
        if self._active:
            raise RuntimeError("vLLM capture already active")
        parameters = inspect.signature(self.engine.start_profile).parameters
        kwargs = {}
        if "profile_prefix" in parameters:
            kwargs["profile_prefix"] = f"{context.capture_id}-r{context.rank}"
        result = self.engine.start_profile(**kwargs)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError("Awaitable vLLM client requires an async bridge")
        self._active = True

    def stop_capture(self):
        if not self._active:
            return []
        try:
            result = self.engine.stop_profile()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Awaitable vLLM client requires an async bridge")
            # Worker files may still be flushing or may live on a remote host.
            return [TraceArtifact(self.name, self.config.to_backend_config()["torch_profiler_dir"],
                                  "vllm_trace_directory", "pending")]
        finally:
            self._active = False

    def read_metrics(self) -> list[MetricRecord]:
        getter = getattr(self.engine, "get_metrics", None)
        if not callable(getter) or inspect.iscoroutinefunction(getter):
            raise TypeError("Metrics require synchronous V1 get_metrics() with statistics enabled")
        records = []
        for metric in getter():
            data = cpu_value(vars(metric) if is_dataclass(metric) else metric)
            if not isinstance(data, dict) or "name" not in data:
                raise TypeError("Unsupported vLLM metric representation")
            if "buckets" in data:
                kind = "histogram"
                value = {k: data[k] for k in ("count", "sum", "buckets")}
            elif "values" in data:
                kind, value = "vector", data["values"]
            else:
                kind = type(metric).__name__.lower() if is_dataclass(metric) else data.get("metric_kind", "gauge")
                value = data["value"]
            records.append(MetricRecord(data["name"], value, metric_kind=kind,
                                        backend=self.name, labels=data.get("labels", {})))
        return records
