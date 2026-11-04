"""PyTorch trace adapter: observes FSDP without owning/wrapping the model."""
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .records import TraceArtifact

_capture_lock = threading.Lock()


@dataclass(frozen=True)
class TorchProfilerConfig:
    activities: tuple[str, ...] = ("cpu",)
    record_shapes: bool = False
    with_stack: bool = False
    profile_memory: bool = False

    def __post_init__(self):
        if not self.activities or any(a not in ("cpu", "cuda") for a in self.activities):
            raise ValueError("activities must contain cpu and/or cuda")
        object.__setattr__(self, "activities", tuple(self.activities))
        for name in ("record_shapes", "with_stack", "profile_memory"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")


class TorchProfilerAdapter:
    name = "torch"

    def __init__(self, config: TorchProfilerConfig | None = None):
        self.config = config or TorchProfilerConfig()
        self._prof = None
        self._path = None

    def start_capture(self, context):
        if self._prof is not None:
            raise RuntimeError("This adapter already has an active capture")
        if not _capture_lock.acquire(blocking=False):
            raise RuntimeError("Another TinyEngine torch profiler is already active in this process")
        try:
            import torch

            if "cuda" in self.config.activities and not torch.cuda.is_available():
                raise RuntimeError("CUDA profiling requested, but CUDA is unavailable")
            directory = Path(context.directory) / self.name
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / "trace.json"
            activities = [getattr(torch.profiler.ProfilerActivity, a.upper()) for a in self.config.activities]
            self._prof = torch.profiler.profile(
                activities=activities, record_shapes=self.config.record_shapes,
                with_stack=self.config.with_stack, profile_memory=self.config.profile_memory,
            )
            self._prof.start()
        except BaseException:
            self._prof = None
            _capture_lock.release()
            raise

    def annotate(self, name):
        if self._prof is None:
            return nullcontext()
        import torch
        return torch.profiler.record_function(name)

    def stop_capture(self):
        if self._prof is None:
            return []
        try:
            self._prof.stop()
            self._prof.export_chrome_trace(str(self._path))
            return [TraceArtifact(self.name, str(self._path), "chrome_trace", "complete")]
        finally:
            self._prof = None
            _capture_lock.release()
