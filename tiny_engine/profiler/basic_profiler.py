"""Global observation policy; runtime rank/step belong to Probe."""
import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProfilerConfig:
    """
    基本Profiler类
    """
    mode: Literal["off", "basic", "sampled", "trace"] = "off"
    ranks: tuple[int, ...] = (0,)
    all_ranks: bool = False
    sample_every_steps: int = 100
    trace_iterations: tuple[int, ...] = ()
    output_dir: str = "./outputs/profile"
    flush_interval_seconds: float = 5.0
    max_pending_records: int = 1024
    batch_size: int = 64

    def __post_init__(self):
        if self.mode not in ("off", "basic", "sampled", "trace"):
            raise ValueError(f"Unsupported profiler mode: {self.mode!r}")
        for name in ("ranks", "trace_iterations"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} must contain non-negative integers")
            object.__setattr__(self, name, tuple(values))
        for name in ("sample_every_steps", "max_pending_records", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.all_ranks) is not bool:
            raise TypeError("all_ranks must be bool")
        if not isinstance(self.flush_interval_seconds, (int, float)) or isinstance(self.flush_interval_seconds, bool):
            raise TypeError("flush_interval_seconds must be a number")
        if not math.isfinite(self.flush_interval_seconds) or self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be finite and > 0")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if self.mode == "trace" and not self.trace_iterations:
            raise ValueError("trace mode requires explicit trace_iterations")

    def selects_rank(self, rank: int) -> bool:
        return self.mode != "off" and (self.all_ranks or rank in self.ranks)
