"""Backend-independent CPU-only observation contracts."""
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


def safe_component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Identifiers must be 1-128 ASCII letters/digits/underscore/dot/hyphen, starting alphanumeric")
    return value


def cpu_value(value: Any, depth: int = 0) -> Any:
    """Never implicitly call item()/cpu() or retain a tensor's graph."""
    if depth > 8:
        raise ValueError("Metric payload nesting exceeds 8")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)) and len(value) <= 2048:
        return [cpu_value(v, depth + 1) for v in value]
    if isinstance(value, dict) and len(value) <= 2048 and all(isinstance(k, str) for k in value):
        return {k: cpu_value(v, depth + 1) for k, v in value.items()}
    raise TypeError("Metrics must be bounded CPU Python values; convert sampled tensors explicitly outside Probe")


@dataclass(frozen=True)
class CaptureContext:
    run_id: str
    capture_id: str
    iteration_id: int
    rank: int
    role: str
    directory: str


@dataclass
class MetricRecord:
    name: str
    value: Any
    unit: str = "unspecified"
    metric_kind: str = "gauge"
    backend: str = "framework"
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class StageRecord:
    name: str
    span_id: str
    parent_span_id: str | None
    duration_seconds: float
    start_monotonic_ns: int
    measurement_kind: str = "host_wall"
    failed: bool = False


@dataclass
class TraceArtifact:
    backend: str
    path: str
    format: str
    status: str  # complete, pending, failed
    error: str | None = None


class CaptureAdapter(Protocol):
    name: str

    def start_capture(self, context: CaptureContext) -> None: ...
    def stop_capture(self) -> list[TraceArtifact]: ...


def record_dict(record: MetricRecord | StageRecord | TraceArtifact) -> dict:
    # asdict would deepcopy tensors before we could reject them.
    return cpu_value(vars(record))
