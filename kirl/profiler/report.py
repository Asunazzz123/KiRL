"""
Offline summaries: preserve rank/clock/labels; do not add overlapping timings.

To build profiler result
```python
from kirl.profiler import build_report
report = build_report("<run_id path>")
report.export("profile_report.json")
```
"""


import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunReport:
    schema_version: int = 1
    runs: list[str] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    sessions: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return path


def build_report(paths) -> RunReport:
    """Pass completed session JSONL files, or a run directory. Streaming scan."""
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        paths = sorted(path.rglob("records.jsonl")) if path.is_dir() else [path]
    report = RunReport()
    runs, stage_groups, metric_groups = set(), {}, {}
    for path in dict.fromkeys(Path(p) for p in paths):
        status_path = path.with_name("status.json")
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            report.sessions.append(status)
            sink = status.get("sink", {})
            if sink.get("dropped") or sink.get("pending") or sink.get("error") or sink.get("writer_alive"):
                report.warnings.append(f"Incomplete telemetry: {path}")
        else:
            report.warnings.append(f"No final session status: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    report.warnings.append(f"Invalid/truncated JSON at {path}:{line_no}")
                    continue
                if record.get("schema_version") != 1:
                    report.warnings.append(f"Unsupported schema at {path}:{line_no}")
                    continue
                runs.add(record["run_id"])
                data = record["data"]
                identity = {k: record.get(k) for k in ("run_id", "session_id", "role", "rank", "host", "pid")}
                identity["observation_mode"] = record.get("observation_mode", "unknown")
                identity["trace_window"] = record.get("capture_id") is not None
                if record["kind"] == "stage":
                    key = (record["session_id"], data["name"], data["measurement_kind"], identity["trace_window"])
                    group = stage_groups.setdefault(key, {**identity, "name": data["name"],
                        "measurement_kind": data["measurement_kind"], "count": 0,
                        "total_seconds": 0.0, "min_seconds": data["duration_seconds"],
                        "max_seconds": 0.0, "failed_count": 0})
                    value = data["duration_seconds"]
                    group["count"] += 1
                    group["total_seconds"] += value
                    group["min_seconds"] = min(group["min_seconds"], value)
                    group["max_seconds"] = max(group["max_seconds"], value)
                    group["failed_count"] += int(data["failed"])
                elif record["kind"] == "metric":
                    key = (record["session_id"], data["backend"], data["name"], data["unit"],
                           data["metric_kind"], json.dumps(data["labels"], sort_keys=True), identity["trace_window"])
                    group = metric_groups.setdefault(key, {**identity, **data, "observations": 0})
                    # Snapshots, not blindly summed counters or averaged histogram percentiles.
                    group["observations"] += 1
                    group["value"] = data["value"]
                    group["last_timestamp_ns"] = record["timestamp_ns"]
                elif record["kind"] == "artifact":
                    report.artifacts.append({**identity, "capture_id": record.get("capture_id"), **data})
    report.runs = sorted(runs)
    report.stages = list(stage_groups.values())
    for stage in report.stages:
        stage["mean_seconds"] = stage["total_seconds"] / stage["count"]
    report.metrics = list(metric_groups.values())
    report.warnings.append("Stage totals may overlap; never sum nested scopes or rank totals as end-to-end time.")
    return report
