"""Single-owner observation controller; one instance per worker process."""
import json
import os
import socket
import sys
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .basic_profiler import ProfilerConfig
from .records import CaptureContext, MetricRecord, StageRecord, TraceArtifact, record_dict, safe_component
from .sink import JsonlSink
from ..utils.logging import Log


class NullProbe:
    enabled = False
    wants_sample = False
    record_path = None
    session_dir = None

    def __init__(self, *, run_id=None, rank=None, role=None, algorithm=None):
        current = Log.get_context()
        self.logging_context = {
            "rank": current["rank"] if rank is None else rank,
            "role": current["role"] if role is None else role,
        }
        if run_id is not None:
            self.logging_context["run_id"] = run_id
        if algorithm is not None:
            self.logging_context["algorithm"] = algorithm
        self.update_step = 0

    @contextmanager
    def iteration(self, iteration_id, *, policy_version=None):
        with Log.bind(**self.logging_context, step=self.update_step,
                      iteration_id=iteration_id, policy_version=policy_version):
            yield self

    @contextmanager
    def stage(self, name):
        with Log.bind(**self.logging_context, phase=name, step=self.update_step):
            yield

    def step(self, update_step):
        self.update_step = update_step

    def metric(self, name, value, **kwargs):
        return False

    def collect_backend_metrics(self):
        return 0

    def close(self):
        return {"enabled": False}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Probe:
    """Use create_probe() for rank gating. Does not synchronize GPUs or ranks.

    stage() measures host wall time, never claims GPU completion. An iteration
    window includes profiler start/stop/export overhead; nested stage times are
    not additive. Values must already be CPU Python values.
    """
    enabled = True

    def __init__(self, config: ProfilerConfig, *, run_id: str, rank=0, role="train", adapters=(), algorithm=None):
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if not config.selects_rank(rank):
            raise ValueError("Use create_probe() for disabled/unselected ranks")
        self.config = config
        self.run_id = safe_component(run_id)
        self.role = safe_component(role)
        self.rank = rank
        self.logging_context = {"run_id": self.run_id, "role": self.role, "rank": rank}
        if algorithm is not None:
            self.logging_context["algorithm"] = safe_component(algorithm)
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.session_id = f"{role}-r{rank}-p{self.pid}-{uuid.uuid4().hex[:12]}"
        self.session_dir = Path(config.output_dir).expanduser().resolve() / run_id / self.session_id
        self.record_path = self.session_dir / "records.jsonl"
        self.adapters = tuple(adapters)
        names = [a.name for a in self.adapters]
        if len(set(names)) != len(names):
            raise ValueError("Adapter names must be unique per Probe")
        self.iteration_id = self.policy_version = self.capture_id = None
        self.update_step = 0
        self._active = []
        self._stack = []
        self._sequence = 0
        self._closed = False
        self._close_result = None
        self._capture_count = 0
        self._errors = []
        self._started_ns = time.perf_counter_ns()
        self._sink = JsonlSink(self.record_path, capacity=config.max_pending_records,
                               batch_size=config.batch_size, flush_interval_seconds=config.flush_interval_seconds)

    @property
    def wants_sample(self):
        return not self._closed and (
            self.config.mode != "sampled" or self.update_step % self.config.sample_every_steps == 0
        )

    def _emit(self, kind, record):
        data = record_dict(record)
        envelope = {
            "schema_version": 1, "kind": kind, "run_id": self.run_id,
            "session_id": self.session_id, "role": self.role, "rank": self.rank,
            "host": self.host, "pid": self.pid, "iteration_id": self.iteration_id,
            "update_step": self.update_step, "policy_version": self.policy_version,
            "capture_id": self.capture_id, "observation_mode": self.config.mode,
            "algorithm": self.logging_context.get("algorithm", Log.get_context()["algorithm"]),
            "phase": Log.get_context()["phase"],
            "timestamp_ns": time.time_ns(), "data": data,
        }
        return self._sink.emit(envelope)

    def _error(self, adapter, operation, exc):
        message = f"{adapter.name}.{operation}: {type(exc).__name__}: {exc}"
        # Bounded diagnostic history; ordinary telemetry must not retain unbounded errors.
        if len(self._errors) < 32:
            self._errors.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        self._emit("artifact", TraceArtifact(adapter.name, "", "error", "failed", message))

    def step(self, update_step):
        """Set the current optimizer-update index; does not advance backend schedules."""
        if type(update_step) is not int or update_step < 0:
            raise ValueError("update_step must be a non-negative integer")
        self.update_step = update_step

    def metric(self, name, value, *, unit="unspecified", metric_kind="gauge", backend="framework", labels=None):
        if not self.wants_sample:
            return False
        return self._emit("metric", MetricRecord(name, value, unit, metric_kind, backend, labels or {}))

    def collect_backend_metrics(self):
        """Explicit, sampled polling; never called implicitly inside every stage."""
        if not self.wants_sample:
            return 0
        count = 0
        for adapter in self.adapters:
            reader = getattr(adapter, "read_metrics", None)
            if reader is None:
                continue
            try:
                for record in reader():
                    count += int(self._emit("metric", record))
            except Exception as exc:
                self._error(adapter, "read_metrics", exc)
        return count

    @contextmanager
    def stage(self, name):
        # Bind even outside sampled windows so event logs keep their identity.
        with Log.bind(**self.logging_context, phase=name, step=self.update_step):
            with self._stage(name):
                yield

    @contextmanager
    def _stage(self, name):
        if not self.wants_sample:
            yield
            return
        self._sequence += 1
        span_id = str(self._sequence)
        parent_id = self._stack[-1] if self._stack else None
        self._stack.append(span_id)
        start = time.perf_counter_ns()
        failed = True
        scopes = []
        try:
            for adapter in self._active:
                annotate = getattr(adapter, "annotate", None)
                if annotate is not None:
                    try:
                        scope = annotate(name)
                        scope.__enter__()
                        scopes.append((adapter, scope))
                    except Exception as exc:
                        self._error(adapter, "annotate", exc)
            yield
            failed = False
        finally:
            exception = sys.exc_info()
            for adapter, scope in reversed(scopes):
                try:
                    scope.__exit__(*exception)
                except Exception as exc:
                    self._error(adapter, "end_annotation", exc)
            duration = (time.perf_counter_ns() - start) / 1e9
            self._stack.pop()
            self._emit("stage", StageRecord(name, span_id, parent_id, duration, start, failed=failed))

    @contextmanager
    def iteration(self, iteration_id, *, policy_version=None):
        with Log.bind(**self.logging_context, step=self.update_step,
                      iteration_id=iteration_id, policy_version=policy_version):
            with self._iteration(iteration_id, policy_version=policy_version):
                yield self

    @contextmanager
    def _iteration(self, iteration_id, *, policy_version=None):
        if self._closed:
            raise RuntimeError("Probe is closed")
        if self.iteration_id is not None:
            raise RuntimeError("Iteration windows cannot be nested")
        if type(iteration_id) is not int or iteration_id < 0:
            raise ValueError("iteration_id must be a non-negative integer")
        if policy_version is not None and type(policy_version) not in (str, int):
            raise TypeError("policy_version must be str, int, or None")
        self.iteration_id, self.policy_version = iteration_id, policy_version
        selected = self.config.mode == "trace" and iteration_id in self.config.trace_iterations
        try:
            with self.stage("iteration"):
                try:
                    if selected:
                        self._capture_count += 1
                        self.capture_id = f"iteration-{iteration_id}-capture-{self._capture_count}"
                        context = CaptureContext(self.run_id, self.capture_id, iteration_id, self.rank, self.role,
                                                 str(self.session_dir / self.capture_id))
                        for adapter in self.adapters:
                            start = time.perf_counter_ns()
                            try:
                                adapter.start_capture(context)
                                self._active.append(adapter)
                            except Exception as exc:
                                self._error(adapter, "start_capture", exc)
                            finally:
                                self.metric("profiler.start_seconds", (time.perf_counter_ns() - start) / 1e9,
                                            unit="seconds", backend=adapter.name)
                    yield self
                finally:
                    for adapter in reversed(self._active):
                        start = time.perf_counter_ns()
                        try:
                            for artifact in adapter.stop_capture():
                                self._emit("artifact", artifact)
                        except Exception as exc:
                            self._error(adapter, "stop_capture", exc)
                        finally:
                            self.metric("profiler.stop_seconds", (time.perf_counter_ns() - start) / 1e9,
                                        unit="seconds", backend=adapter.name)
                    self._active.clear()
        finally:
            self.iteration_id = self.policy_version = self.capture_id = None

    def close(self):
        if self._closed:
            return self._close_result
        if self.iteration_id is not None:
            raise RuntimeError("Exit the iteration window before closing Probe")
        self._closed = True
        stats = self._sink.close()
        self._close_result = {
            "schema_version": 1, "run_id": self.run_id, "session_id": self.session_id,
            "role": self.role, "rank": self.rank, "host": self.host, "pid": self.pid,
            "algorithm": self.logging_context.get("algorithm", Log.get_context()["algorithm"]),
            "configuration": asdict(self.config), "adapters": [a.name for a in self.adapters],
            "elapsed_host_seconds_including_drain": (time.perf_counter_ns() - self._started_ns) / 1e9,
            "measurement_kind": "host_wall_no_device_sync", "sink": stats, "adapter_errors": self._errors,
        }
        try:
            (self.session_dir / "status.json").write_text(json.dumps(self._close_result, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not write profiler status: {exc}", RuntimeWarning, stacklevel=2)
        return self._close_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def create_probe(config: ProfilerConfig, *, run_id: str, rank=0, role="train", adapters=(), algorithm=None):
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if not config.selects_rank(rank):
        return NullProbe(run_id=run_id, rank=rank, role=role, algorithm=algorithm)
    return Probe(config, run_id=run_id, rank=rank, role=role, adapters=adapters, algorithm=algorithm)
