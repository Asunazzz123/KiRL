import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from tiny_engine.profiler import (
    ProfilerConfig, TorchProfilerAdapter, TorchProfilerConfig,
    VLLMProfilerAdapter, VLLMProfilerConfig, CaptureContext,
    create_probe, build_report,
)
from tiny_engine.profiler.settings import ProfilerSettings
from tiny_engine.profiler.sink import JsonlSink


def read_records(probe):
    return [json.loads(line) for line in probe.record_path.read_text().splitlines()]


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"mode": "trace"}, {"sample_every_steps": 0},
    {"ranks": (-1,)}, {"max_pending_records": True}, {"flush_interval_seconds": float("nan")},
])
def test_invalid_policy(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ProfilerConfig(**kwargs)


def test_settings_mapping():
    settings = ProfilerSettings.from_mapping({"global": {"ranks": [0, 2]}, "torch": {"activities": ["cpu"]}})
    assert settings.global_config.ranks == (0, 2)
    assert settings.vllm.profiler is None
    with pytest.raises(ValueError):
        ProfilerSettings.from_mapping({"typo": {}})


def test_off_and_rank_gating_have_no_io(tmp_path):
    class Never:
        name = "never"
        def start_capture(self, ctx):
            raise AssertionError("Must not start")
    for config, rank in ((ProfilerConfig(output_dir=str(tmp_path)), 0),
                         (ProfilerConfig(mode="trace", trace_iterations=(1,), output_dir=str(tmp_path)), 1)):
        with create_probe(config, run_id="test", rank=rank, adapters=[Never()]) as probe:
            with probe.iteration(1), probe.stage("forward"):
                assert not probe.metric("x", object())
    assert list(tmp_path.iterdir()) == []


def test_basic_records_exception_and_report(tmp_path):
    config = ProfilerConfig(mode="basic", output_dir=str(tmp_path))
    with create_probe(config, run_id="test") as probe:
        with pytest.raises(RuntimeError, match="training failed"):
            with probe.iteration(2, policy_version="v3"):
                probe.step(5)
                with probe.stage("forward"):
                    probe.metric("loss", 0.5, unit="dimensionless")
                    raise RuntimeError("training failed")
    records = read_records(probe)
    stages = [r for r in records if r["kind"] == "stage"]
    assert {r["data"]["name"] for r in stages} == {"forward", "iteration"}
    assert all(r["data"]["failed"] for r in stages)
    assert next(r for r in stages if r["data"]["name"] == "forward")["data"]["parent_span_id"] is not None
    report = build_report(tmp_path)
    assert report.metrics[0]["value"] == 0.5
    assert report.sessions[0]["sink"]["pending"] == 0
    assert json.loads(report.export(tmp_path / "report.json").read_text())["runs"] == ["test"]


def test_sampled_never_converts_skipped_values(tmp_path):
    config = ProfilerConfig(mode="sampled", sample_every_steps=3, output_dir=str(tmp_path))
    with create_probe(config, run_id="sample") as probe:
        probe.step(1)
        assert not probe.metric("skipped", object())
        probe.step(3)
        with pytest.raises(TypeError):
            probe.metric("invalid", object())
        probe.metric("loss", 1.0)
    assert [r["data"]["name"] for r in read_records(probe)] == ["loss"]


def test_bounded_queue_drops_without_blocking(tmp_path, monkeypatch):
    gate = threading.Event()
    original = JsonlSink._run
    def paused(self):
        gate.wait(timeout=5)
        original(self)
    monkeypatch.setattr(JsonlSink, "_run", paused)
    sink = JsonlSink(tmp_path / "events.jsonl", capacity=1)
    try:
        assert sink.emit({"id": 1})
        assert not sink.emit({"id": 2})
    finally:
        gate.set()
        stats = sink.close()
    assert stats["written"] == 1 and stats["dropped"] == 1


def test_sink_failure_is_reported(tmp_path):
    path = tmp_path / "directory"
    path.mkdir()
    sink = JsonlSink(path)
    sink.emit({"x": 1})
    stats = sink.close()
    assert stats["error"] is not None
    assert not stats["writer_alive"]


@dataclass
class Counter:
    name: str
    labels: dict
    value: int


@dataclass
class Histogram:
    name: str
    labels: dict
    count: int
    sum: float
    buckets: dict


class FakeEngine:
    def __init__(self):
        self.starts, self.stops = [], 0
    def start_profile(self, profile_prefix=None):
        self.starts.append(profile_prefix)
    def stop_profile(self):
        self.stops += 1
    def get_metrics(self):
        return [Counter("vllm:tokens", {"engine": "0"}, 12),
                Histogram("vllm:latency", {"engine": "0"}, 2, 0.4, {"1.0": 2, "+Inf": 2})]


def test_vllm_configuration_and_capture_contract(tmp_path):
    config = VLLMProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path / "traces"))
    assert config.to_backend_config()["torch_profiler_with_stack"] is False
    engine = FakeEngine()
    adapter = VLLMProfilerAdapter(engine, config)
    policy = ProfilerConfig(mode="trace", trace_iterations=(2,), output_dir=str(tmp_path))
    with create_probe(policy, run_id="vllm-test", adapters=[adapter]) as probe:
        with probe.iteration(1):
            pass
        with pytest.raises(ValueError, match="business"):
            with probe.iteration(2):
                assert probe.collect_backend_metrics() == 2
                raise ValueError("business")
    assert len(engine.starts) == engine.stops == 1
    report = build_report(tmp_path)
    assert report.artifacts[0]["status"] == "pending"
    histogram = next(m for m in report.metrics if m["metric_kind"] == "histogram")
    assert histogram["labels"] == {"engine": "0"}
    assert histogram["value"]["buckets"]["+Inf"] == 2


def test_async_client_is_rejected():
    class AsyncEngine:
        async def start_profile(self): pass
        async def stop_profile(self): pass
    with pytest.raises(TypeError, match="async/RPC"):
        VLLMProfilerAdapter(AsyncEngine(), VLLMProfilerConfig())


def test_backend_failure_does_not_retry_business(tmp_path):
    class Broken:
        name = "broken"
        def start_capture(self, ctx): raise RuntimeError("capture unavailable")
    policy = ProfilerConfig(mode="trace", trace_iterations=(0,), output_dir=str(tmp_path))
    calls = []
    with create_probe(policy, run_id="failure", adapters=[Broken()]) as probe:
        with pytest.warns(RuntimeWarning, match="capture unavailable"):
            with probe.iteration(0):
                calls.append(1)
    assert calls == [1]
    assert build_report(tmp_path).artifacts[0]["status"] == "failed"


def test_cpu_torch_trace_and_repeated_windows(tmp_path):
    torch = pytest.importorskip("torch")
    adapter = TorchProfilerAdapter(TorchProfilerConfig(activities=("cpu",)))
    policy = ProfilerConfig(mode="trace", trace_iterations=(1, 2), output_dir=str(tmp_path))
    with create_probe(policy, run_id="torch-test", adapters=[adapter]) as probe:
        for iteration in (1, 2):
            with probe.iteration(iteration), probe.stage("forward"):
                x = torch.ones(8, requires_grad=True)
                (x.square().sum()).backward()
    artifacts = build_report(tmp_path).artifacts
    assert len(artifacts) == 2
    assert artifacts[0]["path"] != artifacts[1]["path"]
    for artifact in artifacts:
        assert artifact["status"] == "complete"
        data = json.loads(Path(artifact["path"]).read_text())
        assert any(e.get("name") == "forward" for e in data["traceEvents"])


def test_trace_lock_released_after_business_exception(tmp_path):
    torch = pytest.importorskip("torch")
    policy = ProfilerConfig(mode="trace", trace_iterations=(0, 1), output_dir=str(tmp_path))
    with create_probe(policy, run_id="cleanup", adapters=[TorchProfilerAdapter()]) as probe:
        with pytest.raises(ValueError):
            with probe.iteration(0):
                torch.ones(1)
                raise ValueError("business")
        with probe.iteration(1):
            torch.ones(1)
    assert all(a["status"] == "complete" for a in build_report(tmp_path).artifacts)


def test_annotation_failure_is_reported(tmp_path):
    from contextlib import contextmanager
    class BrokenAnnotation:
        name = "annotation"
        def start_capture(self, ctx): pass
        def stop_capture(self): return []
        @contextmanager
        def annotate(self, name):
            yield
            raise RuntimeError("annotation exit failure")
    policy = ProfilerConfig(mode="trace", trace_iterations=(0,), output_dir=str(tmp_path))
    with create_probe(policy, run_id="annotation", adapters=[BrokenAnnotation()]) as probe:
        with pytest.warns(RuntimeWarning, match="annotation exit failure"):
            with probe.iteration(0), probe.stage("work"):
                pass
    assert build_report(tmp_path).artifacts[0]["status"] == "failed"


def test_interrupt_during_start_cleans_previous_adapter(tmp_path):
    stopped = []
    class Started:
        name = "first"
        def start_capture(self, ctx): pass
        def stop_capture(self):
            stopped.append(True)
            return []
    class Interrupted:
        name = "second"
        def start_capture(self, ctx): raise KeyboardInterrupt()
    policy = ProfilerConfig(mode="trace", trace_iterations=(0,), output_dir=str(tmp_path))
    with create_probe(policy, run_id="interrupt", adapters=[Started(), Interrupted()]) as probe:
        with pytest.raises(KeyboardInterrupt):
            with probe.iteration(0):
                pytest.fail("Should not execute")
    assert stopped == [True]


def test_report_preserves_rank_labels_and_counter_snapshots(tmp_path):
    config = ProfilerConfig(mode="basic", all_ranks=True, output_dir=str(tmp_path))
    for rank in (0, 1):
        with create_probe(config, run_id="ranks", rank=rank) as probe:
            for value in (10, 20):
                probe.metric("tokens", value + rank, unit="tokens", metric_kind="counter", labels={"engine": str(rank)})
    report = build_report(tmp_path)
    assert {m["value"] for m in report.metrics} == {20, 21}
    assert len(report.metrics) == 2
    assert all(m["observations"] == 2 for m in report.metrics)


def test_legacy_vllm_start_signature(tmp_path):
    class Legacy(FakeEngine):
        def start_profile(self):
            self.starts.append("legacy")
    engine = Legacy()
    config = VLLMProfilerConfig(profiler="torch", torch_profiler_dir=str(tmp_path))
    adapter = VLLMProfilerAdapter(engine, config)
    adapter.start_capture(CaptureContext("run", "capture", 1, 0, "rollout", str(tmp_path)))
    adapter.stop_capture()
    assert engine.starts == ["legacy"] and engine.stops == 1


def test_tensor_metrics_rejected_without_item(tmp_path):
    torch = pytest.importorskip("torch")
    with create_probe(ProfilerConfig(mode="basic", output_dir=str(tmp_path)), run_id="tensor") as probe:
        with pytest.raises(TypeError, match="CPU Python"):
            probe.metric("loss", torch.tensor(1.0, requires_grad=True))
        probe.metric("nonfinite", float("nan"))
    assert read_records(probe)[0]["data"]["value"] is None


def test_default_yaml_settings():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    config = OmegaConf.load(Path(__file__).parents[1] / "configs/profiler/default.yaml")
    settings = ProfilerSettings.from_mapping(config)
    assert settings.global_config.mode == "off"
    assert settings.torch.activities == ("cpu",)


def test_yaml_plain_mapping():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/profiler/default.yaml").read_text())
    settings = ProfilerSettings.from_mapping(config)
    assert settings.global_config.mode == "off"


def test_annotation_cleanup_never_masks_business_error(tmp_path):
    class Scope:
        def __enter__(self): return self
        def __exit__(self, *exc): raise RuntimeError("broken annotation")
    class Adapter:
        name = "annotation"
        def start_capture(self, ctx): pass
        def stop_capture(self): return []
        def annotate(self, name): return Scope()
    policy = ProfilerConfig(mode="trace", trace_iterations=(0,), output_dir=str(tmp_path))
    with create_probe(policy, run_id="original-error", adapters=[Adapter()]) as probe:
        with pytest.warns(RuntimeWarning, match="broken annotation"):
            with pytest.raises(ValueError, match="original training error"):
                with probe.iteration(0), probe.stage("forward"):
                    raise ValueError("original training error")


def test_report_keeps_trace_and_normal_windows_separate(tmp_path):
    policy = ProfilerConfig(mode="trace", trace_iterations=(1,), output_dir=str(tmp_path))
    with create_probe(policy, run_id="windows") as probe:
        for i in (0, 1):
            with probe.iteration(i), probe.stage("forward"):
                probe.metric("loss", 1.0)
    report = build_report(tmp_path)
    forward = [s for s in report.stages if s["name"] == "forward"]
    assert len(forward) == 2
    assert {s["trace_window"] for s in forward} == {True, False}
    assert len(report.metrics) == 2
