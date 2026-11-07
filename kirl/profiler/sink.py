"""Bounded non-blocking producers; buffered JSONL on one writer thread."""
import json
import queue
import threading
import time
from pathlib import Path


class JsonlSink:
    def __init__(self, path, *, capacity=1024, batch_size=64, flush_interval_seconds=5.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue(maxsize=capacity)
        self._batch_size = batch_size
        self._interval = flush_interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._closed = False
        self.accepted = self.written = self.dropped = 0
        self.error = None
        self._thread = threading.Thread(target=self._run, name="kirl-metrics", daemon=True)
        self._thread.start()

    def emit(self, record: dict) -> bool:
        with self._lock:
            if self._closed or self.error is not None:
                self.dropped += 1
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self.dropped += 1
                return False
            self.accepted += 1
            return True

    def _run(self):
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                batch = []
                deadline = time.monotonic() + self._interval
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                        batch.append(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
                    except queue.Empty:
                        pass
                    if len(batch) >= self._batch_size or time.monotonic() >= deadline:
                        stream.writelines(batch)
                        stream.flush()  # OS cache, intentionally not fsync
                        self.written += len(batch)
                        batch.clear()
                        deadline = time.monotonic() + self._interval
                stream.writelines(batch)
                stream.flush()
                self.written += len(batch)
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"

    def close(self, timeout=10.0) -> dict:
        with self._lock:
            self._closed = True
            self._stop.set()
        self._thread.join(timeout=timeout)
        return self.stats()

    def stats(self) -> dict:
        return {
            "accepted": self.accepted, "written": self.written, "dropped": self.dropped,
            "pending": self.accepted - self.written, "error": self.error,
            "writer_alive": self._thread.is_alive(),
        }
