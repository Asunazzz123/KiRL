"""Rank-local vLLM subprocess with synchronous CUDA IPC weight updates."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist

from ...profiler.probe import NullProbe, Probe
from .base import GenerationBackend, GenerationConfig, GenerationOutput
from .cuda_ipc import export_cuda_buffer, iter_weight_buffers
from .vllm_config import VLLMBackendConfig

_TRAINER_ENV = {
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
}


def _gpu_uuid(device: torch.device) -> str:
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"


class VLLMIPCBackend(GenerationBackend):
    """Launch one TP=1 worker on the caller's CUDA device.

    Each trainer rank owns its worker and must enter `update_weight` in the
    same order as the other ranks if the parameter iterator uses collectives.
    """

    def __init__(
        self,
        config: VLLMBackendConfig,
        *,
        worker_python: str | Path = sys.executable,
        device: torch.device | None = None,
        chunk_bytes: int = 256 * 1024**2,
        log_path: str | Path,
        probe: Probe | NullProbe | None = None,
        language_model_only: bool = False,
        allowed_missing_prefixes: tuple[str, ...] = (),
        startup_timeout: float = 900,
        rpc_timeout: float = 300,
    ):
        if sys.platform != "linux" or not torch.cuda.is_available():
            raise RuntimeError("CUDA IPC requires Linux and a CUDA device.")
        if config.tensor_parallel_size != 1:
            raise ValueError("The rank-local CUDA IPC backend requires tensor_parallel_size=1.")
        if chunk_bytes < 8 or startup_timeout <= 0 or rpc_timeout <= 0:
            raise ValueError("Invalid CUDA IPC chunk size or timeout.")
        device = device if device is not None else torch.device("cuda")
        if device.type != "cuda":
            raise ValueError("CUDA IPC requires a CUDA device.")
        self.device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        self.gpu_uuid = _gpu_uuid(self.device)
        self.chunk_bytes = chunk_bytes
        self.rpc_timeout = rpc_timeout
        self.probe = probe if probe is not None else NullProbe()
        self.version = 0
        self._conn = None
        self._process = None
        self._work_dir = TemporaryDirectory(prefix="kirl-vllm-")
        socket_path = str(Path(self._work_dir.name) / "ipc.sock")
        env = {
            key: value for key, value in os.environ.items()
            if key not in _TRAINER_ENV and not key.startswith(("TORCHELASTIC_", "TORCH_NCCL_"))
        }
        python = Path(worker_python).absolute()
        env.update(
            CUDA_VISIBLE_DEVICES=self.gpu_uuid,
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_ALLOW_INSECURE_SERIALIZATION="1",
            PATH=str(python.parent) + os.pathsep + env.get("PATH", ""),
        )
        # Make the same KiRL package importable in the isolated vLLM environment.
        package_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, env.get("PYTHONPATH"))))
        worker_config = {key: value for key, value in asdict(config).items() if value is not None} | {
            "language_model_only": language_model_only,
            "enable_prefix_caching": False,
            "distributed_executor_backend": "uni",
        }
        try:
            with Path(log_path).open("w") as log:
                self._process = subprocess.Popen(
                    [str(python), "-m", "kirl.engine.inference.vllm_ipc_worker",
                     socket_path, json.dumps(worker_config), json.dumps(allowed_missing_prefixes)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            deadline = time.monotonic() + startup_timeout
            while not Path(socket_path).exists():
                if self._process.poll() is not None:
                    raise RuntimeError(f"CUDA IPC worker exited during startup. See {log_path}.")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"CUDA IPC worker startup timed out. See {log_path}.")
                time.sleep(0.1)
            self._conn = Client(socket_path, family="AF_UNIX")
            if self._receive(max(0, deadline - time.monotonic())) != self.gpu_uuid:
                raise RuntimeError("CUDA IPC worker is on a different physical GPU.")
        except BaseException:
            self._abort()
            raise

    def _receive(self, timeout):
        if not self._conn.poll(timeout):
            raise TimeoutError("Timed out waiting for the vLLM worker.")
        ok, result = self._conn.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def _call(self, command, payload=None):
        if self._conn is None:
            raise RuntimeError("The vLLM CUDA IPC worker is closed.")
        self._conn.send((command, payload))
        return self._receive(self.rpc_timeout)

    def generate(self, prompt_token_ids: Sequence[Sequence[int]], config: GenerationConfig) -> list[GenerationOutput]:
        with self.probe.stage("vllm.generate_rpc"):
            return self._call("generate", (prompt_token_ids, config, self.version))

    def update_weight(self, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
        version = self.version + 1
        chunks = count = total_bytes = max_buffer_bytes = 0
        try:
            self._expect("begin", version, version)
            for buffer, entries in iter_weight_buffers(named_params, self.device, self.chunk_bytes):
                chunks += 1
                with self.probe.stage("vllm.ipc_chunk"):
                    handle = export_cuda_buffer(buffer)
                    self._expect("chunk", (version, chunks, handle, entries), (version, chunks, len(entries)))
                count += len(entries)
                total_bytes += sum(entry[3] for entry in entries) * buffer.element_size()
                max_buffer_bytes = max(max_buffer_bytes, buffer.numel() * buffer.element_size())
                del buffer
            if not count:
                raise ValueError("The policy exporter yielded no weights.")
            self._expect("finish", (version, chunks, count), (version, chunks, count))
            torch.cuda.ipc_collect()
            if dist.is_initialized() and dist.get_world_size() > 1:
                device_ids = [self.device.index] if dist.get_backend() == "nccl" else None
                dist.barrier(device_ids=device_ids)
        except BaseException:
            # End the receiver before discarding a buffer after a failed ACK.
            self._abort()
            raise
        self.version = version
        self.probe.metric("vllm.ipc_chunks", chunks)
        self.probe.metric("vllm.ipc_parameters", count)
        self.probe.metric("vllm.ipc_bytes", total_bytes, unit="bytes")
        self.probe.metric("vllm.ipc_max_buffer_bytes", max_buffer_bytes, unit="bytes")

    def _expect(self, command, payload, expected):
        if self._call(command, payload) != expected:
            raise RuntimeError(f"Unexpected vLLM CUDA IPC acknowledgement for {command}.")

    def _abort(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._process is not None:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait()
            self._process = None
        self._work_dir.cleanup()

    def close(self):
        try:
            if self._conn is not None and self._process.poll() is None:
                self._conn.send(("close", None))
                self._receive(5)
                self._process.wait(timeout=5)
        except (OSError, EOFError, RuntimeError, subprocess.TimeoutExpired):
            pass
        finally:
            self._abort()
