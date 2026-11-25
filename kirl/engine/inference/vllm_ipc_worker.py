"""The receiving process for rank-local vLLM CUDA IPC."""

import json
import sys
import traceback
from functools import partial
from multiprocessing.connection import Listener

import torch
from vllm import LLM

from .cuda_ipc import import_cuda_buffer
from .vllm_engine import vLLMEngine
from .vllm_ipc import _gpu_uuid


@torch.no_grad()
def _load_chunk(model, handle, entries, allowed_missing_prefixes):
    device_index = torch.cuda.current_device()
    buffer = import_cuda_buffer(handle, device_index)
    for name, shape, offset, count in entries:
        tensor = buffer.narrow(0, offset, count).view(shape)
        loaded = model.load_weights([(name, tensor)])
        if not loaded and not name.startswith(allowed_missing_prefixes):
            raise RuntimeError(f"vLLM did not load policy parameter {name!r}.")
    # load_weights copies into model-owned tensors. Complete those reads
    # before acknowledging and dropping the worker's IPC buffer view.
    torch.cuda.current_stream(device_index).synchronize()
    return len(entries)


def _serve(conn, engine, allowed_missing_prefixes):
    version = 0
    pending_version = None
    chunks = count = 0
    while True:
        try:
            command, payload = conn.recv()
        except EOFError:
            return

        try:
            if command == "close":
                conn.send((True, None))
                return
            if command == "generate":
                prompts, config, required_version = payload
                if pending_version is not None or required_version != version:
                    raise RuntimeError(f"vLLM cannot generate with policy version {required_version}.")
                result = engine.generate(prompts, config)
            elif command == "begin":
                if pending_version is not None or payload != version + 1:
                    raise RuntimeError("Unexpected vLLM weight update version.")
                pending_version = payload
                chunks = count = 0
                result = pending_version
            elif command == "chunk":
                update_version, sequence, handle, entries = payload
                if update_version != pending_version or sequence != chunks + 1:
                    raise RuntimeError("Unexpected vLLM weight chunk sequence.")
                [loaded] = engine._engine.apply_model(
                    partial(_load_chunk, handle=handle, entries=entries,
                            allowed_missing_prefixes=allowed_missing_prefixes)
                )
                chunks += 1
                count += loaded
                result = (pending_version, chunks, loaded)
            elif command == "finish":
                if pending_version is None or payload != (pending_version, chunks, count) or not count:
                    raise RuntimeError("Cannot publish an incomplete vLLM weight update.")
                version, pending_version = pending_version, None
                result = (version, chunks, count)
            else:
                raise ValueError(f"Unknown vLLM CUDA IPC command {command!r}.")
            conn.send((True, result))
        except Exception:
            conn.send((False, traceback.format_exc()))
            return


def main():
    socket_path, config, allowed_missing = sys.argv[1:]
    with Listener(socket_path, family="AF_UNIX") as listener, listener.accept() as conn:
        try:
            engine = vLLMEngine(LLM(**json.loads(config)))
            uuid = _gpu_uuid(torch.device("cuda", 0))
        except Exception:
            conn.send((False, traceback.format_exc()))
            return
        conn.send((True, uuid))
        _serve(conn, engine, tuple(json.loads(allowed_missing)))


if __name__ == "__main__":
    main()
