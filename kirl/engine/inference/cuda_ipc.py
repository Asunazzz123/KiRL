"""CUDA weight buffers shared by a trainer rank and its local inference worker."""

from collections.abc import Iterable, Iterator

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

WeightSlice = tuple[str, tuple[int, ...], int, int]


def iter_weight_buffers(
    named_params: Iterable[tuple[str, torch.Tensor]],
    device: torch.device,
    chunk_bytes: int,
) -> Iterator[tuple[torch.Tensor, list[WeightSlice]]]:
    """Pack same-dtype weights in order, holding each buffer until consumption."""
    buffer = None
    entries: list[WeightSlice] = []
    used = 0
    for name, tensor in named_params:
        tensor = tensor.detach()
        if tensor.device != device:
            raise ValueError(f"Weight {name!r} is on {tensor.device}, expected {device}.")
        count = tensor.numel()
        capacity = chunk_bytes // tensor.element_size()
        oversized = count > capacity
        if entries and (oversized or buffer.dtype != tensor.dtype or used + count > capacity):
            yield buffer, entries
            buffer, entries, used = None, [], 0

        if oversized:
            # CUDA IPC attaches bookkeeping to the exported storage. Keep it
            # on a temporary allocation even if the caller owns this weight.
            oversized_buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
            oversized_buffer.copy_(tensor)
            yield oversized_buffer.view(-1), [(name, tuple(tensor.shape), 0, count)]
            del oversized_buffer
            continue

        if buffer is None:
            buffer = torch.empty(capacity, dtype=tensor.dtype, device=device)
        buffer[used:used + count].view(tensor.shape).copy_(tensor)
        entries.append((name, tuple(tensor.shape), used, count))
        used += count

    if entries:
        yield buffer, entries


def export_cuda_buffer(buffer: torch.Tensor) -> tuple:
    # The receiver may open a handle whose packing work was submitted after
    # an earlier export of this storage. Drain this chunk's producing stream.
    torch.cuda.current_stream(buffer.device).synchronize()
    _, args = reduce_tensor(buffer)
    return args


def import_cuda_buffer(args: tuple, device_index: int) -> torch.Tensor:
    args = list(args)
    # reduce_tensor uses the producer's logical device index. The isolated
    # worker exposes just one physical GPU, which it always sees as cuda:0.
    args[6] = device_index
    return rebuild_cuda_tensor(*args)
