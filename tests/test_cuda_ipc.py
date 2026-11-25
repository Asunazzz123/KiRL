"""Check the transport's buffer layout independently of CUDA handle exchange."""

import torch

from kirl.engine.inference.cuda_ipc import iter_weight_buffers


def test_weight_buffers_preserve_order_and_bypass_capacity_for_large_weight():
    device = torch.device("cpu")
    params = [
        ("small", torch.arange(4, dtype=torch.float32).reshape(2, 2).T),
        ("scalar", torch.tensor(7.0)),
        ("new_dtype", torch.tensor([1, 2], dtype=torch.int64)),
        ("large", torch.arange(16, dtype=torch.float32)),
        ("tail", torch.tensor([0.0])),
    ]
    rebuilt = []
    chunk_names = []
    for buffer, entries in iter_weight_buffers(params, device, chunk_bytes=32):
        chunk_names.append([name for name, *_ in entries])
        for name, shape, offset, count in entries:
            rebuilt.append((name, buffer.narrow(0, offset, count).view(shape)))
        if entries[0][0] == "large":
            assert buffer.numel() == params[3][1].numel()
            assert buffer.data_ptr() != params[3][1].data_ptr()

    assert chunk_names == [["small", "scalar"], ["new_dtype"], ["large"], ["tail"]]
    for (name, expected), (received_name, received) in zip(params, rebuilt):
        assert name == received_name
        assert expected.shape == received.shape and expected.dtype == received.dtype
        assert torch.equal(expected, received)


def test_weight_buffers_flush_when_same_dtype_would_exceed_capacity():
    params = [(str(index), torch.arange(3, dtype=torch.bfloat16)) for index in range(3)]
    chunks = list(iter_weight_buffers(params, torch.device("cpu"), chunk_bytes=10))
    assert len(chunks) == 3
    assert [metadata[0] for _, metadata in chunks] == [(str(index), (3,), 0, 3) for index in range(3)]
