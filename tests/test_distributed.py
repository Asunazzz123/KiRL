"""Real two-process tests; pytest uses CPU/Gloo by default.

GPU validation (requires two GPUs and NCCL):
    torchrun --standalone --nproc_per_node=2 tests/test_distributed.py --worker
"""

import os
from pathlib import Path
import subprocess
import sys
import math
import pytest
import torch
import torch.distributed as dist

from tiny_engine.distributed.context import DistributedContext



@pytest.fixture(scope="module")
def ctx():
    """
    启动分布式测试脚本
    """
    if "RANK" not in os.environ:
        pytest.skip("pytest不启动分布式进程, 通过 torchrun 启动测试")

    context  = DistributedContext.from_torchrun()

    try: 
        yield context
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
        "op",[
            pytest.param(dist.ReduceOp.SUM,id="sum"),
            pytest.param(dist.ReduceOp.MAX,id="max"),
            pytest.param(dist.ReduceOp.MIN,id="min"),
            pytest.param(dist.ReduceOp.PRODUCT,id="product"),
        ],
)
def test_all_reduce(ctx,op):
    """
    reduce 测试脚本
    """
    tensor = torch.tensor(
        [ctx.rank + 1.0],
        device=ctx.device
    )

    result = ctx.all_reduce(tensor,op)
    values = range(1, ctx.world_size + 1)

    if op == dist.ReduceOp.SUM:
        expected = sum(values)
    elif op == dist.ReduceOp.MAX:
        expected = ctx.world_size
    elif op == dist.ReduceOp.MIN:
        expected = 1
    else:
        expected = math.prod(values)

    assert result.item() == expected


def test_all_gather(ctx):
    """
    gather 测试脚本
    """
    tensor = torch.tensor(
        [ctx.rank],
        device = ctx.device
    )

    result = ctx.all_gather(tensor)

    assert len(result) == ctx.world_size
    assert [item.item() for item in result] == list(range(ctx.world_size))


def test_root_call(ctx):
    calls = []

    def action(value):
        calls.append(ctx.rank)
        return value
    result = ctx.root_call(action, value = 42)
    if ctx.rank == 0:
        assert calls == [0]
        assert result == 42
    else:
        assert calls == []
        assert result is None