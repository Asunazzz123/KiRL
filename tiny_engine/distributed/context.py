# distributed/context.py
"""Distributed context for multi-GPU coordination."""
from __future__ import annotations

import os
import logging
import torch
import torch.distributed as dist


from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from utils.logging import Log
logger = Log.get_logger(__name__)


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


    @classmethod
    def from_torchrun(cls) -> "DistributedContext":
        # 从环境变量中获得 LOCAL_RANK
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            if not dist.is_nccl_available():
                raise RuntimeError(
                    "Current PyTorch environment does not support NCCL"
                )
            # 支持CUDA的时候使用nccl通信
            device = torch.device("cuda",local_rank)
            torch.cuda.set_device(device=device)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        logger.info(
            "Initializing process group: "
            "rank=%s local_rank=%s world_size=%s backend=%s device=%s",
            os.environ.get("RANK","?"),
            local_rank,
            os.environ.get("WORLD_SIZE","?"),
            backend,
            device
        )
        # 从环境变量中初始化进程组
        dist.init_process_group(backend=backend,init_method="env://")

        logger.info(
            "Process group initialized: rank=%s world_size=%s",
            dist.get_rank(),
            dist.get_world_size(),
        )

        return cls(
            rank = dist.get_rank(),
            local_rank = local_rank,
            world_size = dist.get_world_size(),
            device = device
        )

    def barrier(self):
        """阻塞所有的分布式进程，直到同步所有rank"""
        dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        """
        根据reduce op 对 tensor 在所有rank 上求和
        Args:
            tensor: 输入的tensor
            op: torch.distributed 的reduce 规则
        Returns:
            tensor: 输出求和后的向量
        """
        dist.all_reduce(tensor,op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """收集所有rank的tensor"""
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)] # 按照tensor 形状创建0 tensor
        dist.all_gather(gathered,tensor) # 将 tensor合并后分发给每一个rank
        return gathered

    def broadcast(self,tensor: torch.Tensor, src: int) -> torch.Tensor:
        """
        将src指向的tensor广播到其他rank
        Args:
            tensor: 输入tensor
            src: 广播的rank
        Returns:
            tensor: 广播后的tensor
        """
        dist.broadcast(tensor,src)
        return tensor

    T = TypeVar("T")
    def root_call(self, fn: Callable[...,T], *args, **kwargs) -> T | None:
        """
        只执行 rank0 的函数
        """
        if self.rank == 0:
            return fn(*args,**kwargs)

        return None

    def agree(self, keys: str, value: Any):
        """
        通过每一个rank 的value hash化后比较sha256指纹，检查所有rank在ckeckpoint的数据一致性
        """
        import hashlib
        import pickle

        value_bytes = pickle.dumps(value)
        value_hash = hashlib.sha256(value_bytes).hexdigest()
        hash_tensor = torch.tensor(list(value_hash),dtype=torch.uint8, device=self.device)

        all_hashes = [torch.zeros_like(hash_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_hashes, hash_tensor)

        first_hash = all_hashes[0].item()
        for rank, h in enumerate(all_hashes):
            if h.item() != first_hash:
                raise RuntimeError(
                    f"Disagreement on '{keys}': rank 0 has hash {first_hash}, "
                    f"rank {rank} has hash {h.item()}"
                )








