"""Distributed context for multi-GPU coordination."""
import torch

from typing import Any, Callable, TypeVar

class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def from_torchrun(cls) -> DistributedContext:
        return DistributedContext(
            rank = cls.rank,
            local_rank = cls.local_rank,
            world_size = cls.world_size,
            device = cls.device
        )

    def barrier(self):
        """阻塞所有的分布式进程，直到同步所有rank"""
        torch.distributed.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: torch.distributed.ReduceOp) -> torch.Tensor:
        """
        根据reduce op 对 tensor 在所有rank 上求和
        Args:
            tensor: 输入的tensor
            op: torch.distributed 的reduce 规则
        Returns:
            tensor: 输出求和后的向量
        """
        torch.distributed.all_reduce(tensor,op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """收集所有rank的tensor"""
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)] # 按照tensor 形状创建0 tensor
        torch.distributed.all_gather(gathered,tensor) # 将 tensor合并后分发给每一个rank
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
        torch.distributed.broadcast(tensor,src)
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
        hash_tensor = torch.tensor([int(value_hash[:15], 16)],dtype=torch.int64, device=self.device)

        all_hashes = [torch.zeros_like(hash_tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_hashes, hash_tensor)

        first_hash = all_hashes[0].item()
        for rank, h in enumerate(all_hashes):
            if h.item() != first_hash:
                raise RuntimeError(
                    f"Disagreement on '{keys}': rank 0 has hash {first_hash}, "
                    f"rank {rank} has hash {h.item()}"
                )






