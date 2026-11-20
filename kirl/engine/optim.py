"""Synchronous device placement for interchangeable first-order optimizers."""
from copy import deepcopy

import torch
from torch.distributed.tensor import DTensor


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class HybridDeviceOptimizer(torch.optim.Optimizer):
    """Update FP32 local master weights using the supplied torch optimizer class.

    CPU placement is by whole subgroup. Matrix-coupled algorithms require full
    matrices or an inner optimizer that explicitly supports local shards.
    Pass algorithm options (lr, momentum, betas, ...) as keyword arguments.
    """
    def __init__(self, params, optimizer_cls, *, cpu_ratio=1.0,
                 sub_group_size=10**9, **optimizer_kwargs):
        if not 0 <= cpu_ratio <= 1 or sub_group_size < 1:
            raise ValueError("Invalid CPU ratio or subgroup size")
        super().__init__(params, optimizer_kwargs)
        self.optimizer_cls = optimizer_cls
        self.cpu_ratio = cpu_ratio
        self.sub_group_size = sub_group_size
        self._inner = []

    def _groups(self):
        buckets = []
        for group in self.param_groups:
            bucket, size = [], 0
            for param in group["params"]:
                bucket.append(param)
                size += _local(param).numel()
                if size >= self.sub_group_size:
                    buckets.append((group, bucket))
                    bucket, size = [], 0
            if bucket:
                buckets.append((group, bucket))
        return buckets

    def _prepare(self):
        buckets = self._groups()
        cutoff = int((1 - self.cpu_ratio) * len(buckets))
        prepared = []
        for index, (group, params) in enumerate(buckets):
            masters = []
            for param in params:
                if (self.optimizer_cls is getattr(torch.optim, "Muon", None)
                        and isinstance(param, DTensor)
                        and any(p.is_shard() for p in param.placements)):
                    raise ValueError("Muon requires a complete matrix, not a local FSDP shard")
                local = _local(param)
                device = "cpu" if index >= cutoff else local.device
                state = self.state[param]
                if "master" not in state:
                    state["master"] = local.detach().to(device=device, dtype=torch.float32, copy=True)
                elif state["master"].device != torch.device(device):
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and key != "step":
                            state[key] = value.to(device)
                masters.append(state["master"])
            options = {k: v for k, v in group.items() if k not in ("params", "param_names")}
            inner = self.optimizer_cls([dict(options, params=masters)], **self.defaults)
            for param, master in zip(params, masters):
                inner.state[master] = {k: v for k, v in self.state[param].items() if k != "master"}
            prepared.append((group, params, inner))
        self._inner = prepared

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        self._inner = []

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        if not self._inner:
            self._prepare()
        for group, params, inner in self._inner:
            target = inner.param_groups[0]
            target.update({k: v for k, v in group.items() if k not in ("params", "param_names")})
            for param, master in zip(params, target["params"]):
                master.grad = (None if param.grad is None else
                               _local(param.grad).detach().to(master))
            inner.step()
            for param, master in zip(params, target["params"]):
                if master.grad is not None:
                    _local(param).copy_(master)
                    master.grad = None
                self.state[param].update(inner.state[master])
        return loss

    def load_state_dict(self, state_dict):
        # The default loader casts floating states to the training weight dtype.
        # Restore the original FP32 tensors directly on their update device.
        self._inner = []
        saved = state_dict["state"]
        super().load_state_dict({**state_dict, "state": {}})
        ids = {p: key for group, old in zip(self.param_groups, state_dict["param_groups"])
               for p, key in zip(group["params"], old["params"])}
        buckets = self._groups()
        cutoff = int((1 - self.cpu_ratio) * len(buckets))
        for index, (_, bucket) in enumerate(buckets):
            for p in bucket:
                if ids[p] in saved:
                    device = "cpu" if index >= cutoff else _local(p).device
                    self.state[p] = {k: v.to("cpu" if k == "step" else device, copy=True)
                                     if isinstance(v, torch.Tensor) else deepcopy(v)
                                     for k, v in saved[ids[p]].items()}


class CPUOffloadAdamW(HybridDeviceOptimizer):
    """AdamW compatibility entry point, including existing optimizer checkpoints."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, *, cpu_ratio=1.0, sub_group_size=10**9):
        super().__init__(params, torch.optim.AdamW, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, cpu_ratio=cpu_ratio, sub_group_size=sub_group_size)
