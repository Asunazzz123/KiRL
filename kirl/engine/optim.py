"""Synchronous AdamW with FP32 optimizer state on CPU or the parameter device."""
import torch
from torch.distributed.tensor import DTensor


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class CPUOffloadAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, *, cpu_ratio=1.0, sub_group_size=10**9):
        if not 0 <= cpu_ratio <= 1 or sub_group_size < 1:
            raise ValueError("Invalid CPU ratio or subgroup size")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.cpu_ratio = cpu_ratio
        self.sub_group_size = sub_group_size

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

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        buckets = self._groups()
        cutoff = int((1 - self.cpu_ratio) * len(buckets))
        for index, (group, bucket) in enumerate(buckets):
            for param in bucket:
                if param.grad is None:
                    continue
                local = _local(param)
                device = "cpu" if index >= cutoff else local.device
                grad = _local(param.grad).to(device=device, dtype=torch.float32)
                state = self.state[param]
                if not state:
                    master = local.detach().to(device=device, dtype=torch.float32, copy=True)
                    state.update(master=master, exp_avg=torch.zeros_like(master),
                                 exp_avg_sq=torch.zeros_like(master), step=torch.tensor(0.))
                torch.optim._functional.adamw(
                    [state["master"]], [grad], [state["exp_avg"]], [state["exp_avg_sq"]],
                    [], [state["step"]], foreach=False, fused=False, amsgrad=False,
                    beta1=group["betas"][0], beta2=group["betas"][1],
                    lr=group["lr"], weight_decay=group["weight_decay"],
                    eps=group["eps"], maximize=False)
                local.copy_(state["master"])
        return loss

    def load_state_dict(self, state_dict):
        # The default loader casts floating states to the training weight dtype.
        # Restore the original FP32 tensors directly on their update device.
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
                                     for k, v in saved[ids[p]].items()}
