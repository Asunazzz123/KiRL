"""Base algorithm interface and registry."""
from dataclasses import dataclass
import torch


@dataclass
class RolloutData:
    """
    Rollout 产物对象
    """
    rewards: torch.Tensor
    log_probs: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor = None

    def to(self,device):
        return RolloutData(
            rewards=self.rewards.to(device),
            log_probs=self.log_probs.to(device),
            response_mask=self.response_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            values = self.values.to(device) if self.values is not None else None
        )

