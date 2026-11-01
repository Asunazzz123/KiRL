"""FSDP configuration and utilities."""

import torch



class FSDPPConfig:
    compute_dtype: torch.dtype = torch.dtype16
    param_dtype: torch.dtype = torch.float32
    learing_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    sharding_strategy = str = "FULL_SHARD" 
    cpu_offload = bool = False




