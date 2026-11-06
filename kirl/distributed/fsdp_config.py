"""FSDP configuration and utilities."""

import torch
from typing import Literal
from packaging import version
from dataclasses import dataclass
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard as FSDP2

@dataclass
class FSDPConfig:
    compute_dtype: torch.dtype = torch.float16
    param_dtype: torch.dtype = torch.float32
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    mode: Literal["FSDP","FSDP2"] = "FSDP"
    cpu_offload: bool = False
    # FSDP strategy
    sharding_strategy : Literal[
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "NO_SHARD"
    ]| None = None
    # FSDP2 strategy
    reshard_after_forward: bool | None = None

    def __post_init__(self) -> None:
        self._validate_common()
        self._configure_backend()

    def _validate_common(self) -> None:
        if self.mode not in ("FSDP", "FSDP2"):
            raise ValueError(f"Unsupported mode: {self.mode!r}")
        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be > 0")
        if not self.weight_decay >= 0:
            raise ValueError("weight_decay must be >= 0")
        if not self.max_grad_norm > 0:
            raise ValueError("max_grad_norm must be > 0")

        supported_dtypes = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )

        for name in ("compute_dtype", "param_dtype"):
            if getattr(self, name) not in supported_dtypes:
                raise ValueError(
                    f"{name} must be float16, bfloat16, or float32"
                )

        for name in ("gradient_checkpointing", "cpu_offload"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")



    def _configure_backend(self):
        """FSDP/FSDP2版本校验"""
        if self.mode == "FSDP":
            if self.reshard_after_forward is not None:
                raise ValueError(
                    "Reshard_after_forward is only used with FSDP2"
                )

            if self.sharding_strategy is None:
                self.sharding_strategy = "FULL_SHARD"

            if self.sharding_strategy not in (
                "FULL_SHARD",
                "SHARD_GRAD_OP",
                "NO_SHARD"
            ):
                raise ValueError(
                    "Unsupported sharding strategy: f{self.sharding_stratergy!r}"
                )
        else:
            if self.sharding_strategy is not None:
                raise ValueError(
                    "sharding_strategy is only used with FSDP"
                )

            if self.reshard_after_forward is None:
                self.reshard_after_forward = True

            if not isinstance(self.reshard_after_forward, bool):
                raise TypeError(
                    "reshard_after_forward must be bool"
                )

