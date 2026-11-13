"""Algorithm module for pluggable RL algorithms."""

from __future__ import annotations

from .base import _BaseTrainer
from .grpo import GRPOTrainer
from .sft import SFTTrainer

__all__ = [
    "_BaseTrainer",
    "GRPOTrainer",
    "SFTTrainer",
]
