"""GRPO algorithm implementation."""

import torch

from collections.abc import Callable
from typing import TypeAlias


from .base import _BaseTrainer
from ..utils.logging import Log
logger = Log.get_logger(__name__, algorithm="grpo", role="trainer")

RewardFunc: TypeAlias = Callable[..., list[float | None]]

class GRPOTrainer(_BaseTrainer):
    def __init__(
        self,
        model: str,
        reward_fn: RewardFunc | list[RewardFunc] | None = None,
        # train_dataset:
        # eval_dataset:
    ):
        ...