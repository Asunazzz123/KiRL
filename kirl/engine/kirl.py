"""KiRL core implementation."""
import torch
import torch.nn as nn
from typing import Any, Callable, TypeVar

def _init_model(self,model_factory:Callable, fsdp_config):
    ...