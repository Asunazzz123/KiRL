"""Create and prepare independently owned, frozen reference models."""
import torch
import torch.nn as nn

from copy import deepcopy
from itertools import chain
from torch.distributed.tensor import DTensor
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer

from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel, FSDPModule
from accelerate import Accelerator
from accelerate.utils import DistributedType
from transformers import PreTrainedModel

def _validate_reference_source(model: torch.nn.Module) -> None:
    """Reject wrapped, distributed or unmaterialized reference inputs."""
    if not isinstance(model, nn.Module):
        raise TypeError("Reference source must be a torch.nn.Module.")
    wrapped_types = (
        nn.DataParallel, DistributedDataParallel,
        FullyShardedDataParallel, FSDPModule,
    )
    for name, module in model.named_modules():
        if isinstance(module, wrapped_types):
            raise ValueError(
                f"Create reference before distributed wrapping: {name or '<root>'}."
            )
    for name, tensor in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(tensor, (UninitializedParameter, UninitializedBuffer)):
            raise ValueError(f"Reference tensor {name!r} is uninitialized.")
        if isinstance(tensor, DTensor):
            raise ValueError(f"Reference tensor {name!r} is distributed; create before sharding.")
        if tensor.is_meta:
            raise ValueError(f"Reference tensor {name!r} is on meta; materialize weights first.")


def create_reference_policy(model: PreTrainedModel) -> PreTrainedModel:
    """Copy an unwrapped, materialized model and freeze it for reference use."""
    _validate_reference_source(model)
    ref = deepcopy(model)
    ref.requires_grad_(False)
    ref.eval()
    return ref


def prepare_reference_policy(
    ref: torch.nn.Module,
    accelerator: Accelerator
) -> torch.nn.Module:
    """Prepare once after policy setup; keep reference outside auto checkpoints.

    Supports single-device/DDP replicas and FSDP1/FSDP2 references. The caller
    owns reference persistence and must reproduce its original weights on resume.
    Registry restoration depends on Accelerate 1.15.0 internals and does not
    roll back a failed preparation; such failures must abort initialization.
    """
    backend = accelerator.distributed_type
    # DeepSpeed reference preparation is intentionally unsupported.
    supported = {
        DistributedType.NO,
        DistributedType.MULTI_CPU,
        DistributedType.MULTI_GPU,
        DistributedType.FSDP
    }
    if backend not in supported:
        raise NotImplementedError(
            f"Reference preparation does not support {backend}. "
            "Supported modes: single device, DDP, FSDP1 and FSDP2. "
            "DeepSpeed reference policies are not implemented."
        )
    _validate_reference_source(ref)

    ref.requires_grad_(False)
    ref.eval()

    # Accelerate 1.15.0 prepare_model() registers models in _models.
    # Preserve the training models and keep this reference separately owned.
    registered_models = list(accelerator._models)
    try:
        if backend == DistributedType.FSDP:
            plugin = accelerator.state.fsdp_plugin

            if plugin.fsdp_version == 1:
                # evaluation_mode=True would skip FSDP1 wrapping.
                ref = accelerator.prepare_model(ref)
            elif plugin.fsdp_version == 2:
                from accelerate.utils import fsdp2_prepare_model

                ref = fsdp2_prepare_model(accelerator, ref)
            else:
                raise NotImplementedError(
                    f"Unsupported FSDP version: {plugin.fsdp_version}"
                )
        else:
            # Apply device placement/mixed precision without DDP wrapping.
            ref = accelerator.prepare_model(
                ref,
                evaluation_mode=True,
            )
    finally:
        accelerator._models[:] = registered_models

    ref.eval()
    return ref


