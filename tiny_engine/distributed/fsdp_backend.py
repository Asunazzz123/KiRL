import torch


import torch.distributed.fsdp as fsdp
import torch.distributed as dist
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from dataclasses import dataclass
from .fsdp_config import FSDPConfig
from .context import DistributedContext
from ..profiler.probe import NullProbe
from functools import partial



class FSDPBackend:
    def __init__(
        self,
        config: FSDPConfig,
        context: DistributedContext,
        probe = None,
    ):
        self.config = config
        self.context = context
        self.probe = probe if probe is not None else NullProbe()


    def prepare_mode(self,model:torch.nn.Module, transformer_layer_cls):
        self._validate_model(model,transformer_layer_cls)
        if self.config.mode == "FSDP":
            self._prepare_fsdp1(model,transformer_layer_cls)

        else:
            self._prepare_fsdp2(model,transformer_layer_cls)


    def _validate_model(model,transformer_layer_cls) -> None:
        if not isinstance(model,torch.nn.Module):
            raise TypeError("Model must be torch.nn.Module")

        if not isinstance(transformer_layer_cls,(set,tuple,list)):
            raise TypeError(
                "transformer_layer_cls must be a collection of nn.Module classes"
            )
        if not transformer_layer_cls:
            raise ValueError("transformer_layer_cls must not be empty")
        for layer_cls in transformer_layer_cls:
            if not isinstance(layer_cls, type) or not issubclass(
                layer_cls, torch.nn.Module
            ):
                raise TypeError(
                    "Each transformer layer type must be an nn.Module subclass"
                )

        layer_types = tuple(transformer_layer_cls)

        if not any(
            name and isinstance(module, layer_types)
            for name, module in model.named_modules()
        ):
            raise ValueError(
                "No child module matches transformer_layer_cls"
            )

        wrapped_types = [fsdp.FullyShardedDataParallel]
        fsdp2_module = getattr(fsdp, "FSDPModule", None)
        if fsdp2_module is not None:
            wrapped_types.append(fsdp2_module)

        for name, module in model.named_modules():
            if isinstance(module, tuple(wrapped_types)):
                raise ValueError(
                    f"Module {name or '<root>'} is already FSDP-managed"
                    )
        has_trainable_parameter = False

        for name, parameter in model.named_parameters():
            if isinstance(parameter, UninitializedParameter):
                raise ValueError(f"Parameter {name} is not initialized")

            if parameter.is_meta:
                raise ValueError(f"Meta parameter is not supported: {name}")

            if (
                not parameter.is_floating_point()
                or parameter.layout != torch.strided
            ):
                raise ValueError(
                    f"Only dense floating-point parameters are supported: {name}"
                )

            has_trainable_parameter |= parameter.requires_grad

        if not has_trainable_parameter:
            raise ValueError("Training model has no trainable parameters")

        # buffer 可以是整数类型，但也必须已物化
        for name, buffer in model.named_buffers():
            if isinstance(buffer, UninitializedBuffer):
                raise ValueError(f"Buffer {name} is not initialized")

            if buffer.is_meta:
                raise ValueError(f"Meta buffer is not supported: {name}")



    def _prepare_fsdp1(self,model,transformer_layer_cls):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        # 模型放入device, 设置基础参数存储精度
        model = model.to(
            device = self.context.device,
            dtype = self.config.param_dtype
        )
        # 子模块配置FSDP单元
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls = set(transformer_layer_cls)
        )
        # 混合精度设置
        #TODO: 混合精度字段加入config
        mixed_precesion = MixedPrecision(
            param_dtype=self.config.compute_dtype, # 计算精度
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )

        return FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=ShardingStrategy[self.config.sharding_strategy],
            mixed_precision=mixed_precesion,
            device_id=self.context.device,
            sync_module_states=True,
            use_orig_params=True
        )


    def _prepare_fsdp2(self,model,transformer_layer_cls):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import (
            fully_shard,
            MixedPrecisionPolicy
        )
        model = model.to(
            device = self.context.device,
            dtype = self.config.param_dtype
        )

        with torch.no_grad():
            states = [
                *model.named_parameters(),
                *model.named_buffers()
            ]
            for name, tensor in sorted(states,key = lambda item: item[0]):
                if tensor.layout != torch.strided:
                    raise ValueError(f"Unsupported sparse state: {name}")

                contiguous = tensor.countiguous()
                dist.broadcast(contiguous,src=0)
                if not tensor.is_contiguous():
                    tensor.copy_(contiguous)
        mesh = init_device_mesh(
            "cuda",
            (self.context.world_size,),
            mesh_dim_names=("fsdp",)
        )
        # 加载混合精度策略
        mixed_precision = MixedPrecisionPolicy(
                param_dtype=self.config.compute_dtype,
                reduce_dtype=torch.float32,
        )
        layer_types = tuple(transformer_layer_cls)
        blocks = [
            module
            for module in model.modules()
            if module is not model and isinstance(module, layer_types)
        ]
        for block in reversed(blocks):
            fully_shard(
                block,
                mesh=mesh,
                mp_policy=mixed_precision,
                reshard_after_forward=self.config.reshard_after_forward,
            )

        fully_shard(
            model,
            mesh=mesh,
            mp_policy=mixed_precision,
            reshard_after_forward=self.config.reshard_after_forward,
        )

        return model

@torch.no_grad()
def offload_fsdp_optimizer(optimizer,device_id):
    if not optimizer:
        return
    for param_group in optimizer.param_group:
        for param in param_group["params"]:
            state = optimizer.state(param)
        for key,value in state.items():
            if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
