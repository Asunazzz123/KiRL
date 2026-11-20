# ZeRO-Offload++ 的实现与 KiRL 实验设计

2026-11-20。KiRL 源码基于 `develop` 的 `2276316`。DeepSpeed 的逐步分析固定到 `v0.19.1`，另对 `v0.19.9` 核对了分组、混合优化器、学习率和常规 checkpoint 路径。

KiRL 有两个可以独立交付的结果。其一，用已有的 `SFTTrainer` 在同一个训练负载上测出 DeepSpeed 全 GPU、全 CPU 和混合更新的显存与吞吐关系。其二，在 KiRL 内部实现一个具有标准 `torch.optim.Optimizer` 接口的 CPU AdamW，让后续的分片、混合更新和流水优化都有数值基线。前者可以直接复用 DeepSpeed，后者的第一版只需要接通优化器构造和更新，不必接管整个 Trainer。

## 1. DeepSpeed 把哪些工作移到了 CPU

ZeRO-Offload 的基础论文研究如何在 GPU 负责前反向时，把优化器状态和更新计算移到 CPU，并限制设备间通信。团队公开的 ZeRO-Offload++ 实现称为 **Twin-Flow**，针对全 CPU Adam 更新成为瓶颈、GPU 仍有显存余量的情形。每个 rank 把一部分 ZeRO-3 参数子组交给 CPU，剩余子组交给 GPU，二者共同完成**同一个** optimizer step。参数的 ZeRO 分片、前反向计算和 optimizer 状态的放置是三个不同的选择。官方发布时将减少 memcpy 和 CPUAdam 内即时类型转换列为后续优化。[ZeRO-Offload 基础论文][paper] · [Offload++ 团队发布][announce] · [Twin-Flow 官方教程][tutorial]

在 `v0.19.1` 的常规 ZeRO-3 路径中，一次更新经过以下步骤。[Stage 3 源码][stage3]

| 时机 | 做什么 | 源码入口 |
| --- | --- | --- |
| 初始化分组 | 保留 optimizer 参数组边界。在每组中按参数顺序累加本 rank 的 `partition_numel()`，达到 `sub_group_size` 才封组。单个参数分片可以超过阈值 | `_create_fp16_sub_groups` |
| 决定设备 | 设全部子组数量为 `G`、配置 CPU 比例为 `r`。索引小于 `int((1-r)*G)` 的组由 GPU 更新，后续组由 CPU 更新。常规 FP32 master 放在对应设备 | `_create_fp32_partitions` |
| 分配状态 | 部分卸载要求主优化器为 `DeepSpeedCPUAdam`，GPU 侧实际构造 `torch.optim.AdamW`。启用 optimizer offload 后，覆盖所有本地分片的梯度累积平坦缓冲位于 CPU | Stage 3 构造函数及梯度缓冲初始化 |
| 更新与写回 | 在累积边界做 overflow 和全局范数处理，逐组准备梯度、反缩放与裁剪，按组选择 optimizer，再把 master 的更新写回训练用参数分片 | `step`、`_optimizer_step`、`_reassign_or_swap_out_partitioned_parameters` |

例如 4 个子组分别有 `6、6、3、1` 百万个本地元素，`ratio=0.5` 会把后两组分给 CPU。CPU 更新 2/4 组，却只更新 4/16 的元素。如果只有 1 个组，任何正比例都会落到全 CPU。`sub_group_size` 的默认值是十亿个元素。小模型先看实际组数，才能决定是否需要降低阈值。`v0.19.9` 的设备切分规则相同。[子组构造与放置][stage3] · [ZeRO 配置][zero-config] · [v0.19.9 源码][stage3-new]

源码以 Python 循环逐组提交工作，GPU kernel 的异步执行给 CPU 更新留下重叠机会。配置出现两个设备，只能确认工作被分配到两侧。要判断确实重叠多少，需要把 CPUAdam、GPU kernel 和 H2D/D2H memcpy 放在同一条时间线上。团队报告的最高约 6× 对应 8×H100 80GB 上的 OPT-30B，约 3× 对应 8×A100 40GB 上的 OPT-13B，对照是全卸载 ZeRO-Offload。[团队实验条件][announce] · [逐组更新循环][stage3]

## 2. 在 KiRL 中构造可比较的 SFT 实验

当前 `SFTTrainer` 接收 `TrainingArguments`，并在构造函数内部解析模型。因此先构造带 `deepspeed` 配置的 arguments，再传入模型路径，可以让 ZeRO-3 参与 `from_pretrained` 的加载过程。数据要用 KiRL 的 `BaseDataset` 封装，里面每条样本提供 `input_ids`、`attention_mask` 和 `labels`。当前 Hydra 示例只保存解析后的配置，engine registry 仅有 `ddp` 和 `fsdp`。第一轮从 `SFTTrainer.train()` 进入，省去 Hydra、rollout 和 reward 的影响。[KiRL SFT](../kirl/algorithms/sft.py) · [数据入口](../kirl/datasets/base.py) · [现有 Hydra 示例](../examples/train_grpo_gsm8k.py) · [HF ZeRO-3 初始化说明][hf-ds]

以下是目标 CUDA 环境的入口示意。`model_path`、`token_rows`、`run_dir` 由实验驱动提供。`subgroup_elements` 在首次检查分组后选定，然后在各组实验中固定。建议从能够稳定装入纯 GPU ZeRO-3 的模型开始，选择支持 BF16 的 GPU。DeepSpeed 自己构造 AdamW，CPU 路径应实际加载可用的 `DeepSpeedCPUAdam` 算子。[DeepSpeed optimizer 工厂][engine] · [HF DeepSpeed 集成][hf-ds]

```python
from transformers import TrainingArguments

from kirl.algorithms.sft import SFTTrainer
from kirl.datasets.base import BaseDataset


def sft_experiment(model_path, token_rows, run_dir, cpu_ratio, subgroup_elements):
    zero = {"stage": 3, "sub_group_size": subgroup_elements}
    if cpu_ratio is not None:
        zero["offload_optimizer"] = {
            "device": "cpu", "pin_memory": True, "ratio": cpu_ratio,
        }
    ds_config = {
        "zero_optimization": zero,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto", "betas": "auto", "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "bf16": {"enabled": "auto"},
        "gradient_clipping": "auto",
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }
    args = TrainingArguments(
        output_dir=run_dir, deepspeed=ds_config, bf16=True,
        max_steps=60, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-5, lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.0, max_grad_norm=1.0, seed=19, data_seed=19,
        save_strategy="no", logging_strategy="no", report_to="none",
    )
    trainer = SFTTrainer(model=model_path, train_dataset=BaseDataset(token_rows), args=args)
    return trainer.train()
```

### 实验矩阵

| 组别 | `cpu_ratio` | 能回答的问题 |
| --- | --- | --- |
| A · ZeRO-3 全 GPU | `None`，省略 `offload_optimizer` | 同一模型与 batch 全在 GPU 更新时的成本和峰值 |
| B · ZeRO-3 全 CPU 更新 | `1.0` | 全 CPU Adam 的显存收益、主机开销和传输成本 |
| C · Twin-Flow | `0.25 / 0.50 / 0.75`，合并实际放置相同的组 | 中间比例在相同 GPU 显存预算下能否比 B 更快 |
| D · 可选诊断 | `0.0` 且 `device=cpu` | 所有更新落在 GPU 时，保留 offload 缓冲和执行器路径有多大代价 |

D 同时改变了累积缓冲位置和 GPU optimizer 实现。若需要把差值归因于 memcpy，应额外核对 kernel 和传输时间。全部组保持 `offload_param` 关闭、master 与 moments 的 dtype 相同。固定初始 checkpoint、有效 token、每 rank batch、序列长度、ZeRO 通信配置、activation checkpointing、GPU 数、CPU 核绑定和 `OMP_NUM_THREADS`。记录 CPU/GPU 拓扑、主机带宽条件、DeepSpeed/PyTorch/CUDA 版本和实际 optimizer 类。[梯度缓冲与 backup optimizer][stage3] · [CPU 核绑定说明][cpu-tutorial]

**数值验收先于性能采样。** 固定一批无 dropout 数据，比较同一初值下 1 步及多步的 `Δweight`、FP32 master、`exp_avg`、`exp_avg_sq` 和更新计数。同步记录相同输入的 logits/loss、全局 grad norm 与实际 LR。比较误差时按 dtype 设阈值，同时记录相对更新误差，防止很小的学习率让错误更新看似通过。首轮用恒定非零 LR、`weight_decay=0` 和 `GA=1`。另测 decay/no-decay 分组、变 LR、overflow 跳步、GA>1、保存恢复。KiRL 的 SFT `compute_loss` 当前没有把 `num_items_in_batch` 传给模型，应先让 GA>1 的梯度与同一批有效 token 的大 batch 对齐。[SFT loss](../kirl/algorithms/sft.py) · [HF Trainer][hf-trainer]

**分组与性能记录。** 在 engine 初始化后、第一次有效更新后分别按 rank 记录 `fp16_partitioned_groups_flat_numel`、master/moments 的真实设备和 dtype。开启 offload 的各组额外读取 `subgroup_to_device`，纯 GPU 组从实际 tensor 设备统计。用实际元素数计算 CPU 比例，区分 ZeRO padding 和有效元素。建议先预热 10 步，再计量 50 步，重复至少 3 次。窗口边界同步 GPU，以最慢 rank 的墙钟时间为分母，分子是全 rank 实际用于 next-token loss 的 token 总数。报告吞吐、step 中位数和 P95、初始化与稳态峰值显存（allocated/reserved）、主机 RSS/锁页内存、D2H/H2D 字节与时长。另开短 trace 看更新重叠，避免逐子组同步破坏吞吐测量。随后独立扫描最大可行 batch 或模型规模，记录 OOM 发生阶段。[KiRL Probe](../kirl/profiler/probe.py) · [Stage 3 内部字段][stage3]

**计时位置。** KiRL 的 `train.microstep` 包住 HF `training_step()`。本项目环境中的 Accelerate 1.15.0 在最后一个 microbatch 的 `backward()` 内调用 `engine.step()`，外层 optimizer wrapper 的 `step()` 是空操作。GPU 内部阶段用 CUDA events/NVTX 和 CPU trace，CPUAdam 用主机时钟。KiRL `store_metrics` 会对 loss 调 `.item()`，所有组要使用相同的日志路径。只计 Trainer 外层的两个优化器 callback 会漏掉真正的更新。[KiRL 基类](../kirl/algorithms/base.py) · [Accelerate DeepSpeed wrapper][accelerate-ds]

**版本核验。** `v0.19.1` 和 `v0.19.9` 的 GPU backup 第一个参数组独立读取初始 LR，Stage 3 的 `set_lr()` 遍历主 optimizer 的参数组。常规 `_rigid_state_dict()` 保存主 optimizer 的 state dict 和全部 master，函数中未显式合并 backup optimizer 的 moments。要在所选版本上分别核对 CPU/GPU 子组实际 LR，以及“连续 N+1 步”和“N 步保存、恢复再更新 1 步”的状态差异。这是针对具体源码的验证项，尚无 KiRL GPU 运行结果。[v0.19.1 Stage 3][stage3] · [v0.19.9 Stage 3][stage3-new]

## 3. KiRL 基础模块如何加入 CPU offload

### 3.1 只卸载优化器

建议新增一个 `CPUOffloadAdamW`，由 SFT/GRPO 的构造函数透传 HF 已有的 `optimizer_cls_and_kwargs`。HF 继续建立 decay/no-decay 参数组，负责累积、梯度同步、AMP、裁剪、scheduler 和回调。新的 optimizer 只负责 master、moments、step 计数与写回。`param_groups` 始终指向训练模型参数，使 scaler 和 scheduler 能操作正确的梯度和学习率。CPU master 作为 optimizer state 保存。DeepSpeed 实验继续由 DeepSpeed 拥有其 optimizer。[HF optimizer 注入点][hf-opt] · [KiRL SFT](../kirl/algorithms/sft.py) · [KiRL 基类](../kirl/algorithms/base.py)

具体落点可以是 `kirl/engine/optim.py`，这是建议的新文件。算法层只透传构造参数，配置层单独表示 `optimizer_offload`，保留已有 `engine.cpu_offload` 的 FSDP 参数卸载语义。等 Hydra 训练入口接通后，由 adapter 将“优化器仅 CPU 更新”“FSDP 参数和梯度一并卸载”“DeepSpeed 按比例更新”翻译为各自后端的配置。三个功能对应不同的状态所有者与传输路径。[当前 schema](../kirl/config/schema.py) · [当前 adapter](../kirl/config/adapters.py)

同步第一版可以按下列数据流实现。单 GPU 或 DDP 使用普通参数，扩展到 FSDP2 时只读写当前 rank 的本地 shard。分布式梯度归约与全局裁剪在更新开始前完成。

```text
GPU 上训练用的参数或参数分片
    → forward / backward → 已归约、反缩放、裁剪的梯度
    → D2H 复制本地梯度
    → CPU 用当前参数组的 lr / betas / eps / weight_decay 更新 FP32 master、m、v
    → 将 master 转为模型参数 dtype，H2D 原位写回已有参数或分片
    → 写回完成 → 下一个 forward、checkpoint 或 rollout 权重导出
```

实现时保留四个不变量。

1. **一个训练参数只有一份更新状态。** 按唯一参数去重 tied weights，跳过 `grad is None` 和冻结参数。每次成功更新只增加对应参数的计数。AMP 检出 overflow 时，两侧都不得写回或发布新版本。首版先用普通 FP32 CPU AdamW 验证数学结果，之后再用优化的 CPUAdam 内核做性能比较。
2. **状态跟随最终分片。** 本地 HF/Accelerate 的 FSDP2 路径会在 wrap 后替换 `optimizer.param_groups` 中的参数。等替换完成再绑定 CPU master 和 moments，避免从构造时的整模型引用分配状态。模型由 FSDP2 管理时，更新本 rank shard 即可。权重导出才有全量 gather 的需求。[HF Trainer][hf-trainer] · [Accelerate 准备逻辑][accelerator] · [FSDP2 API][fsdp2]
3. **checkpoint 能恢复下一次更新。** 保存参数组、每个参数的 master、moments、step，并在 restore 后校验 dtype 和设备。多 rank 时保存参数全名、分片形状与位置，通过分片 checkpoint 表达布局，防止将本地 CPU tensor 当作完整的 replicated state。先在相同 world size 下做下一步对齐。KiRL 的算法元数据和仅保存模型的接口不足以验收 optimizer 恢复，比较时走 Trainer 的完整训练 checkpoint。[KiRL checkpoint](../kirl/engine/checkpoint.py)
4. **统计真实状态字节。** 单个 FP32 master 和两个 FP32 moments 共 12 字节/元素。只有 GPU 基线单独持有三份时，才能把 12 字节全部算为可迁移的常驻量。若 GPU 模型参数本身已是 FP32 且还要保留，两个 moments 对应 8 字节/元素，CPU 另设 master 会增加主机占用。梯度、staging buffer、all-gather 工作区和 activations 分开归账。

### 3.2 用同一个优化器扩展到 Twin-Flow 式实验

先支持静态分组。记录每个子组的原参数组、参数范围、本地元素数和更新设备。CPU 子组沿用同步基线，GPU 子组在本设备用相同 AdamW 语义更新。可以直接复用 DeepSpeed 的按子组数量切分规则，与官方做对照。若再探索按元素数或字节预算分配，把它作为独立实验因素，并保存固定的放置表。

在**同一张放置表和同一对 CPU/GPU 内核**上比较两种调度。串行组等待每个子组完全更新、写回后再处理下一组。流水组使用有限个 pinned staging buffers、copy stream 和 CUDA events，GPU 更新与另一组的 D2H、CPUAdam 或 H2D 可重叠。CPU 读取梯度前等待对应 D2H event，buffer 重用前等待消费者结束，forward / rollout / checkpoint 读取前等待全部更新完成。对这两组比较，才可以单独估计重叠贡献。对“全 CPU”和“串行混合”的比较，则测的是工作放置与内核变化的综合效果。

初期只需单卡验证更新，然后双 rank 验证 FSDP2 本地 shard 和恢复，再打开流水。选择最优比例的条件是给定显存预算下的实际 tokens/s。当 CPU 内存带宽、PCIe 传输或前反向占据大部分 step 时间时，最快比例可以不同。[Twin-Flow 官方实现目标][announce]

### 3.3 接通 FSDP 自带的参数和梯度卸载

KiRL 的 `engine.cpu_offload` 目前经配置适配器翻译成 HF FSDP 的 `offload` 选项。另一路独立的 `FSDPBackend` 在 FSDP1 的构造函数以及 FSDP2 的逐 block 和 root `fully_shard()` 调用中都没有传这个开关。`offload_fsdp_optimizer()` 仅移动已经创建的 state tensor。它适合表达状态搬运，本身不负责把梯度送到 CPU、执行更新或写回权重。[配置适配器](../kirl/config/adapters.py) · [FSDPBackend 与 helper](../kirl/distributed/fsdp_backend.py)

如果需求是给这个基础 backend 接通 PyTorch 原生卸载，FSDP1 传 `cpu_offload=CPUOffload(offload_params=True)`，FSDP2 给每个 block 和 root 传 `offload_policy=CPUOffloadPolicy(pin_memory=True)`。FSDP2 会在 all-gather 前把 CPU 参数分片搬到 GPU，反向后把梯度分片搬回 CPU，优化器直接更新 CPU 分片。这种放置会改变前反向的传输量，可以单列为参数也卸载的容量对照。修改独立 backend 只影响调用它的入口，Trainer/Accelerate 管理的模型要通过对应 FSDP plugin 配置。KiRL 独立 backend 目前还在分片前对整模型 `.to(device)`，FSDP2 随后广播整模型。需要解决大模型加载 OOM 时，单独改初始化路径并测其峰值。[FSDP1 API][fsdp1] · [FSDP2 API][fsdp2] · [KiRL 分片前初始化](../kirl/distributed/fsdp_backend.py)

## 4. 从 SFT 迁移到 GRPO

`GRPOTrainer` 在 `beta>0` 时创建 frozen reference，并对所有 beta 注册 rollout 权重同步回调。DeepSpeed reference 创建和 DeepSpeed policy 参数导出在当前基类里都显式抛出 `NotImplementedError`。所以 `beta=0` 可以先简化 reference，但首次成功更新后仍需要 ZeRO-3 导出。当前 vLLM backend 通过 embedded driver 的 `load_weights` 消费参数。[GRPO 注册回调](../kirl/algorithms/grpo.py) · [基类与导出边界](../kirl/algorithms/base.py) · [vLLM 权重加载](../kirl/engine/inference/vllm_engine.py)

迁移可以分三步验收。

1. 固定一份 rollout 的 token、mask、old logprob 和 advantage，先比较后端更新得到的 policy 增量，避免采样随机性掩盖 optimizer 差异。
2. 在 `beta=0` 下接通按参数或有界桶的 `deepspeed.zero.GatheredParameters`。所有训练 rank 按同一顺序进入 collective，在 gather 上下文有效时完成消费和复制。先用每 rank 一份 embedded、TP=1 的 vLLM 验证，再扩展到多 worker 的实际广播和确认。成功更新后，全部相关 worker 确认版本，下一轮才采样。发生 overflow 跳步时保持原版本。用固定 token 在训练侧和 rollout 侧核对 logprob，并检查第二轮确实读取更新后的权重。[DeepSpeed 参数 gather API][zero-config] · [KiRL 权重同步回调](../kirl/algorithms/base.py)
3. 对 `beta>0` 初始化与 policy 独立的冻结 reference，验证多步更新后 reference 在相同 token 上输出不变。分别报告 rollout、训练更新、权重 gather/load 和整轮用时。这样能看出 SFT 中的收益在 vLLM、reference 和训练共同占用显存时还能保留多少。

## 本次验证范围

本地项目 `.venv` 实测 PyTorch `2.15.0`、Transformers `5.22.2`、Accelerate `1.15.0`，没有 DeepSpeed、vLLM 或 CUDA。在同一环境中，用离线初始化的一层 GPT-2、两条长度为 8 的固定样本、`GA=1` 和普通 CPU AdamW 执行了两步真实的 `SFTTrainer.train()`。`global_step=2`，embedding 最大绝对变化为 `0.00200136`，平均 loss 为 `2.090904`。这项检查验证了 KiRL 的 SFT 入口和参数更新。DeepSpeed 混合更新、原生 CPU offload、多 rank 恢复、vLLM 同步与目标节点吞吐仍属于待执行实验。

[paper]: https://www.usenix.org/conference/atc21/presentation/ren-jie
[announce]: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-offloadpp/README.md
[tutorial]: https://github.com/deepspeedai/Megatron-DeepSpeed/blob/main/examples_deepspeed/offload_pp/README.md
[stage3]: https://github.com/deepspeedai/DeepSpeed/blob/v0.19.1/deepspeed/runtime/zero/stage3.py
[stage3-new]: https://github.com/deepspeedai/DeepSpeed/blob/v0.19.9/deepspeed/runtime/zero/stage3.py
[engine]: https://github.com/deepspeedai/DeepSpeed/blob/v0.19.1/deepspeed/runtime/engine.py
[zero-config]: https://deepspeed.readthedocs.io/en/stable/zero3.html
[cpu-tutorial]: https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/zero-offload.md
[hf-ds]: https://huggingface.co/docs/transformers/en/deepspeed
[hf-opt]: https://huggingface.co/docs/transformers/main/optimizers
[hf-trainer]: https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/trainer.py
[accelerator]: https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/accelerator.py
[accelerate-ds]: https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/deepspeed.py
[fsdp1]: https://docs.pytorch.org/docs/2.15/fsdp.html
[fsdp2]: https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html

## zero-cpuoffload 分支使用

从 KiRL 根目录运行以下命令。省略 `--cpu-ratio` 是全 GPU 基线，设为 1 是全 CPU 更新。原生模式使用同步 FP32 AdamW 和本 rank 的梯度，DeepSpeed 模式使用官方 ZeRO-3。示例报告包含初始化和预热的端到端吞吐。

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=2 -m examples.bench_sft_offload --backend deepspeed --model /path/to/model --cpu-ratio 0.5 --sub-group-size 1000000
uv run --no-sync python -m examples.bench_sft_offload --backend native --model /path/to/model --cpu-ratio 1
```

SFTTrainer 和 GRPOTrainer 可通过 `optimizer_cls_and_kwargs=(CPUOffloadAdamW, {"lr": 1e-5})` 选择原生实现。GRPO 的 DeepSpeed 权重导出按参数 gather，冻结 reference 仍需独立实现。原生优化器 checkpoint 以当前 rank 的本地形状恢复，跨布局重分片与异步流水仍待实现。
