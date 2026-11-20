# KiRL 的 Hydra 训练配置

从 KiRL 根目录使用项目的 uv 环境运行示例。当前 Hydra 入口完成配置组合、类型与取值校验，并将结果保存为 `resolved_config.yaml`。要执行训练，还需要用解析结果创建 Trainer 并调用 `train()`。[入口](../examples/train_grpo_gsm8k.py) · [配置适配器](../kirl/config/adapters.py)

## 选择配置组与覆盖字段

`configs/train.yaml` 的 `defaults` 先加载 `train_schema`，再加载各组 YAML。末尾的 `_self_` 表示最后合入 `train.yaml` 本身，命令行可以进一步覆盖。以 `optimizer=hybrid` 为例，这会选择 `configs/optimizer/hybrid.yaml`；`engine.optimizer=adam` 则修改已选 engine 配置里的一个字段。字典中新加的键使用 `+`，例如 `+optimizer.kwargs.eps=1e-7`。[训练入口 YAML](../configs/train.yaml) · [结构化字段定义](../kirl/config/schema.py)

| 配置组 | 默认选择 | 如何使用 |
| --- | --- | --- |
| algo | grpo | 设置组内采样和策略损失。仓库也有 ppo.yaml，但它含有当前 AlgoSchema 未定义的字段，不能直接通过该入口组合 |
| dataset | gsm8k | 可切换 math，同时按数据集调整路径和筛选条件 |
| engine | fsdp | 可切换 ddp，影响 Trainer 分布式参数 |
| optimizer | native | 可切换 hybrid，使适配器生成可注入 Trainer 的混合设备优化器 |
| rollout | vllm | 控制推理引擎资源和并行设置 |
| profiler | default | 控制采集范围、输出和后端选项 |

查看组合后的字段和值（只执行 Hydra 组合）：

```bash
uv run --no-sync python -m examples.train_grpo_gsm8k --cfg job --resolve
```

验证一组 FSDP + 混合 Adam 配置并生成快照：

```bash
uv run --no-sync python -m examples.train_grpo_gsm8k \
  engine=fsdp optimizer=hybrid engine.optimizer=adam \
  engine.cpu_offload=false optimizer.cpu_ratio=0.5 \
  optimizer.sub_group_size=1000000 +optimizer.kwargs.eps=1e-7
```

## 训练、数据与生成

下表默认值来自组合后的 GRPO + GSM8K 配置。更新步、训练序列和 rollout prompt 使用不同的计数单位。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| num_steps / batch_size / gradient_accumulation_steps | 1000 / 8 / 4 | 最大优化器更新步数、每设备每个 microbatch 的训练序列数、每次更新累积的 microbatch 数 |
| prompts_per_batch / algo.group_size | 32 / 4 | 前者是配置预留的全局 rollout prompt 数；后者实际传入生成配置，每个 prompt 返回多少条回答，至少为 2 |
| algo.beta / algo.cliprange | 0.1 / 0.2 | KL 系数和策略比率裁剪范围。beta 为 0 时跳过 reference KL，cliprange 在 [0, 1) 内 |
| algo.loss_normalization / algo.advantage_normalization | token / group | 损失可按 token 或 sequence 归一化，优势支持 group |
| dataset.train_path / eval_path / prompt_template_path | 见 GSM8K YAML | 数据和模板位置。相对路径基于启动目录解析，模板与评估路径可为 null |
| dataset.max_samples / shuffle / seed | null / true / 42 | 样本上限、打乱开关、数据随机种子。MATH 另有 difficulty_filter（1–5）和 subject_filter |
| generation.max_new_tokens / temperature / top_p / top_k | 256 / 1.0 / 0.9 / -1 | 每条回答的新 token 上限、温度与采样截断。top_p 在 (0, 1] 内，top_k 为 -1 或正整数 |
| generation.do_sample / stop / stop_token_ids / ignore_eos | true / [] / [] / false | 是否采样、停止字符串、停止 token、是否忽略 EOS |
| reward.type / reward.format_weight | gsm8k_reward / 0.1 | 奖励类型与格式分权重。切换 dataset 后要单独选择匹配的 reward.type |

[训练与采样默认值](../configs/train.yaml) · [GRPO 默认值](../configs/algo/grpo.yaml) · [GSM8K 配置](../configs/dataset/gsm8k.yaml) · [MATH 配置](../configs/dataset/math.yaml)

## 分布式引擎与优化器

| 字段 | 默认值 | 设置依据 |
| --- | --- | --- |
| engine.model_id | meta-llama/Llama-2-7b-hf | 模型 ID 或本地路径，也传入 rollout 配置 |
| engine.compute_dtype / param_dtype | bfloat16 / float32 | 均接受 float16、bfloat16、float32。适配器将 compute_dtype 转成 Trainer 的 bf16/fp16 标志；param_dtype 当前保留在解析配置里 |
| engine.sharding_strategy / cpu_offload | FULL_SHARD / false（FSDP） | 可选 FULL_SHARD、SHARD_GRAD_OP、NO_SHARD。cpu_offload 控制 FSDP 参数卸载；DDP 要求分片策略为 null 且此开关关闭 |
| engine.gradient_checkpointing / find_unused_parameters | false / false | 前者控制激活重计算，后者传给 DDP |
| engine.learning_rate / weight_decay / max_grad_norm | 1e-5 / 0 / 1 | 学习率、权重衰减、梯度裁剪阈值 |
| engine.lr_scheduler / warmup_steps | cosine / 100 | 调度器可选 cosine、linear、constant，预热按更新步计 |
| engine.optimizer | adamw | native 可选 adamw、sgd；hybrid 可选 adam、adamw、sgd、muon |
| optimizer.cpu_ratio / sub_group_size | 1.0 / 1000000000 | hybrid 使用。本地元素数达到阈值后封组，再按子组数量分配设备；CPU 实际更新的元素占比可能偏离配置比例 |
| optimizer.kwargs | {} | hybrid 透传给底层算法的额外参数，如 eps。学习率和权重衰减统一使用 engine 字段 |

对 GPU 上的参数，hybrid 的 cpu_ratio 为 0、1 时分别在 GPU、CPU 更新，介于两者之间时按组分配。做混合更新实验时保持 engine.cpu_offload=false，使分配到 GPU 的组有 GPU 参数分片可用。Muon 需要适用的二维参数；当前包装层会拒绝对普通 Muon 传入分片矩阵。配置适配器不会自动为向量和矩阵选择两套算法。[混合优化器配置](../configs/optimizer/hybrid.yaml) · [实际更新逻辑](../kirl/engine/optim.py)

接入训练时，`resolved = resolve_train_config(cfg)` 的 `resolved.training_args` 用于构造 `TrainingArguments(**resolved.training_args)`。将这个对象作为 Trainer 的 `args`，并将 `resolved.optimizer_cls_and_kwargs` 作为 Trainer 的同名参数传入。只传 training_args 会漏掉 hybrid 的优化器选择。[适配器](../kirl/config/adapters.py) · [GRPOTrainer](../kirl/algorithms/grpo.py)

## Rollout、观测与输出

- **推理资源**：rollout.gpu_memory_utilization 默认为 0.4，tensor_parallel_size 为 1。max_model_len、max_num_seqs、max_num_batched_tokens 默认为 null。显式设置 max_model_len 时必须大于 generation.max_new_tokens，给 prompt 留出空间。enable_sleep_mode、enforce_eager、trust_remote_code 默认关闭。[vLLM 默认配置](../configs/rollout/vllm.yaml)
- **日志和评估**：log_interval、eval_interval、save_interval 分别默认 10、100、500 个更新步，适配器将它们映射为 logging_steps、eval_steps、save_steps。eval_examples 默认为 500，留给评估流程消费。设置 eval_steps 本身不会启动评估。[适配器](../kirl/config/adapters.py)
- **输出路径**：output_dir 默认 ./outputs。Hydra 在其下创建按日期和时间区分的运行目录，示例将其作为本次 output_dir 并写入 resolved_config.yaml。best_checkpoint_dir 默认相对此运行目录的 checkpoints/best。resume_from_checkpoint 默认 null，相对恢复路径从启动目录解析。[训练配置](../configs/train.yaml)
- **随机性与分析**：seed 默认 42，deterministic 默认 true。profiler.global.mode 默认 off，global.ranks、sample_every_steps、trace_iterations 控制采集范围，global.output_dir 默认是本次运行目录下的 profile。profiler.torch 控制活动类型和内存跟踪，profiler.vllm 控制推理侧跟踪。[Profiler 字段](../configs/profiler/default.yaml) · [KiRL Profiler 说明](kirl_profiler.md)

## 给项目增加配置

新增已有组的预设时，在 configs 对应子目录增加 YAML，然后通过命令行选择，或修改 train.yaml 的 defaults 设为默认。新增结构化字段时，还要在对应 schema 声明类型和默认值，并在适配器或实际使用方消费它。对于 engine、algo、dataset 等有名称白名单的组件，还要更新 registry.yaml。优化器算法的允许值和类映射目前维护在 adapters.py。验证时先用 `--cfg job --resolve` 核对组合结果，再运行去掉该选项的入口检查 KiRL 的约束和路径解析。[Schema](../kirl/config/schema.py) · [名称注册表](../kirl/config/registry.yaml) · [适配器](../kirl/config/adapters.py)
