# KiRL Profiling

## 结构与职责

```text
ProfilerSettings
├── global: ProfilerConfig       开关、rank、采样频率、trace 窗口、输出目录
├── torch: TorchProfilerConfig   CPU/CUDA activity、shape、stack、memory
└── vllm: VLLMProfilerConfig     vLLM 引擎侧 trace 配置
                 ↓
create_probe(...) → Probe / NullProbe
├── step(update_step)            设置 optimizer update 编号
├── iteration(id)                管理一次迭代及其 trace 采集窗口
├── stage(name)                  记录阶段 host wall time 与嵌套关系
├── metric(...)                 写入 CPU 标量或有界结构化指标
├── collect_backend_metrics()   显式读取后端指标
└── close()                     排空输出队列并写入最终状态
        ├── JsonlSink → records.jsonl / status.json
        └── adapters
            ├── TorchProfilerAdapter → Chrome trace JSON
            └── VLLMProfilerAdapter → vLLM worker trace 目录引用

build_report(run_directory) → RunReport → export(report_path)
```

源文件位于 `kirl/profiler/`。配置示例为 `configs/profiler/default.yaml`。
`Probe` 不持有训练模型，不执行 GPU/rank 同步；创建者负责关闭它。
同一 worker 的阶段、指标与采集窗口应共享其 Probe。不同 worker 各自创建
Probe，并通过相同 `run_id` 关联。

| mode | 行为 |
| --- | --- |
| `off` | 返回 `NullProbe`，不生成 profiling 文件 |
| `basic` | 记录调用方提供的阶段和指标 |
| `sampled` | 仅在 `update_step % sample_every_steps == 0` 时记录 |
| `trace` | 记录阶段和指标，在指定 `trace_iterations` 中启动已挂载的 adapter |

`step()` 只设置编号，不会启动 trace；trace 由 `iteration()` 控制。
未选中的 rank 返回 `NullProbe`。默认配置为 `mode: off`、仅选 rank 0。

## 输出形式与目录

通过当前 Hydra 入口解析配置时，`global.output_dir: profile` 相对于本次
Hydra run 输出目录解析。以项目根目录启动、使用默认输出配置为例：

```text
outputs/<日期>/<时间>/
├── resolved_config.yaml
└── profile/
    └── <run_id>/
        ├── <role>-r0-p<pid>-<随机ID>/
        │   ├── records.jsonl
        │   ├── status.json
        │   └── iteration-<id>-capture-<次数>/
        │       └── torch/
        │           └── trace.json
        ├── <role>-r1-p<pid>-<随机ID>/
        │   ├── records.jsonl
        │   └── status.json
        └── profile_report.json          手动生成的每 run 汇总
```

直接使用 `ProfilerConfig()` 时，其默认输出根目录为 `./outputs/profile`；
最终目录始终是 `<config.output_dir>/<run_id>/<session_id>/`。
可从 `probe.session_dir` 和 `probe.record_path` 获取实际路径。

| 文件 | 内容与生成时机 |
| --- | --- |
| `records.jsonl` | 每行一条 `stage`、`metric` 或 `artifact` 记录；后台线程缓冲写入，默认每 64 条或 5 秒刷新 |
| `status.json` | `close()` 时保存配置、adapter、运行时长、写入/丢弃/待写数量和错误 |
| `torch/trace.json` | 选定窗口结束时导出；包含已启用 activity 的事件，可用 Perfetto 等工具查看 |
| `profile_report.json` | 调用 `build_report()` 后显式 `export()` 生成，不会自动产生 |

JSONL 的公共字段包含 `run_id`、`session_id`、`role`、`rank`、`host`、
`pid`、`iteration_id`、`update_step`、`policy_version`、`capture_id`、
`timestamp_ns` 和 `data`。阶段数据包含名称、父子 span、耗时及失败标记；
指标数据包含名称、值、单位、类型、backend 和 labels。

vLLM 的实际 trace 由 worker 写入 `vllm.torch_profiler_dir`，不会自动跟随
上述 Torch trace 目录。该目录应对 worker 可访问；JSONL 中仅保存目录引用，
状态为 `pending`，不代表文件已经刷新或下载到本机。

## 每次 run 构建报告

在所有 worker 退出采集窗口并关闭各自 Probe 后，由一个汇总进程执行。
分布式运行时先收集各 worker 的完整 session 目录（含 JSONL 和状态文件），
放到同一 run 目录下。仅关闭 rank 0 的 Probe 不足以保证其他 rank 写入完成。



`build_report()` 递归查找目录内的 `records.jsonl`，也接受单个 JSONL 路径或
路径列表。每 run 汇总应传入对应的 `<run_id>` 目录，避免混入其他实验。
对已经关闭的同一批记录重复执行，会覆盖指定报告文件，不修改原始记录。

调用方式:

```python
from pathlib import Path
from kirl.profiler import build_report

# 所有 worker 已关闭 Probe，日志也已收集到 run_dir。
run_dir = Path(profile_output_dir) / run_id
report = build_report(run_dir)
report.export(run_dir / "profile_report.json")
```

报告字段如下：

| 字段 | 含义 |
| --- | --- |
| `runs` | 输入记录中的 run ID |
| `stages` | 按 session、阶段、计时类型和是否处于 trace 窗口分组的次数、总时长、最小/最大/平均时长、失败次数 |
| `metrics` | 按 session、backend、名称、单位、类型、labels、trace 窗口分组的观测次数和最后读取的指标值 |
| `artifacts` | trace 文件或目录引用及状态 |
| `sessions` | 各 session 的最终 `status.json` |
| `warnings` | 缺少最终状态、丢记录、未排空、损坏记录等提示 |

指标不会自动取平均或累加 counter；多个文件的遍历顺序也不等同于全局时间顺序。
阶段计时为无设备同步的 host wall time，嵌套阶段及各 rank 时长不能直接相加
作为端到端耗时。报告始终包含这条计时解释，不表示采集本身失败。

## 当前接入状态

上述为现有 profiling API 的输出行为。训练代码目前已有 `train.microstep`
和 vLLM `rollout` 阶段埋点，但尚未贯通 `step()`、`iteration()`、后端指标
轮询及每 run 自动报告。`examples/train_grpo_gsm8k.py` 当前只解析并保存配置，
不执行训练。因此仅修改 YAML 不会自动产生完整 profiling 结果。
