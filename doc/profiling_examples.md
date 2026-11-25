# Profiling 使用示例

本文给出在训练脚本、AutoML、分布式任务以及 MCP/Nsight 工作流中调用统一 profiling 包的代码模板。包级 API 和每个参数的详细说明见 [`code/eneuro/profiling/README.md`](../code/eneuro/profiling/README.md)。

## Profiling 自检脚本

仓库根目录可直接运行 `code/profile_profiling.py`。默认使用合成数据和 CPU，仅需 NumPy；`--device cuda` 会要求 CuPy/CUDA 可用，并真实执行短矩阵训练负载，采集整卡显存、CUDA Event、系统/GPU 样本、CuPy 内存池和 Nsight CLI 信息：

```powershell
# CPU 快速检查
python code/profile_profiling.py

# CUDA 检查，附加 NVTX 标记
python code/profile_profiling.py --device cuda --nvtx

# CUDA 上使用自己的 CSV 特征，CSV 第一行是字段名，label 列自动剔除
python code/profile_profiling.py --device cuda --dataset csv --data data/features.csv --label-column label

# CUDA 上读取 NPZ 的 X 数组，并查询 PyTorch allocator（可选）
python code/profile_profiling.py --device cuda --dataset npz --data data/features.npz --x-key X --allocator-framework torch
```

支持 `synthetic`、`csv`、`npz` 三种数据源。输入特征会展平为二维数值矩阵；CSV 必须有表头，NPZ 特征键默认为 `X`。脚本对当前数据构造一个小型回归目标，不要求固定数据集、模型权重或下载 MNIST。`--samples` 限制载入行数，`--iterations` 控制负载时长，`--device-id` 选择 GPU，`--sample-interval` 调整遥测采样间隔，`--output-dir` 设置 JSON/JSONL 报告位置。传入 `--no-system-metrics` 可跳过后台采样，保留直接快照和阶段计时。

运行完成时会打印报告路径、阶段耗时、系统快照、CUDA 显存/allocator 结果、NVTX backend 和 Nsight 命令计划。退出码 `0` 表示所选检查通过；CUDA 初始化或 CUDA Event 失败返回非零；显式传入 `--nvtx` 但无 NVTX provider 时也会返回非零。Nsight 命令只检查发现并生成参数，不会自动启动长时间的 Nsight 捕获。Windows CUDA 环境如遇到 CUDA 工具链版本兼容问题，可按项目 `AGENTS.md` 临时切换 CUDA 12.6。

## 1. 安装和公共配置

在仓库根目录安装可选依赖，并让 Python 找到 `code/eneuro`：

```powershell
python -m pip install -r requirements-profiling.txt
$env:PYTHONPATH = (Resolve-Path code).Path
```

建立可在 Trainer、AutoML 或分布式入口之间复用的配置：

```python
from eneuro.profiling import ProfileConfig

PROFILE = ProfileConfig(
    enabled=True,
    output_dir="artifacts/profiles",
    sample_interval=0.5,
    system_metrics=True,
    gpu=True,
    cuda_timing=True,
    nvtx=True,
    batch_every=10,
)
```

普通训练默认关闭采集。调用方可用 `profiling=True` 开启默认配置，用 `ProfileConfig` 或配置字典传入选项，用 `profiling=False` 显式禁用。嵌套调用省略 `profiling` 时会继承当前 `ProfileSession`。这适合 AutoML 在总搜索任务上统一开关，并允许单个 trial 显式设为 `False`。

## 2. EnNeuro Trainer

`Trainer.fit` 会采集训练和验证阶段、样本计数及吞吐量。按 `batch_every` 间隔插入 data wait、transfer、forward、loss、backward 和 optimizer 阶段；设为 `0` 关闭批次阶段插桩。

```python
from eneuro.profiling import ProfileConfig
from eneuro.train import Trainer

prof = ProfileConfig(
    output_dir="artifacts/profiles",
    sample_interval=0.25,
    cuda_timing=True,
    nvtx=True,
    batch_every=5,
)
trainer = Trainer(model, loss_fn, optimizer)
trainer.fit(
    train_loader,
    val_loader,
    epochs=3,
    device="cuda",
    profiling=prof,
)

print("report:", trainer.profile_path)
print("throughput:", trainer.profile_report["metrics"])
print("phase timings:", trainer.profile_report["timings"])
```

如果没有 `output_dir`，可直接读取 `trainer.profile_report`，不会落盘。若要把多个函数的阶段归到同一份运行报告，建立外层 `ProfileSession`：

```python
from eneuro.profiling import ProfileConfig, ProfileSession

with ProfileSession(PROFILE, name="experiment", metadata={"model": "ResNet18", "seed": 1}) as run:
    trainer.fit(train_loader, val_loader, epochs=3, device="cuda")
    run.mark("checkpoint_saved", path="artifacts/model.pt")

print(run.report_path)
```

内层 `fit` 未传 `profiling` 时沿用外层 session，并将 `fit` 作为阶段计时；要单独关闭这一层可传 `profiling=False`。

## 3. 自定义训练循环

非 EnNeuro Trainer 的训练代码可以直接使用 `ProfileSession`。记时和资源采样解耦，跨度名可按模型或 pipeline 定义：

```python
from eneuro.profiling import ProfileConfig, ProfileSession

config = ProfileConfig(
    output_dir="artifacts/profiles",
    sample_interval=0.5,
    cuda_timing=True,
    nvtx=True,
)

with ProfileSession(config, name="pytorch-custom", metadata={"model": "MyModel"}) as profile:
    for epoch in range(epochs):
        with profile.span("epoch"):
            for images, labels in loader:
                with profile.span("data_to_device"):
                    images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with profile.span("forward", cuda=True):
                    logits = model(images)
                with profile.span("backward", cuda=True):
                    loss = criterion(logits, labels)
                    loss.backward()
                with profile.span("optimizer", cuda=True):
                    optimizer.step()
                profile.count("examples", len(images))
                profile.sample_cupy_pool()  # optional; only samples CuPy's default pool
            profile.mark("epoch_end", epoch=epoch + 1, loss=float(loss.detach().cpu()))

print(profile.report_path)
```

若使用 PyTorch/Paddle/TensorFlow allocator 指标，可在需要的位置调用 `cuda_allocator_memory("torch")` 等函数，再用 `profile.observe("framework_allocator", result)` 写入报告。该值代表框架分配器统计，不代表整卡已用显存。

## 4. AutoML 搜索和评估

### `create_search`

同一份配置覆盖搜索过程、trial 执行和最终重训。搜索结果可序列化 profiling 摘要：

```python
from eneuro.agent import create_search
from eneuro.profiling import ProfileConfig

config = ProfileConfig(
    output_dir="artifacts/automl/profiles",
    sample_interval=0.5,
    cuda_timing=True,
    nvtx=True,
)
search = create_search(
    "random",
    executor,
    n_trials=20,
    seed=42,
    profiling=config,
)
result = search.run()
print(result.to_dict().get("profiling"))
```

`profiling=None` 会继承已有 session，因此可以用一层配置包裹多个搜索阶段；`profiling=False` 在某个搜索入口禁用 profiling。AutoML profile 把候选配置、试验窗口和 refit 信息关联进报告。

### 固定预算评估和计时搜索

`run_evaluation(..., profiling=...)` 和 `run_timed_search(..., profiling=...)` 采用相同的开关与 `ProfileConfig` 格式。评估函数会在其 `output` 下生成 `profiling/` 子目录；计时搜索在输出目录下生成同名子目录。

```python
from eneuro.agent.evaluation import run_evaluation

result = run_evaluation(
    executor_factory,
    search_space,
    baseline_config=baseline,
    n_trials=12,
    seed=42,
    runs=3,
    output="artifacts/evaluation",
    protocol=protocol,
    profiling=config,
)
print(result.to_dict().get("profiling"))
```

`run_timed_search` 除必需的 `profiling=config` 外还需要搜索空间、预算、输出路径、协议和报告格式化回调。项目内 AutoML 命令行脚本可统一接受 `--profile`、`--profile-interval`、`--profile-nvtx`、`--profile-cuda-timing`，例如：

```powershell
python demos/automl_mnist_cuda.py --help
python demos/automl_mnist_cuda.py --profile --profile-interval 0.5 --profile-nvtx --profile-cuda-timing
```

`demos/automl_mnist_agent.py`、`automl_mnist_cuda.py`、`automl_car_agent.py`、`automl_car_cuda.py` 和 `automl_donkeycar_cuda.py` 都接入这些开关。不同脚本的训练参数不同，请先用 `--help` 查看对应选项。关闭时省略 `--profile`。

## 5. 分布式训练

分布式 launcher 使用同一配置，除 system/GPU samples 外会把已有 worker/PS timing JSONL 汇总到 profile 报告：

```python
from eneuro.distributed.launcher import launch
from eneuro.profiling import ProfileConfig

launch(
    model_factory,
    dataset_factory,
    loss_fn,
    optimizer_factory,
    distributed_config,
    profiling=ProfileConfig(
        output_dir="artifacts/distributed/profiles",
        sample_interval=1.0,
        gpu=True,
    ),
)
```

launcher 的 CPU/RSS 指标是 launcher 进程本身的采样；主机和 GPU 指标覆盖当前节点。已有分布式明细可使用 `eneuro.profiling.distributed` 中的 `summarize_timings`、`benchmark_metrics`、`worker_loss_trace` 做二次分析。

## 6. 直接读取当前系统和 GPU

用于服务、基准测试或自动调参决策时，可以不启动完整 profile session：

```python
from eneuro.profiling import system_snapshot, process_snapshot, cuda_device_memory
from eneuro.profiling.cuda import cuda_allocator_memory

snapshot = system_snapshot(gpu=True)
process = process_snapshot()
device = cuda_device_memory()
allocator = cuda_allocator_memory("torch")
```

`system_snapshot` 给一次性主机、当前进程和 GPU 快照；`process_snapshot` 只采当前进程 CPU/RSS；`cuda_device_memory` 使用 CuPy 读取当前逻辑设备的整卡显存；`cuda_allocator_memory` 读取指定框架 allocator 的分配、保留和峰值字节数。驱动或框架未提供的值可能为 `None`。

## 7. MCP 客户端读取训练数据

### 配置 server

MCP server 与训练脚本通过 profile 文件目录交换数据。先安装 profiling 可选依赖，再将如下片段加入 MCP 客户端配置，按本机环境替换路径：

```json
{
  "mcpServers": {
    "eneuro-profiling": {
      "command": "C:/repo/.venv/Scripts/python.exe",
      "args": ["-m", "eneuro.profiling.mcp_server", "--profile-dir", "C:/repo/artifacts/profiles"],
      "env": {"PYTHONPATH": "C:/repo/code"}
    }
  }
}
```

训练 `ProfileConfig.output_dir` 与 `--profile-dir` 指向同一目录。server 是 stdio 传输，只开放七项只读工具：

| Tool | 用途 |
| --- | --- |
| `get_system_snapshot(gpu=True)` | 查看 server 所在机器当前资源。进程数据属于 MCP server 进程。 |
| `list_profiles(limit=20)` | 浏览最近训练记录。 |
| `get_profile(run_id)` | 读取报告全文。 |
| `get_profile_samples(run_id, limit=100)` | 查看最新时间序列样本。 |
| `analyze_profile(run_id)` | 查看 Trainer 阶段耗时排名、峰值、吞吐量。 |
| `get_nsight_tools()` | 查找 `nsys` 和 `ncu`。 |
| `plan_nsight(tool, ...)` | 生成 Nsight 命令参数数组，不会替客户端运行。 |

典型调用顺序是 `list_profiles` 取得 `run_id`，再调用 `analyze_profile`、`get_profile` 或 `get_profile_samples`。

## 8. Nsight Systems 与 Nsight Compute

### 捕获完整训练进程

```python
import subprocess
import sys
from eneuro.profiling.nsight import nsight_command

argv = nsight_command(
    "nsys",
    [sys.executable, "train.py", "--profile", "--profile-nvtx"],
    output="artifacts/nsight/train",
    cpu_sampling=True,
)
subprocess.run(argv, check=True)
```

`nsys` 会跟踪 CUDA 和 NVTX。`ProfileConfig(nvtx=True)` 或训练脚本的 `--profile-nvtx` 会给 EnNeuro 阶段添加范围标记。分析已生成的 report：

```python
subprocess.run(
    nsight_command("nsys_stats", report="artifacts/nsight/train.nsys-rep"),
    check=True,
)
```

### 只捕获选定 CUDA 范围

```python
from eneuro.profiling.nsight import cuda_profiler_capture

with cuda_profiler_capture():
    train_one_representative_step()
```

启动 `nsys` 时使用 `nsight_command("nsys", application, capture_cuda=True)`。此 capture helper 使用 CuPy 的 CUDA profiler API，并默认在范围两端同步设备。选择正确 CUDA device，并避免同一进程内捕获范围重叠。

### 分析特定 kernel

```python
argv = nsight_command(
    "ncu",
    [sys.executable, "train.py", "--profile", "--profile-nvtx"],
    output="artifacts/nsight/forward.ncu-rep",
    nvtx_include="eneuro@forward/",
    launch_skip=10,
    launch_count=1,
)
subprocess.run(argv, check=True)
```

`ncu` 可能要求管理员配置 NVIDIA GPU performance counter 权限。系统指标采集、NVTX timeline 和硬件 counter 权限是彼此独立的。Nsight 命令函数只构造参数，是否执行由调用者决定。
