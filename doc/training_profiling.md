# 训练性能采集

在已有 EnNeuro 环境中，可一次安装系统采集、NVTX 和 MCP 的可选依赖。MCP 入口使用 Python 3.10 或更新版本。

```powershell
python -m pip install -r requirements-profiling.txt
$env:PYTHONPATH = (Resolve-Path code).Path
```

新增的完整接口参考见[Profiling README](../code/eneuro/profiling/README.md)，按场景组织的训练脚本和 AutoML 代码见[使用示例](profiling_examples.md)。

## Trainer 与自定义训练

```python
from eneuro.profiling import ProfileConfig, ProfileSession

config = ProfileConfig(
    output_dir="artifacts/profiles",
    sample_interval=0.5,
    batch_every=10,
    nvtx=True,
    cuda_timing=True,
)
trainer.fit(train_loader, val_loader, device="cuda", profiling=config)
print(trainer.profile_path)
print(trainer.profile_report["metrics"])
```

`profiling` 接受 `True`、`False`、`ProfileConfig` 或配置字典。省略时继承当前进程中外层的 `ProfileSession`，普通调用默认关闭。显式传入 `False` 可关闭本次调用的插桩。去掉 `output_dir` 时，可从 Python 读取内存报告。

Trainer 自动统计训练、验证、样本数和样本吞吐。阶段计时每隔 `batch_every` 个批次记录一次，从各轮的首批开始，覆盖 DataLoader 等待、输入转移、前向、损失、反向和优化器更新。`batch_every=0` 可关闭批次插桩。`system_metrics=False` 可单独关闭后台系统采样，`gpu=False` 可关闭 GPU 资源采样。

自定义训练循环可用相同配置标记业务阶段并复用已有计数器。

```python
with ProfileSession(config, name="custom-train", metadata={"model": "MyModel"}) as profile:
    with profile.span("forward", cuda=True):
        prediction = model(batch)
    profile.count("examples", len(batch))
    profile.mark("epoch_end", epoch=1)
    # 已使用 EnNeuro DeviceAllocator 时，可记录其分配器统计。
    profile.observe("device_allocator", allocator.stats())
```

`observe()` 保留每个名称的最近一次指标。例如 `DeviceAllocator.stats()` 可提供分配器维护的 `peak_live_bytes`、池命中率和传输字节。`system_snapshot()` 适用于独立读取当前资源，Web 系统状态接口使用同一采集器。

## 源码接口

以下模块均位于 `code/eneuro/profiling/`。

| 模块 | Python 接口 | 功能 |
| --- | --- | --- |
| `collectors.py` | `SystemCollector`、`system_snapshot()`、`process_snapshot()`、`cuda_device_memory()` | 主机及进程 CPU、内存和 I/O，多张物理 GPU 的 NVML 或 `nvidia-smi` 数据，以及当前 CuPy 逻辑设备的整卡显存 |
| `cuda.py` | `cuda_device_name()`、`cuda_allocator_memory()` | CUDA 可见设备名称，以及 PyTorch、Paddle、TensorFlow 各自分配器的已分配、保留和峰值字节数 |
| `session.py`、`timing.py` | `ProfileConfig`、`ProfileSession`、`current_profile()`、`CudaEventTimer` | 后台采样、训练和自定义阶段计时、CUDA 事件、样本吞吐、标记、观测值和 JSON/JSONL 报告 |
| `nsight.py` | `NVTX`、`cuda_profiler_capture()`、`discover_nsight()`、`nsight_command()` | NVTX 标记、定段抓取、Nsight CLI 发现与 `nsys profile`、`nsys stats`、`ncu` 命令参数 |
| `distributed.py` | `summarize_timings()`、`benchmark_metrics()`、`worker_loss_trace()` | 复用 PS/Ring 日志，计算各角色耗时、通信、屏障等待、吞吐和按样本加权的 loss |
| `perf.py` | `perf_stat()` | Linux PMU 硬件计数器，由卷积缓存基准复用 |
| `mcp_server.py` | `create_server()` | 启动 stdio 服务，暴露下文列出的七个工具 |

`process_snapshot()` 适用于只需当前进程 CPU 和内存的脚本，无需等待系统 CPU 基线。Windows / Linux 在可用时提供 `rss_lifetime_peak_bytes`，其时间范围从进程启动起算。运行区间内的采样峰值可读取 `peaks.process.rss_bytes`。旧 `TimeMeter` 的计时用途可迁移到 `ProfileSession.span()`。

`cuda_allocator_memory()` 的用量来自所选框架的分配器。`cuda_device_memory()` 从当前 CUDA 逻辑设备读取整卡用量。跨框架 DonkeyCar 对照脚本复用这些入口，并按原有批次频率采样。

## AutoML 与分布式

```python
from eneuro.agent import create_search

result = create_search("random", executor, n_trials=10, profiling=config).run()
print(result.to_dict()["profiling"])
```

`create_search`、`run_evaluation` 和 `run_timed_search` 共用 `profiling` 参数。报告关联候选配置、试验时间窗和最终重训。`run_evaluation` 和 `run_timed_search` 在未指定 `output_dir` 时，把采集文件写入原输出目录的 `profiling/` 子目录。仓库的 MNIST 和车模 AutoML 脚本支持 `--profile`、`--profile-interval`、`--profile-nvtx` 和 `--profile-cuda-timing`。

`launch_distributed(..., profiling=config)` 汇总 PS 或 Ring 的既有指标日志，包括各 rank 的阶段耗时、通信字节、样本数和 PS 屏障等待。采集进程的 CPU 和 RSS 对应 launcher，整机及整卡指标覆盖所在机器。详见[分布式训练性能采集](../code/eneuro/distributed/README.md)。

## 报告口径

| 字段 | 含义 |
| --- | --- |
| `<run_id>.json` | 原子更新的状态、配置、采样峰值、均值、阶段统计、标记和可用性信息 |
| `<run_id>.samples.jsonl` | 可流式读取的主机、进程和各 GPU 快照。每行含 Unix 纳秒时间戳及运行内的 `offset_ns` |
| `peaks.gpus` | 按 GPU UUID（可用时）聚合的采样最大值。显存字段分别记录整卡和当前进程，`index` 为物理 GPU 编号 |
| `peaks.cupy_default_pool` | 按 CUDA 逻辑设备编号聚合已采样批次的 CuPy 默认池已用和保留显存 |
| `timings.*.wall_ms` / `thread_cpu_ms` / `cuda_ms` | 墙钟、当前线程 CPU、当前 CUDA 流事件计时，含次数、均值、总和和极值 |

系统采样优先使用 psutil 和 NVML。GPU 采集可回退到 `nvidia-smi`。驱动未提供的指标（例如部分 Windows WDDM 环境的进程显存）为 `null`，峰值中没有对应条目。采样间隔内的瞬时峰值可能漏采。极短的运行可能只留下一个快照，该快照中的 CPU 使用率可能缺少时间基线。GPU 利用率是驱动给出的设备活动比例，SM occupancy 和内核硬件计数器可通过 Nsight Compute 采集。

CUDA Event 计时默认关闭，启用后在同一当前流记录事件，延迟到退出时汇总未完成的事件。报告中的墙钟与 CUDA 流时间可用于定位不同阶段，嵌套范围会重叠。DataLoader 预取模式下，`data_wait` 衡量训练线程取数据的等待时间。

在受限的 Windows 运行环境中，可把 `CUPY_CACHE_DIR` 指向当前会话可写的目录。CuPy 或 NVRTC 发生 CUDA 工具链版本错误时，按仓库 `AGENTS.md` 临时切换到 CUDA 12.6。

## MCP 客户端

启动 stdio 服务时，把 `--profile-dir` 指向训练使用的 `output_dir`。以下配置可改成客户端的 MCP 配置格式并填入绝对路径。

```json
{
  "mcpServers": {
    "eneuro-profiling": {
      "command": "C:/path/to/python.exe",
      "args": ["-m", "eneuro.profiling.mcp_server", "--profile-dir", "C:/repo/artifacts/profiles"],
      "env": {"PYTHONPATH": "C:/repo/code"}
    }
  }
}
```

- `get_system_snapshot` 读取本机资源。进程字段对应 MCP 服务进程。
- `list_profiles`、`get_profile`、`get_profile_samples` 读取训练运行和时间序列。
- `analyze_profile` 按平均墙钟和 CUDA 流耗时对 Trainer 阶段排序，并返回资源和吞吐数据。
- `get_nsight_tools` 发现本机 Nsight CLI，`plan_nsight` 生成 argv。训练代码负责设置采集配置。

SDK、工具声明和传输文档见 [MCP Python SDK](https://py.sdk.modelcontextprotocol.io/v1/)。

## Nsight Systems 与 Compute

给训练脚本开启 `nvtx=True` 后，可用以下方式生成和运行 CLI 参数。

```python
import subprocess
import sys
from eneuro.profiling.nsight import nsight_command

argv = nsight_command("nsys", [sys.executable, "train_script.py"], output="train_trace")
subprocess.run(argv, check=True)
subprocess.run(nsight_command("nsys_stats", report="train_trace.nsys-rep"), check=True)
```

`nsys` 默认跟踪 CUDA 和 NVTX。只抓取指定区段时，在训练代码中用 `cuda_profiler_capture()` 包住区段，并给 `nsight_command("nsys", ...)` 传入 `capture_cuda=True`。分析指定内核时使用 `ncu`，可配 `nvtx_include="eneuro@forward/"`（独立 NVTX）或 `"forward/"`（CuPy NVTX），并用 `launch_skip`、`launch_count` 控制抓取范围。具体权限与参数见 [Nsight Systems 指南](https://docs.nvidia.com/nsight-systems/UserGuide/) 和 [Nsight Compute CLI 指南](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)。
