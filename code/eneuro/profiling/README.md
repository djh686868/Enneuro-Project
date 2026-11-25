# EnNeuro Profiling

EnNeuro profiling consolidates optional system telemetry, training phase timing, CUDA allocator measurements, distributed log analysis, NVIDIA Nsight helpers, and read-only MCP access. The implementation lives in this package. Collection is opt-in for training entry points and does not require an MCP server.

For end-to-end examples, see [Profiling usage examples](../../../doc/profiling_examples.md). The existing [training profiling guide](../../../doc/training_profiling.md) covers report fields and Nsight interpretation.

## Install and import

From the repository root, install the optional dependencies and add `code` to Python's import path:

```powershell
python -m pip install -r requirements-profiling.txt
$env:PYTHONPATH = (Resolve-Path code).Path
```

The collector can run without optional packages, though each missing provider reduces available metrics. `psutil` enables detailed process and host metrics, NVML (`nvidia-ml-py`) enables NVIDIA device metrics, and `nvidia-smi` is used as a fallback when available. CuPy is needed for CUDA device memory and CUDA-event timing. NVTX support uses the `nvtx` package or CuPy's NVTX interface. MCP support additionally requires the MCP Python SDK.

## One configuration across training entry points

Create one reusable config and pass it to `Trainer.fit`, AutoML search/evaluation, or distributed launch:

```python
from eneuro.profiling import ProfileConfig

profile_config = ProfileConfig(
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

`ProfileConfig` fields:

| Field | Default | Purpose |
| --- | --- | --- |
| `enabled` | `True` | Enable this session. Set `False` for a no-op session. |
| `output_dir` | `None` | Write `<run_id>.json` and, when system sampling is on, `<run_id>.samples.jsonl`. `None` keeps the report in memory. |
| `sample_interval` | `1.0` | Seconds between host/process/GPU snapshots. Must be finite and positive. |
| `system_metrics` | `True` | Run background system sampling and write sample JSONL. |
| `gpu` | `True` | Include GPU collection in system snapshots. Can be `False` while retaining CPU/RAM samples. |
| `cuda_timing` | `False` | Record CUDA event elapsed time for spans explicitly opened with `cuda=True`. |
| `nvtx` | `False` | Annotate spans with NVTX if an NVTX provider is installed. |
| `batch_every` | `10` | Trainer stage instrumentation interval in batches. `0` disables per-batch spans. |
| `max_marks` | `256` | Maximum recent custom marks retained in the report. |
| `max_cuda_events` | `1024` | Bound outstanding CUDA timing events. |

Training integrations accept these forms:

| Argument | Behavior |
| --- | --- |
| omitted or `None` | Inherit the enabled `current_profile()` session, if there is one; otherwise no profiling. |
| `True` | Enable with default `ProfileConfig` settings. |
| `False` | Explicitly disable profiling for this call, even inside an outer session. |
| `ProfileConfig(...)` | Use the supplied options. |
| `dict` | Equivalent to `ProfileConfig(**dict)`. |

This makes the same AutoML/model configuration portable: put the profiling config beside the model/trial config, then enable or disable it with one `profiling` field. `None` keeps normal training runs free of collection threads and reports.

## Public Python APIs

### System and process collectors

```python
from eneuro.profiling import (
    SystemCollector,
    system_snapshot,
    process_snapshot,
    cuda_device_memory,
)

now = system_snapshot(gpu=True)       # one host/process/GPU sample
process = process_snapshot()          # this process only, immediate
device_memory = cuda_device_memory()  # current CuPy logical device, or None

collector = SystemCollector(gpu=True)
try:
    sample = collector.sample()
    process_sample = collector.sample_process()
finally:
    collector.close()
```

`system_snapshot()` performs a short CPU baseline before it returns. `process_snapshot()` avoids host/GPU collection and is useful inside existing benchmark loops. `SystemCollector.sample()` returns normalized `host`, `process`, `gpus`, and optional `errors` sections. Not all OS/driver combinations expose every field; unavailable values are omitted or `None`. Under Windows WDDM, per-process GPU memory can be unavailable even when whole-device memory is reported.

### Framework allocator and device metadata

```python
from eneuro.profiling.cuda import cuda_allocator_memory, cuda_device_name

print(cuda_device_name(0))
torch_bytes = cuda_allocator_memory("torch", device=0)
paddle_bytes = cuda_allocator_memory("paddle", device=0)
tensorflow_bytes = cuda_allocator_memory("tensorflow", device=0)
```

Allocator results use bytes and normalized keys `allocated_bytes`, `reserved_bytes`, and `peak_allocated_bytes`. Unsupported framework/device combinations return `None`; the framework's own allocator scope is not the same as whole-device memory. `cuda_device_memory()` reports current device free/total/used bytes through CuPy.

### ProfileSession and custom timing

```python
from eneuro.profiling import ProfileConfig, ProfileSession, current_profile

with ProfileSession(
    ProfileConfig(output_dir="artifacts/profiles", sample_interval=0.25),
    name="custom-run",
    metadata={"model": "MyModel", "seed": 7},
) as profile:
    with profile.span("data_and_forward", cuda=True):
        output = model(batch)
    profile.count("examples", len(batch))
    profile.mark("epoch_end", epoch=1)
    profile.observe("my_allocator", {"live_bytes": 12345})
    profile.sample_cupy_pool()

report = profile.report
print(profile.report_path, profile.samples_path)
```

`ProfileSession` is a single-use context manager. `span(name, cuda=False)` records wall-clock and current-thread CPU time. With `cuda=True` and `cuda_timing=True`, it also records CUDA event time on the current stream without synchronizing each span. `count()` accumulates numeric counters, `mark()` adds timestamped metadata, and `observe()` keeps the latest application-provided metrics under a name. The optional CuPy pool sampler records default-pool used/reserved bytes at the time it is called. `current_profile()` returns the active session in the current context, or `None`.

Reports are JSON with aggregate peaks, averages, timing statistics, counters, marks, observations, provider errors, and config. Samples are JSONL with Unix timestamps and run-relative `offset_ns`. Sampled peaks can miss brief spikes. CUDA event time is stream work duration and may not equal wall time. A missing metric is not equivalent to zero.

### CUDA event timer

```python
from eneuro.profiling.timing import CudaEventTimer

timer = CudaEventTimer()  # records start on the current CuPy stream
run_cuda_work()
timer.stop()
elapsed_ms = timer.elapsed_ms(wait=True)
```

Use this low-level helper when the caller needs explicit event control. The timer synchronizes only when `elapsed_ms(wait=True)` requests the result. For normal training instrumentation, prefer `ProfileSession.span(..., cuda=True)`.

### Nsight API

```python
from eneuro.profiling.nsight import NVTX, cuda_profiler_capture, discover_nsight, nsight_command

tools = discover_nsight()  # {"nsys": path-or-None, "ncu": path-or-None}
argv = nsight_command("nsys", ["python", "train.py"], output="artifacts/trace")
```

`NVTX(enabled=True).range("forward")` is a context manager; it uses `nvtx` or CuPy when installed and is a no-op when no provider exists. Training sessions set NVTX on spans through `ProfileConfig(nvtx=True)`. `cuda_profiler_capture()` can bracket an explicit CUDA capture region for Nsight Systems. `nsight_command()` returns an argument list for `nsys`, `nsys_stats`, or `ncu`; it does not execute the command. Use `subprocess.run(argv, check=True)` only when you want to launch the profiler. `ncu` supports `nvtx_include`, `launch_skip`, and `launch_count`; Nsight Systems supports `capture_cuda` and `cpu_sampling`.

### Distributed logs and Linux perf

```python
from eneuro.profiling.distributed import (
    summarize_timings,
    benchmark_metrics,
    worker_loss_trace,
)
from eneuro.profiling.perf import perf_stat

roles = summarize_timings("artifacts/run", backend="ps", world_size=2)
metrics = benchmark_metrics("artifacts/run", world_size=2)
loss_trace = worker_loss_trace("artifacts/run", world_size=2)
counters, stdout = perf_stat(["python", "bench.py"], ["cycles", "instructions"])
```

Distributed helpers read the existing `worker-<rank>.jsonl` and, for the PS backend, `ps.jsonl` format. `benchmark_metrics()` expects completed step records. `perf_stat()` requires Linux `perf` and event permissions; it raises `FileNotFoundError` or `RuntimeError` when unavailable.

## MCP server

Install the optional MCP dependency, then run the stdio server with the directory that contains profile reports:

```powershell
python -m eneuro.profiling.mcp_server --profile-dir artifacts/profiles
```

Example MCP client configuration (use absolute paths for your machine):

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

The server exposes seven read-only tools:

| MCP tool | Inputs | Returns |
| --- | --- | --- |
| `get_system_snapshot` | `gpu: bool = true` | Current host and MCP-server process metrics, plus visible GPUs. |
| `list_profiles` | `limit: int = 20` (1–100) | Recent profile summaries. |
| `get_profile` | `run_id: str` | Full JSON report. |
| `get_profile_samples` | `run_id: str`, `limit: int = 100` (1–1000) | Latest timestamped system samples. |
| `analyze_profile` | `run_id: str` | Trainer phase rankings, resource peaks/averages, throughput, errors. |
| `get_nsight_tools` | none | Discovered `nsys` and `ncu` executable paths. |
| `plan_nsight` | `tool`, optional application/output/report/filter fields | `{"argv": [...]}` command plan, never executes it. |

Set `ProfileConfig.output_dir` and server `--profile-dir` to the same directory. MCP access is scoped to that configured directory and run IDs must be 32 lowercase hexadecimal characters.

## Files in this package

| Module | Main API | Responsibility |
| --- | --- | --- |
| `collectors.py` | `SystemCollector`, `system_snapshot`, `process_snapshot`, `cuda_device_memory` | Host, process, and device telemetry. |
| `cuda.py` | `cuda_device_name`, `cuda_allocator_memory` | CUDA metadata and framework allocator counters. |
| `session.py` | `ProfileConfig`, `ProfileSession`, `current_profile` | Opt-in session, samples, spans, reports. |
| `timing.py` | `CudaEventTimer` | Low-level current-stream CUDA event timing. |
| `nsight.py` | `NVTX`, `cuda_profiler_capture`, `discover_nsight`, `nsight_command` | NVTX scopes, Nsight discovery, command construction. |
| `distributed.py` | `summarize_timings`, `benchmark_metrics`, `worker_loss_trace` | Existing distributed JSONL analysis. |
| `perf.py` | `perf_stat` | Linux hardware-counter wrapper. |
| `mcp_server.py` | `create_server`, CLI `main` | Read-only MCP tools over local reports. |

