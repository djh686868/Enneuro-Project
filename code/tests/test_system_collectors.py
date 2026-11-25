import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from eneuro.profiling import collectors


def test_smi_fallback_preserves_physical_devices_and_missing_fields(monkeypatch):
    def import_module(name):
        raise ImportError(name)

    monkeypatch.setattr(collectors, "importlib", SimpleNamespace(import_module=import_module))
    monkeypatch.setattr(collectors.shutil, "which", lambda name: name)
    calls = []

    def run(command, **options):
        calls.append(command)
        assert options["check"] and options["timeout"] > 0
        if command[1].startswith("--query-gpu="):
            output = (
                '1, GPU-first, "GPU, first", 20, 8, 2048, 512, 1536, 53, [N/A], 555.12\n'
                '3, GPU-other, GPU other, [N/A], [N/A], 4096, 1024, 3072, [N/A], 75.50, 555.12\n'
            )
        else:
            pid = os.getpid()
            output = f"GPU-first, {pid}, 48\nGPU-first, {pid}, 64\nGPU-other, {pid}, [N/A]\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(collectors, "subprocess", SimpleNamespace(run=run))
    collector = collectors.SystemCollector()
    try:
        snapshot = collector.sample()
    finally:
        collector.close()

    first, other = snapshot["gpus"]
    assert snapshot["sources"]["gpu"] == "nvidia-smi"
    assert snapshot["process"]["pid"] == os.getpid()
    assert first["index"] == 1 and first["name"] == "GPU, first"
    assert first["memory_used_bytes"] == 512 * 1024 ** 2
    assert first["memory_percent"] == 25
    assert first["power_w"] is None
    assert first["process_memory_used_bytes"] == 64 * 1024 ** 2
    assert other["index"] == 3 and other["power_w"] == 75.5
    assert other["utilization_percent"] is None and other["process_memory_used_bytes"] is None
    assert len(calls) == 2
    json.dumps(snapshot, allow_nan=False)


def test_nvml_unsupported_fields_and_collector_shutdown(monkeypatch):
    updates = []
    pid = os.getpid()
    api = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        nvmlInit=lambda: updates.append("init"),
        nvmlShutdown=lambda: updates.append("shutdown"),
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: b"GPU-test",
        nvmlDeviceGetName=lambda handle: b"GPU test",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=128, used=96, free=32),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=75, memory=30),
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [SimpleNamespace(pid=pid, usedGpuMemory=72)],
        nvmlDeviceGetGraphicsRunningProcesses=lambda handle: [SimpleNamespace(pid=pid, usedGpuMemory=2 ** 64 - 1)],
        nvmlDeviceGetPowerUsage=lambda handle: 52000,
    )

    def import_module(name):
        if name == "pynvml":
            return api
        raise ImportError(name)

    monkeypatch.setattr(collectors, "importlib", SimpleNamespace(import_module=import_module))
    collector = collectors.SystemCollector()
    try:
        gpu = collector.sample()["gpus"][0]
        assert collector.sample()["sources"]["gpu"] == "nvml"
    finally:
        collector.close()
        collector.close()

    assert gpu["uuid"] == "GPU-test" and gpu["memory_percent"] == 75
    assert gpu["process_memory_used_bytes"] == 72
    assert gpu["power_w"] == 52 and gpu["temperature_c"] is None
    assert updates == ["init", "shutdown"]


def test_gpu_disabled_skips_vendor_discovery(monkeypatch):
    def vendor_discovery(name):
        raise AssertionError(f"unexpected GPU discovery: {name}")

    monkeypatch.setattr(collectors.shutil, "which", vendor_discovery)
    collector = collectors.SystemCollector(gpu=False)
    monkeypatch.setattr(collector, "_load_gpu", vendor_discovery)
    try:
        snapshot = collector.sample()
        assert snapshot["gpus"] == [] and snapshot["sources"]["gpu"] is None
    finally:
        collector.close()


def test_process_snapshot_uses_working_set_fallback_without_gpu(monkeypatch):
    def unavailable(name):
        raise ImportError(name)

    def unexpected_query(*args):
        raise AssertionError("process-only collection queried host or GPU resources")

    monkeypatch.setattr(collectors, "importlib", SimpleNamespace(import_module=unavailable))
    monkeypatch.setattr(collectors, "os", SimpleNamespace(name="nt", getpid=os.getpid))
    monkeypatch.setattr(collectors, "_windows_process_memory",
                        lambda: {"rss_bytes": 192, "rss_lifetime_peak_bytes": 256})
    monkeypatch.setattr(collectors.SystemCollector, "_sample_host", unexpected_query)
    monkeypatch.setattr(collectors.SystemCollector, "_sample_gpus", unexpected_query)

    process = collectors.process_snapshot()
    assert process["pid"] == os.getpid()
    assert process["cpu_time_s"] >= 0
    assert process["rss_bytes"] == 192
    assert process["rss_lifetime_peak_bytes"] == 256


def test_cuda_device_memory_uses_current_logical_device(monkeypatch):
    device = SimpleNamespace(id=4, mem_info=(32, 160))
    cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: device))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    assert collectors.cuda_device_memory() == {
        "device_id": 4, "memory_free_bytes": 32,
        "memory_total_bytes": 160, "memory_used_bytes": 128,
    }


def test_profiling_import_does_not_initialize_training_runtime():
    code_dir = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, "-c",
        "import sys; from eneuro.profiling import process_snapshot; "
        "assert 'eneuro.base' not in sys.modules and 'cupy' not in sys.modules",
    ], env={**os.environ, "PYTHONPATH": str(code_dir)}, check=True)
