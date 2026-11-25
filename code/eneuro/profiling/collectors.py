"""Sample system and process resources without creating a CUDA context."""

import csv
import ctypes
import importlib
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time


_MIB = 1024 ** 2
_SMI_GPU_FIELDS = (
    ("index", "index", int),
    ("uuid", "uuid", str),
    ("name", "name", str),
    ("utilization.gpu", "utilization_percent", float),
    ("utilization.memory", "memory_utilization_percent", float),
    ("memory.total", "memory_total_bytes", int),
    ("memory.used", "memory_used_bytes", int),
    ("memory.free", "memory_free_bytes", int),
    ("temperature.gpu", "temperature_c", int),
    ("power.draw", "power_w", float),
    ("driver_version", "driver_version", str),
)


def _number(value, convert=float):
    try:
        result = convert(value)
        if 0 <= result < 2 ** 64 - 1 and math.isfinite(result):
            return result
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _text(value):
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if value is not None:
        value = str(value).strip()
        if value and value.upper() not in {"N/A", "[N/A]", "[NOT SUPPORTED]"}:
            return value
    return None


def _percentage(part, total):
    return 100 * part / total if part is not None and total else None


def _try(call, *args):
    """A provider may reject a supported field on a particular host or device."""
    try:
        return call(*args)
    except Exception:
        return None


class _NvmlGPU:
    source = "nvml"

    def __init__(self):
        self.api = importlib.import_module("pynvml")
        self.api.nvmlInit()

    def close(self):
        _try(self.api.nvmlShutdown)

    def _query(self, name, *args):
        return _try(getattr(self.api, name, None), *args)

    def _process_memory(self, handle, pid):
        values = []
        for name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            for process in self._query(name, handle) or ():
                if process.pid == pid:
                    used = _number(getattr(process, "usedGpuMemory", None), int)
                    if used is not None:
                        values.append(used)
        return max(values, default=None)

    def sample(self, pid):
        result = []
        driver = _text(self._query("nvmlSystemGetDriverVersion"))
        for index in range(self.api.nvmlDeviceGetCount()):
            handle = self._query("nvmlDeviceGetHandleByIndex", index)
            if handle is None:
                continue
            memory = self._query("nvmlDeviceGetMemoryInfo", handle)
            utilization = self._query("nvmlDeviceGetUtilizationRates", handle)
            used = getattr(memory, "used", None)
            total = getattr(memory, "total", None)
            power_mw = self._query("nvmlDeviceGetPowerUsage", handle)
            result.append({
                "index": index,
                "uuid": _text(self._query("nvmlDeviceGetUUID", handle)),
                "name": _text(self._query("nvmlDeviceGetName", handle)),
                "utilization_percent": getattr(utilization, "gpu", None),
                "memory_utilization_percent": getattr(utilization, "memory", None),
                "memory_total_bytes": total,
                "memory_used_bytes": used,
                "memory_free_bytes": getattr(memory, "free", None),
                "memory_percent": _percentage(used, total),
                "process_memory_used_bytes": self._process_memory(handle, pid),
                "temperature_c": self._query("nvmlDeviceGetTemperature", handle, self.api.NVML_TEMPERATURE_GPU),
                "power_w": power_mw / 1000 if power_mw is not None else None,
                "driver_version": driver,
            })
        return result


def _parse_smi_gpus(output):
    result = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != len(_SMI_GPU_FIELDS):
            continue
        gpu = {}
        for value, (_, key, convert) in zip(row, _SMI_GPU_FIELDS):
            parsed = _text(value) if convert is str else _number(value, convert)
            if parsed is not None and key.startswith("memory_") and key.endswith("_bytes"):
                parsed *= _MIB
            gpu[key] = parsed
        if gpu["index"] is not None:
            gpu["memory_percent"] = _percentage(gpu["memory_used_bytes"], gpu["memory_total_bytes"])
            result.append(gpu)
    return result


def _parse_smi_process_memory(output, pid):
    result = {}
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) == 3 and _number(row[1], int) == pid:
            used = _number(row[2], int)
            if used is not None:
                result[row[0]] = max(used * _MIB, result.get(row[0], 0))
    return result


class _SmiGPU:
    source = "nvidia-smi"

    def __init__(self):
        self.executable = shutil.which("nvidia-smi")
        if self.executable is None:
            raise FileNotFoundError("nvidia-smi was not found on PATH")

    def close(self):
        pass

    def _query(self, kind, fields):
        result = subprocess.run(
            [self.executable, f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", check=True, timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout

    def sample(self, pid):
        gpus = _parse_smi_gpus(self._query("gpu", [field for field, _, _ in _SMI_GPU_FIELDS]))
        if not gpus:
            return gpus
        processes = _try(self._query, "compute-apps", ["gpu_uuid", "pid", "used_gpu_memory"])
        memory = _parse_smi_process_memory(processes or "", pid)
        for gpu in gpus:
            gpu["process_memory_used_bytes"] = memory.get(gpu["uuid"])
        return gpus


def _windows_memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in (
                "total", "available", "page_total", "page_available", "virtual_total",
                "virtual_available", "extended_virtual_available",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.total, status.available


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("page_fault_count", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_working_set", "working_set", "quota_peak_paged_pool", "quota_paged_pool",
            "quota_peak_nonpaged_pool", "quota_nonpaged_pool", "pagefile_usage", "peak_pagefile_usage",
        )
    ]


def _windows_process_memory():
    from ctypes import wintypes

    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WindowsProcessMemoryCounters), wintypes.DWORD]
    get_memory_info.restype = wintypes.BOOL
    counters = _WindowsProcessMemoryCounters()
    counters.length = ctypes.sizeof(counters)
    if not get_memory_info(get_process(), ctypes.byref(counters), counters.length):
        return {}
    return {"rss_bytes": counters.working_set, "rss_lifetime_peak_bytes": counters.peak_working_set}


def _linux_process_memory():
    fields = {"VmRSS": "rss_bytes", "VmHWM": "rss_lifetime_peak_bytes"}
    memory = {}
    with Path("/proc/self/status").open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in fields:
                memory[fields[name]] = int(value.split()[0]) * 1024
                if len(memory) == len(fields):
                    break
    return memory


class SystemCollector:
    """Return JSON-compatible host, current-process, and physical NVIDIA GPU samples.

    CPU load is sampled without blocking. The first sample may have no CPU
    percentage. Process percentage is relative to one logical CPU. Unsupported
    metrics have value None. A collector can be shared by concurrent readers.
    """

    def __init__(self, gpu=True):
        self._gpu_enabled = gpu
        self._lock = threading.Lock()
        self._closed = False
        self._host_checked = False
        self._gpu_checked = False
        self._psutil = None
        self._process = None
        self._last_host_cpu = None
        self._last_process_cpu = None
        self._gpu = None
        self._errors = {}

    def _load_host(self):
        try:
            self._psutil = importlib.import_module("psutil")
            self._process = self._psutil.Process()
        except Exception as error:
            self._errors["host"] = f"psutil unavailable ({type(error).__name__})"
        self._host_checked = True

    def _sample_host(self):
        if not self._host_checked:
            self._load_host()
        host = {
            "cpu_percent": None,
            "cpu_logical_count": os.cpu_count(),
            "cpu_physical_count": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "memory_available_bytes": None,
            "memory_percent": None,
            "os": platform.system(),
            "machine": platform.machine(),
        }
        if self._psutil is not None:
            host["cpu_physical_count"] = _try(self._psutil.cpu_count, False)
            cpu = _try(self._psutil.cpu_times)
            memory = _try(self._psutil.virtual_memory)
            if cpu is not None:
                total = sum(cpu) - getattr(cpu, "guest", 0) - getattr(cpu, "guest_nice", 0)
                idle = cpu.idle + getattr(cpu, "iowait", 0)
                if self._last_host_cpu is not None:
                    previous_total, previous_idle = self._last_host_cpu
                    elapsed = total - previous_total
                    if elapsed > 0:
                        host["cpu_percent"] = min(100.0, max(0.0, 100 * (1 - (idle - previous_idle) / elapsed)))
                self._last_host_cpu = total, idle
            if memory is not None:
                host.update(memory_used_bytes=memory.used, memory_total_bytes=memory.total,
                            memory_available_bytes=memory.available, memory_percent=memory.percent)
        elif os.name == "nt":
            memory = _try(_windows_memory)
            if memory is not None:
                total, available = memory
                used = total - available
                host.update(memory_used_bytes=used, memory_total_bytes=total,
                            memory_available_bytes=available, memory_percent=_percentage(used, total))
        return host

    def _sample_process(self):
        if not self._host_checked:
            self._load_host()
        now = time.monotonic()
        cpu_time = time.process_time()
        cpu_percent = None
        if self._last_process_cpu is not None:
            previous_time, previous_cpu = self._last_process_cpu
            cpu_percent = _percentage(cpu_time - previous_cpu, now - previous_time)
        self._last_process_cpu = now, cpu_time

        process = {
            "pid": os.getpid(),
            "cpu_percent": cpu_percent,
            "cpu_time_s": cpu_time,
            "rss_bytes": None,
            "rss_lifetime_peak_bytes": None,
            "thread_count": None,
            "read_bytes": None,
            "write_bytes": None,
        }
        if self._process is not None:
            memory = _try(self._process.memory_info)
            io = _try(getattr(self._process, "io_counters", None))
            process.update(rss_bytes=getattr(memory, "rss", None),
                           rss_lifetime_peak_bytes=getattr(memory, "peak_wset", None),
                           thread_count=_try(self._process.num_threads),
                           read_bytes=_number(getattr(io, "read_bytes", None), int),
                           write_bytes=_number(getattr(io, "write_bytes", None), int))
        if process["rss_bytes"] is None or process["rss_lifetime_peak_bytes"] is None:
            memory = {}
            if os.name == "nt":
                memory = _try(_windows_process_memory) or {}
            elif sys.platform.startswith("linux"):
                memory = _try(_linux_process_memory) or {}
            for name, value in memory.items():
                if process[name] is None:
                    process[name] = value
        return process

    def _load_gpu(self):
        errors = []
        for provider in (_NvmlGPU, _SmiGPU):
            try:
                self._gpu = provider()
                break
            except Exception as error:
                errors.append(f"{provider.source} unavailable ({type(error).__name__})")
        if self._gpu is None:
            self._errors["gpu"] = ", ".join(errors)
        self._gpu_checked = True

    def _sample_gpus(self, pid):
        if not self._gpu_enabled:
            return []
        if not self._gpu_checked:
            self._load_gpu()
        if self._gpu is None:
            return []
        try:
            result = self._gpu.sample(pid)
        except Exception as error:
            self._errors["gpu"] = f"{self._gpu.source} sampling failed ({type(error).__name__})"
            if self._gpu.source != "nvml":
                return []
            self._gpu.close()
            self._gpu = _try(_SmiGPU)
            return self._sample_gpus(pid)
        self._errors.pop("gpu", None)
        return result

    def sample(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("system collector is closed")
            timestamp = time.time_ns()
            host = self._sample_host()
            process = self._sample_process()
            gpus = self._sample_gpus(process["pid"])
            snapshot = {
                "timestamp_ns": timestamp,
                "host": host,
                "process": process,
                "gpus": gpus,
                "sources": {
                    "host": "psutil" if self._psutil is not None else "stdlib",
                    "process": "psutil" if self._process is not None else "stdlib",
                    "gpu": self._gpu.source if self._gpu is not None else None,
                },
            }
            if self._errors:
                snapshot["errors"] = self._errors.copy()
            return snapshot

    def sample_process(self):
        """Read current-process metrics without querying host or GPU resources."""
        with self._lock:
            if self._closed:
                raise RuntimeError("system collector is closed")
            return self._sample_process()

    def close(self):
        with self._lock:
            if not self._closed:
                if self._gpu is not None:
                    self._gpu.close()
                self._closed = True


def system_snapshot(gpu=True):
    """Take a one-off snapshot using a short baseline for CPU utilization."""
    collector = SystemCollector(gpu=gpu)
    try:
        collector._sample_host()
        collector._sample_process()
        time.sleep(0.05)
        return collector.sample()
    finally:
        collector.close()


def process_snapshot():
    """Read current process CPU clock and memory immediately."""
    return SystemCollector(gpu=False).sample_process()


def cuda_device_memory():
    """Read memory on the current CuPy logical device, if available."""
    try:
        device = importlib.import_module("cupy").cuda.Device()
        free, total = device.mem_info
    except Exception:
        return None
    return {"device_id": device.id, "memory_free_bytes": free,
            "memory_total_bytes": total, "memory_used_bytes": total - free}
