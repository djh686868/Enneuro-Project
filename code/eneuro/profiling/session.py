"""Collect system samples and phase timings for one training run."""

from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

from .collectors import SystemCollector
from .nsight import NVTX
from .timing import CudaEventTimer


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool = True
    output_dir: str | Path | None = None
    sample_interval: float = 1.0
    system_metrics: bool = True
    gpu: bool = True
    cuda_timing: bool = False
    nvtx: bool = False
    batch_every: int = 10
    max_marks: int = 256
    max_cuda_events: int = 1024

    def __post_init__(self):
        if not math.isfinite(self.sample_interval) or self.sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        for name in ("batch_every", "max_marks", "max_cuda_events"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def coerce(cls, value):
        if value is None or isinstance(value, bool):
            return cls(enabled=bool(value))
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        raise TypeError("profiling must be a ProfileConfig, a mapping, a bool, or None")


_ACTIVE = ContextVar("eneuro_active_profile", default=None)


def current_profile():
    """Return the enabled session in the current execution context."""
    return _ACTIVE.get()


def _json_value(value):
    # Preserve numeric JSON values, including NumPy scalar metrics, and handle NaN.
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _update_stat(stats, key, value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return
    entry = stats.setdefault(key, {"count": 0, "total": 0.0, "min": value, "max": value})
    entry["count"] += 1
    entry["total"] += value
    entry["min"] = min(entry["min"], value)
    entry["max"] = max(entry["max"], value)


class ProfileSession:
    """Opt-in context manager. Use spans for durations and marks for run metadata."""

    def __init__(self, config=None, *, name="training", metadata=None):
        self.config = ProfileConfig.coerce(config)
        self.enabled = self.config.enabled
        self.name = name
        self.run_id = uuid4().hex if self.enabled else None
        self._metadata = _json_value(metadata or {}) if self.enabled else {}
        output = Path(self.config.output_dir).resolve() if self.enabled and self.config.output_dir is not None else None
        self.report_path = output / f"{self.run_id}.json" if output is not None else None
        self.samples_path = (output / f"{self.run_id}.samples.jsonl"
                             if output is not None and self.config.system_metrics else None)
        self._status = "created"
        self._start_ns = self._start_unix_ns = self._end_ns = None
        self._token = None
        self._worker = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._publish_interval = max(1.0, self.config.sample_interval)
        self._last_published = 0.0
        self._nvtx = NVTX(self.enabled and self.config.nvtx)
        self._nvtx_backend = None
        self._cuda_drain_lock = threading.Lock()
        self._pending_cuda = deque()
        self._cuda_unavailable = False
        self._cuda_dropped = 0
        self._cuda_pool_unavailable = False
        self._cuda_pools = {}
        self._timings = {}
        self._counters = {}
        self._observations = {}
        self._errors = {}
        self._marks = deque(maxlen=self.config.max_marks)
        self._mark_count = 0
        self._last_sample = None
        self._sample_count = 0
        self._system_stats = {"host": {}, "process": {}, "gpus": {}}

    def __enter__(self):
        if self._status != "created":
            raise RuntimeError("ProfileSession instances are single-use")
        self._status = "running"
        if not self.enabled:
            self._token = _ACTIVE.set(None)
            return self
        self._nvtx_backend = self._nvtx.backend
        if self.config.nvtx and self._nvtx_backend is None:
            self._error("nvtx", "no NVTX provider is available")
        self._start_unix_ns = time.time_ns()
        self._start_ns = time.perf_counter_ns()
        self._token = _ACTIVE.set(self)
        self._publish()
        if self.config.system_metrics:
            self._worker = threading.Thread(target=self._collect, name="EnNeuro-Profiler", daemon=True)
            self._worker.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        _ACTIVE.reset(self._token)
        if not self.enabled:
            self._status = "completed"
            return
        self._end_ns = time.perf_counter_ns()
        self._stop.set()
        if self._worker is not None:
            self._worker.join()
        self._drain_cuda(wait=True)
        if exc is not None:
            self._error("workload", type(exc).__name__)
        self._status = "failed" if exc is not None else "completed"
        self._publish()

    def _error(self, source, error):
        with self._lock:
            self._errors.setdefault(source, str(error))

    def _collect(self):
        collector = None
        try:
            collector = SystemCollector(gpu=self.config.gpu)
            while True:
                self._sample(collector)
                if self._stop.is_set():
                    break
                if time.monotonic() - self._last_published >= self._publish_interval:
                    self._publish()
                self._stop.wait(self.config.sample_interval)
        except Exception as exc:
            self._error("system_metrics", exc)
        finally:
            if collector is not None:
                try:
                    collector.close()
                except Exception as exc:
                    self._error("system_metrics", exc)

    def _sample(self, collector):
        offset_ns = time.perf_counter_ns() - self._start_ns
        sample = {**collector.sample(), "offset_ns": offset_ns}
        with self._lock:
            self._last_sample = sample
            self._sample_count += 1
            for scope in ("host", "process"):
                for key, value in sample[scope].items():
                    if key.endswith(("_percent", "_bytes")):
                        _update_stat(self._system_stats[scope], key, value)
            for gpu in sample["gpus"]:
                identity = gpu.get("uuid") or str(gpu["index"])
                gpu_stats = self._system_stats["gpus"].setdefault(identity, {
                    "index": gpu["index"], "name": gpu.get("name"), "metrics": {},
                })
                for key, value in gpu.items():
                    if key.endswith(("_percent", "_bytes", "_w", "_c")):
                        _update_stat(gpu_stats["metrics"], key, value)
            for source, error in sample.get("errors", {}).items():
                if error:
                    self._errors.setdefault(source, str(error))
        if self.samples_path is not None:
            try:
                with self.samples_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(_json_value(sample), ensure_ascii=False, allow_nan=False) + "\n")
            except OSError as exc:
                self._error("samples_output", exc)

    def _publish(self):
        if self.report_path is None:
            return
        self._last_published = time.monotonic()
        tmp_path = self.report_path.with_suffix(".json.tmp")
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(self.report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            for attempt in range(3):
                try:
                    tmp_path.replace(self.report_path)
                    break
                except PermissionError:
                    # On Windows a reader may briefly block renaming the report.
                    if attempt == 2:
                        raise
                    time.sleep(0.01)
        except OSError as exc:
            self._error("report_output", exc)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @property
    def offset_ns(self):
        """Monotonic time since the session started, shared by spans and samples."""
        if self._start_ns is None:
            return None
        return (self._end_ns or time.perf_counter_ns()) - self._start_ns

    @property
    def report(self):
        if not self.enabled:
            return None
        offset = self.offset_ns
        with self._lock:
            timing_stats = {
                phase: {
                    unit: {**entry, "mean": entry["total"] / entry["count"]}
                    for unit, entry in units.items()
                }
                for phase, units in self._timings.items()
            }
            peaks = {}
            averages = {}
            for scope in ("host", "process"):
                stats = self._system_stats[scope]
                peaks[scope] = {key: entry["max"] for key, entry in stats.items()}
                averages[scope] = {key: entry["total"] / entry["count"] for key, entry in stats.items()}
            peaks["gpus"] = {
                identity: {
                    "index": gpu["index"], "name": gpu["name"],
                    **{key: entry["max"] for key, entry in gpu["metrics"].items()},
                }
                for identity, gpu in self._system_stats["gpus"].items()
            }
            averages["gpus"] = {
                identity: {
                    key: entry["total"] / entry["count"] for key, entry in gpu["metrics"].items()
                }
                for identity, gpu in self._system_stats["gpus"].items()
            }
            peaks["cupy_default_pool"] = {
                device: {key: entry["max"] for key, entry in stats.items()}
                for device, stats in self._cuda_pools.items()
            }
            metrics = dict(self._counters)
            for phase in ("train", "validation"):
                seconds = timing_stats.get(phase, {}).get("wall_ms", {}).get("total", 0) / 1000
                samples = metrics.get(f"{phase}_samples")
                if seconds and samples is not None:
                    metrics[f"{phase}_samples_per_s"] = samples / seconds
            return _json_value({
                "schema_version": 1,
                "run_id": self.run_id, "name": self.name, "status": self._status,
                "config": asdict(self.config), "metadata": self._metadata,
                "start_unix_ns": self._start_unix_ns,
                "duration_s": offset / 1e9 if offset is not None else None,
                "sample_count": self._sample_count, "last_sample": self._last_sample,
                "peaks": peaks, "averages": averages, "timings": timing_stats, "metrics": metrics,
                "observations": self._observations, "marks": list(self._marks),
                "marks_dropped": self._mark_count - len(self._marks),
                "nvtx_backend": self._nvtx_backend, "cuda_events_dropped": self._cuda_dropped,
                "errors": self._errors,
            })

    def mark(self, name, **metadata):
        if self.enabled and self._status == "running" and self.config.max_marks:
            mark = {"name": name, "offset_ns": self.offset_ns,
                    "metadata": _json_value(metadata)}
            with self._lock:
                self._marks.append(mark)
                self._mark_count += 1

    def count(self, name, value=1):
        if self.enabled and self._status == "running":
            with self._lock:
                self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name, metrics):
        """Keep the latest application metrics, e.g. from DeviceAllocator.stats()."""
        if self.enabled and self._status == "running":
            observation = {"offset_ns": self.offset_ns,
                           "metrics": _json_value(metrics)}
            with self._lock:
                self._observations[name] = observation

    def sample_cupy_pool(self):
        """Capture the CuPy default allocator's counters on the active CUDA device."""
        if not self.enabled or self._status != "running" or not self.config.gpu or self._cuda_pool_unavailable:
            return
        try:
            import cupy as cp

            device = str(cp.cuda.Device().id)
            pool = cp.get_default_memory_pool()
            metrics = {"used_bytes": pool.used_bytes(), "reserved_bytes": pool.total_bytes()}
        except Exception as exc:
            self._error("cupy_default_pool", exc)
            self._cuda_pool_unavailable = True
            return
        with self._lock:
            stats = self._cuda_pools.setdefault(device, {})
            for key, value in metrics.items():
                _update_stat(stats, key, value)
        self.observe(f"cupy_default_pool:{device}", metrics)

    @contextmanager
    def span(self, name, *, cuda=False):
        """Measure wall and current-thread CPU time, optionally current-stream CUDA time."""
        if not self.enabled or self._status != "running":
            yield
            return
        cuda_start = self._start_cuda() if cuda and self.config.cuda_timing else None
        exhausted = False
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            with self._nvtx.range(name):
                yield
        except StopIteration:
            exhausted = True
            raise
        finally:
            if not exhausted:
                wall_elapsed = time.perf_counter_ns() - wall
                cpu_elapsed = time.thread_time_ns() - cpu
                with self._lock:
                    _update_stat(self._timings.setdefault(name, {}), "wall_ms", wall_elapsed / 1e6)
                    _update_stat(self._timings[name], "thread_cpu_ms", cpu_elapsed / 1e6)
                if cuda_start is not None:
                    self._finish_cuda(name, cuda_start)

    def _start_cuda(self):
        if self._cuda_unavailable:
            return None
        if len(self._pending_cuda) >= self.config.max_cuda_events:
            self._drain_cuda(wait=False)
            if len(self._pending_cuda) >= self.config.max_cuda_events:
                self._cuda_dropped += 1
                return None
        try:
            return CudaEventTimer()
        except Exception as exc:
            self._error("cuda_timing", exc)
            self._cuda_unavailable = True
            return None

    def _finish_cuda(self, name, pending):
        try:
            with pending.device:
                pending.stop()
            self._pending_cuda.append((name, pending))
        except Exception as exc:
            self._error("cuda_timing", exc)
            self._cuda_unavailable = True

    def _drain_cuda(self, *, wait):
        with self._cuda_drain_lock:
            for _ in range(len(self._pending_cuda)):
                name, pending = self._pending_cuda.popleft()
                try:
                    with pending.device:
                        elapsed = pending.elapsed_ms(wait=wait)
                    if elapsed is None:
                        self._pending_cuda.append((name, pending))
                        continue
                    with self._lock:
                        _update_stat(self._timings[name], "cuda_ms", elapsed)
                except Exception as exc:
                    self._error("cuda_timing", exc)
                    self._cuda_unavailable = True
