import builtins
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType

import pytest

from eneuro.profiling import nsight


def test_nvtx_is_lazy_and_balances_ranges_on_error(monkeypatch):
    events = []
    provider = ModuleType("nvtx")

    @contextmanager
    def annotate(name, *, domain):
        events.append(("start", domain, name))
        try:
            yield
        finally:
            events.append(("stop", domain, name))

    provider.annotate = annotate
    monkeypatch.setitem(sys.modules, "nvtx", provider)

    disabled = nsight.NVTX()
    with disabled.range("forward"):
        pass
    assert disabled.backend is None
    assert "_provider" not in vars(disabled)

    enabled = nsight.NVTX(enabled=True)
    with pytest.raises(RuntimeError), enabled.range("forward"):
        raise RuntimeError("failed forward pass")
    assert enabled.backend == "nvtx"
    assert enabled.domain == "eneuro"
    assert events == [("start", "eneuro", "forward"), ("stop", "eneuro", "forward")]


def test_nvtx_ignores_optional_provider_initialization_errors(monkeypatch):
    original_import = builtins.__import__

    def fail_optional_import(name, *args, **kwargs):
        if name in {"nvtx", "cupy.cuda"}:
            raise RuntimeError("initialization failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_optional_import)
    markers = nsight.NVTX(enabled=True)
    with markers.range("forward"):
        assert markers.backend is None


def test_nsight_plans_preserve_paths_and_training_arguments(monkeypatch):
    monkeypatch.setattr(nsight, "_find_nsight_executable", lambda name: f"path with space/{name}")
    target = ["python", "train.py", "--dataset", "path with space/data"]
    systems = nsight.nsight_command("nsys", target, output=Path("trace run"), capture_cuda=True)
    assert systems == [
        "path with space/nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
        "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--output=trace run", *target,
    ]

    stats = nsight.nsight_command("nsys_stats", report=Path("trace run.nsys-rep"))
    assert stats[:3] == ["path with space/nsys", "stats", "--report"]
    assert {"cuda_gpu_kern_sum", "nvtx_sum"} <= set(stats[3].split(","))
    assert stats[4:] == ["--format", "csv", "--output", "-", "trace run.nsys-rep"]

    compute = nsight.nsight_command(
        "ncu", target, output="kernel report", nvtx_include="eneuro@forward/", launch_count=2,
    )
    assert compute == [
        "path with space/ncu", "--nvtx", "--nvtx-include", "eneuro@forward/",
        "--launch-count", "2", "--export", "kernel report", *target,
    ]
    with pytest.raises(ValueError, match="one file prefix"):
        nsight.nsight_command("nsys_stats", report="run.nsys-rep", output="safe-prefix,@command")


def test_windows_nsight_discovery_sorts_installed_versions(monkeypatch, tmp_path):
    monkeypatch.delenv("ProgramW6432", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    for version in ("2025.9", "2025.10"):
        system = tmp_path / "NVIDIA Corporation" / f"Nsight Systems {version}"
        executable = system / "target-windows-x64" / "nsys.exe"
        executable.parent.mkdir(parents=True)
        executable.touch()
    assert nsight._windows_nsight_executable("nsys") == str(executable)
    assert nsight._windows_nsight_executable("ncu") is None
