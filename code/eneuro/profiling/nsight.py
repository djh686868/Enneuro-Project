"""Optional NVTX annotations and argv plans for NVIDIA Nsight CLI tools."""

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from functools import cached_property
import os
from pathlib import Path
import re
import shutil


NVTX_DOMAIN = "eneuro"
_NSYS_REPORTS = (
    "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,"
    "nvtx_sum,nvtx_gpu_proj_sum"
)
_WINDOWS_LOCATIONS = {
    "nsys": ("Nsight Systems *", "target-windows*/nsys.exe"),
    "ncu": ("Nsight Compute *", "target/windows*/ncu.exe"),
}


class NVTX:
    """Create scoped annotations without synchronizing CUDA streams."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    @cached_property
    def _provider(self):
        try:
            from nvtx import annotate

            if callable(annotate):
                return "nvtx", annotate
        except Exception:
            pass

        try:
            from cupy.cuda import nvtx
            from cupyx.profiler import time_range

            if nvtx.available:
                return "cupy", time_range
        except Exception:
            return None, None
        return None, None

    @property
    def backend(self) -> str | None:
        """The annotation provider, if enabled and available."""
        return self._provider[0] if self.enabled else None

    @property
    def domain(self) -> str | None:
        """The named NVTX domain, or None for the default domain."""
        return NVTX_DOMAIN if self.backend == "nvtx" else None

    def range(self, name: str):
        """Return an annotation context, or a no-op when unavailable."""
        if not self.enabled:
            return nullcontext()
        backend, factory = self._provider
        if backend == "nvtx":
            return factory(name, domain=NVTX_DOMAIN)
        if backend == "cupy":
            return factory(name)
        return nullcontext()


@contextmanager
def cuda_profiler_capture(enabled: bool = True, *, synchronize: bool = True):
    """Explicitly limit an Nsight Systems capture to a CUDA region.

    Use with ``capture_cuda=True`` in the Nsight Systems argv plan. The
    caller selects the CUDA device and ensures capture regions do not overlap
    in the same process. Synchronization, when enabled, applies to that device.
    """
    if not enabled:
        yield
        return

    from cupy.cuda import runtime
    from cupyx.profiler import profile

    if synchronize:
        runtime.deviceSynchronize()
    with profile():
        try:
            yield
        finally:
            if synchronize:
                runtime.deviceSynchronize()


def _windows_nsight_executable(name: str) -> str | None:
    install_glob, relative_glob = _WINDOWS_LOCATIONS[name]
    roots = {os.environ.get(key) for key in ("ProgramW6432", "ProgramFiles")}
    candidates = []
    for root in sorted(roots - {None, ""}):
        vendor_dir = Path(root) / "NVIDIA Corporation"
        for install_dir in vendor_dir.glob(install_glob):
            version = tuple(int(part) for part in re.findall(r"\d+", install_dir.name))
            for executable in install_dir.glob(relative_glob):
                if executable.is_file():
                    candidates.append((version, str(executable)))
    return max(candidates)[1] if candidates else None


def _find_nsight_executable(name: str) -> str | None:
    if os.name != "nt":
        return shutil.which(name)
    return (
        shutil.which(f"{name}.exe")
        or _windows_nsight_executable(name)
        or shutil.which(name)
    )


def discover_nsight() -> dict[str, str | None]:
    """Resolve installed Nsight tools on PATH or in standard Windows paths."""
    return {name: _find_nsight_executable(name) for name in _WINDOWS_LOCATIONS}


def nsight_command(
    tool: str,
    application: Sequence[str] = (),
    *,
    output: str | os.PathLike[str] | None = None,
    report: str | os.PathLike[str] | None = None,
    nvtx_include: str | None = None,
    launch_skip: int | None = None,
    launch_count: int | None = None,
    capture_cuda: bool = False,
    cpu_sampling: bool = False,
) -> list[str]:
    """Plan an ``nsys``, ``nsys_stats``, or ``ncu`` invocation as argv."""
    if tool not in ("nsys", "nsys_stats", "ncu"):
        raise ValueError("tool must be nsys, nsys_stats, or ncu")
    if isinstance(application, (str, bytes)):
        raise ValueError("application must be an argv sequence")
    target = list(application)
    if any(not isinstance(token, str) or "\0" in token for token in target):
        raise ValueError("application must contain valid string arguments")
    if target and (not target[0] or target[0].startswith("-")):
        raise ValueError("application must start with an executable")
    output_path = os.fspath(output) if output is not None else None
    if output_path is not None and (not output_path or "\0" in output_path):
        raise ValueError("output must be a nonempty path")
    if tool != "ncu" and any(value is not None for value in (nvtx_include, launch_skip, launch_count)):
        raise ValueError("NVTX and launch filters require ncu")
    if tool != "nsys" and (capture_cuda or cpu_sampling):
        raise ValueError("CUDA capture and CPU sampling require nsys")

    if tool == "nsys_stats":
        if report is None or target:
            raise ValueError("nsys_stats requires a report path and takes no application")
        report_path = os.fspath(report)
        if not report_path or report_path.startswith("-") or "\0" in report_path:
            raise ValueError("report must be a valid path")
        # Nsight also accepts lists of destinations and @-prefixed commands.
        if output_path is not None and (output_path.startswith("@") or "," in output_path):
            raise ValueError("nsys_stats output must be one file prefix or '-'")
        executable = _find_nsight_executable("nsys") or "nsys"
        return [
            executable, "stats", "--report", _NSYS_REPORTS, "--format", "csv",
            "--output", output_path or "-", report_path,
        ]

    if not target or report is not None:
        raise ValueError("profiling requires an application and takes no existing report")
    executable = _find_nsight_executable(tool) or tool
    if tool == "nsys":
        sampling = "process-tree" if cpu_sampling else "none"
        args = [
            executable, "profile", "--trace=cuda,nvtx", f"--sample={sampling}",
            f"--cpuctxsw={sampling}",
        ]
        if capture_cuda:
            args += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
        if output_path is not None:
            args.append(f"--output={output_path}")
        return args + target

    args = [executable, "--nvtx"]
    if nvtx_include is not None:
        if not nvtx_include or "\0" in nvtx_include:
            raise ValueError("nvtx_include must be a nonempty NVTX filter")
        args += ["--nvtx-include", nvtx_include]
    if launch_skip is not None:
        if type(launch_skip) is not int or launch_skip < 0:
            raise ValueError("launch_skip must be a nonnegative integer")
        args += ["--launch-skip", str(launch_skip)]
    if launch_count is not None:
        if type(launch_count) is not int or launch_count < 1:
            raise ValueError("launch_count must be a positive integer")
        args += ["--launch-count", str(launch_count)]
    if output_path is not None:
        args += ["--export", output_path]
    return args + target
