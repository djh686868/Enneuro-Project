"""Shared profiling scopes for AutoML searches and refits."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..profiling import ProfileConfig, ProfileSession, current_profile

Profiling = ProfileConfig | Mapping[str, Any] | bool | None


@contextmanager
def profile_run(profiling: Profiling, *, name: str, output_dir: Path | None = None,
                **metadata: Any):
    """Yield the session and whether the caller owns its completed report."""
    if profiling is None:
        yield current_profile(), False
        return
    config = ProfileConfig.coerce(profiling)
    if not config.enabled:
        if current_profile() is None:
            yield None, False
        else:
            with ProfileSession(config):
                yield None, False
        return
    if config.output_dir is None and output_dir is not None:
        config = replace(config, output_dir=output_dir)
    with ProfileSession(config, name=name, metadata=metadata) as profile:
        yield profile, True


def profile_reference(profile: ProfileSession, *, completed: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_id": profile.run_id,
        "report_path": str(profile.report_path) if profile.report_path is not None else None,
        "samples_path": str(profile.samples_path) if profile.samples_path is not None else None,
    }
    if completed:
        report = profile.report
        for key in ("status", "duration_s", "sample_count", "peaks", "averages",
                    "timings", "metrics", "errors"):
            result[key] = report.get(key)
    return result


def profile_window(profile: ProfileSession, start_offset_ns: int | None) -> dict[str, Any]:
    """Reference the interval in the session's sample time base."""
    return {
        "run_id": profile.run_id,
        "start_offset_ns": start_offset_ns,
        "end_offset_ns": profile.offset_ns,
    }


@contextmanager
def profile_phase(profile: ProfileSession | None, phase: str, **metadata: Any):
    if profile is None:
        yield
        return
    profile.mark(f"{phase}.start", **metadata)
    try:
        with profile.span(phase):
            yield
    finally:
        profile.mark(f"{phase}.end", **metadata)


def add_profiling_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--profile", action="store_true", help="采集训练系统指标和阶段耗时")
    parser.add_argument("--profile-interval", type=float, help="开启采集并设置采样间隔（秒）")
    parser.add_argument("--profile-nvtx", action="store_true", help="开启采集和 Nsight NVTX 标记")
    parser.add_argument("--profile-cuda-timing", action="store_true", help="开启采集和 CUDA Event 计时")


def profiling_from_args(args: Namespace) -> ProfileConfig | None:
    interval = getattr(args, "profile_interval", None)
    nvtx = getattr(args, "profile_nvtx", False)
    cuda_timing = getattr(args, "profile_cuda_timing", False)
    if not (getattr(args, "profile", False) or interval is not None or nvtx or cuda_timing):
        return None
    config = {"nvtx": nvtx, "cuda_timing": cuda_timing}
    if interval is not None:
        config["sample_interval"] = interval
    return ProfileConfig.coerce(config)
