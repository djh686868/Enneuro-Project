"""Read-only Model Context Protocol tools for local training profiles."""

import argparse
from collections import deque
import json
from pathlib import Path
import re
from typing import Any

from .collectors import system_snapshot
from .nsight import discover_nsight, nsight_command


_RUN_ID = re.compile(r"[0-9a-f]{32}")


def _run_path(root, run_id, suffix=".json"):
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be a 32-character lowercase hexadecimal identifier")
    path = (root / f"{run_id}{suffix}").resolve()
    if path.parent != root:
        raise ValueError("profile artifact is outside the configured directory")
    return path


def _read_report(root, run_id):
    return json.loads(_run_path(root, run_id).read_text(encoding="utf-8"))


def _rank_timings(timings, names, unit):
    phases = [
        {"phase": name, "mean_ms": timing["mean"], "count": timing["count"]}
        for name in names
        if (timing := timings.get(name, {}).get(unit)) is not None
    ]
    return sorted(phases, key=lambda phase: phase["mean_ms"], reverse=True)


def create_server(profile_dir):
    """Build a stdio MCP server scoped to one directory of profile artifacts."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    root = Path(profile_dir).resolve()
    server = FastMCP("EnNeuro profiling")
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @server.tool(annotations=read_only)
    def get_system_snapshot(gpu: bool = True) -> dict[str, Any]:
        """Read host resource usage and visible GPUs. The process fields refer to the MCP server."""
        return system_snapshot(gpu=gpu)

    @server.tool(annotations=read_only)
    def list_profiles(limit: int = 20) -> list[dict[str, Any]]:
        """List recent running and completed profiles in the configured directory."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        paths = []
        for path in root.glob("*.json"):
            if _RUN_ID.fullmatch(path.stem) and path.is_file() and path.resolve().parent == root:
                paths.append((path.stat().st_mtime_ns, path.stem))
        profiles = []
        for _, run_id in sorted(paths, reverse=True)[:limit]:
            report = _read_report(root, run_id)
            keys = ("run_id", "name", "status", "start_unix_ns", "duration_s", "sample_count")
            profiles.append({key: report[key] for key in keys})
        return profiles

    @server.tool(annotations=read_only)
    def get_profile(run_id: str) -> dict[str, Any]:
        """Read a profile's metadata, sampled peaks, timings, counters, marks and collector status."""
        return _read_report(root, run_id)

    @server.tool(annotations=read_only)
    def get_profile_samples(run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Read the latest timestamped host and GPU samples for a training profile."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        path = _run_path(root, run_id, ".samples.jsonl")
        if not path.exists():
            _read_report(root, run_id)
            return []
        with path.open("r", encoding="utf-8") as stream:
            lines = deque((line for line in stream if line.endswith("\n")), maxlen=limit)
        return [json.loads(line) for line in lines]

    @server.tool(annotations=read_only)
    def analyze_profile(run_id: str) -> dict[str, Any]:
        """Rank instrumented Trainer phases by mean time and return resource and throughput figures."""
        report = _read_report(root, run_id)
        phases = ("data_wait", "transfer", "forward", "loss", "backward", "optimizer")
        return {
            "run_id": run_id,
            "status": report["status"],
            "sample_count": report["sample_count"],
            "duration_s": report["duration_s"],
            "peaks": report["peaks"],
            "averages": report["averages"],
            "metrics": report["metrics"],
            "trainer_wall": _rank_timings(report["timings"], phases, "wall_ms"),
            "trainer_cuda_stream": _rank_timings(report["timings"], phases, "cuda_ms"),
            "errors": report["errors"],
        }

    @server.tool(annotations=read_only)
    def get_nsight_tools() -> dict[str, str | None]:
        """Locate NVIDIA Nsight Systems and Compute CLI executables."""
        return discover_nsight()

    @server.tool(annotations=read_only)
    def plan_nsight(
        tool: str, application: list[str] | None = None, output: str | None = None,
        report: str | None = None, nvtx_include: str | None = None,
        launch_skip: int | None = None, launch_count: int | None = None,
        capture_cuda: bool = False, cpu_sampling: bool = False,
    ) -> dict[str, list[str]]:
        """Build argv for nsys, nsys_stats or ncu. The caller can run the returned command locally."""
        return {"argv": nsight_command(
            tool, application or (), output=output, report=report,
            nvtx_include=nvtx_include, launch_skip=launch_skip, launch_count=launch_count,
            capture_cuda=capture_cuda, cpu_sampling=cpu_sampling,
        )}

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", default="artifacts/profiles",
                        help="directory shared with ProfileConfig.output_dir")
    args = parser.parse_args()
    try:
        server = create_server(args.profile_dir)
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise
        parser.exit(2, "Install the optional MCP SDK with pip install -r requirements-profiling.txt\n")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
