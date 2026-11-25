"""Short smoke check for EnNeuro profiling, with optional CuPy CUDA work.

Examples (run from the repository root):
  python code/profile_profiling.py
  python code/profile_profiling.py --device cuda --dataset synthetic
  python code/profile_profiling.py --device cuda --dataset npz --data data/features.npz --x-key X
  python code/profile_profiling.py --device cuda --dataset csv --data data/features.csv --label-column label
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_data(args):
    if args.dataset == "synthetic":
        rng = np.random.default_rng(args.seed)
        return rng.normal(size=(args.samples, args.features)).astype(np.float32)

    if not args.data:
        raise ValueError("--data is required for csv and npz datasets")
    path = Path(args.data)
    if args.dataset == "npz":
        with np.load(path) as archive:
            if args.x_key not in archive:
                raise ValueError("NPZ key {!r} not found; available: {}".format(
                    args.x_key, ", ".join(archive.files)))
            values = np.asarray(archive[args.x_key], dtype=np.float32)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError("CSV must have a header row")
            columns = [name for name in reader.fieldnames if name != args.label_column]
            if not columns:
                raise ValueError("CSV has no feature columns")
            rows = []
            for row in reader:
                rows.append([float(row[name]) for name in columns])
                if len(rows) >= args.samples:
                    break
        values = np.asarray(rows, dtype=np.float32)

    if values.ndim < 2:
        values = values.reshape(len(values), -1)
    else:
        values = values.reshape(values.shape[0], -1)
    values = values[:args.samples]
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("dataset must contain at least two rows and one feature")
    if not np.isfinite(values).all():
        raise ValueError("dataset features contain NaN or infinity")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="select CPU or require a working CuPy CUDA device")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dataset", choices=("synthetic", "csv", "npz"), default="synthetic")
    parser.add_argument("--data", help="CSV or NPZ input path; not needed for synthetic data")
    parser.add_argument("--x-key", default="X", help="feature array key for NPZ input")
    parser.add_argument("--label-column", default="label", help="CSV label column to exclude")
    parser.add_argument("--samples", type=int, default=256, help="maximum rows to load")
    parser.add_argument("--features", type=int, default=64, help="feature width for synthetic data")
    parser.add_argument("--seed", type=int, default=7, help="synthetic dataset seed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8, help="short training iterations")
    parser.add_argument("--sample-interval", type=float, default=0.1,
                        help="system sampling period in seconds")
    parser.add_argument("--output-dir", default="artifacts/profiling-smoke")
    parser.add_argument("--cupy-cache-dir", help="CuPy kernel cache path (default: <output-dir>/.cupy-cache)")
    parser.add_argument("--nvtx", action="store_true", help="enable NVTX annotations")
    parser.add_argument("--allocator-framework", choices=("torch", "paddle", "tensorflow"),
                        help="also query one framework CUDA allocator")
    parser.add_argument("--no-system-metrics", action="store_true",
                        help="disable background system/GPU sampling")
    args = parser.parse_args()
    if args.samples < 2 or args.features < 1 or args.batch_size < 1 or args.iterations < 1:
        parser.error("samples >= 2, features/batch-size/iterations must be positive")
    if args.device_id < 0:
        parser.error("device-id must be non-negative")
    return args


def main():
    args = parse_args()
    from eneuro.profiling import ProfileConfig, ProfileSession, cuda_device_memory, system_snapshot
    from eneuro.profiling.cuda import cuda_allocator_memory, cuda_device_name
    from eneuro.profiling.nsight import discover_nsight, nsight_command

    values = load_data(args)
    xp = np
    cp = None
    device_name = "CPU"
    if args.device == "cuda":
        cache_dir = Path(args.cupy_cache_dir or (Path(args.output_dir) / ".cupy-cache")).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
        try:
            import cupy as cp
            cp.cuda.Device(args.device_id).use()
            device_name = cuda_device_name(args.device_id)
        except Exception as exc:
            raise RuntimeError("CUDA was requested but CuPy could not initialize device {}: {}".format(
                args.device_id, exc))
        xp = cp

    config = ProfileConfig(
        enabled=True,
        output_dir=args.output_dir,
        sample_interval=args.sample_interval,
        system_metrics=not args.no_system_metrics,
        gpu=args.device == "cuda",
        cuda_timing=args.device == "cuda",
        nvtx=args.nvtx,
        batch_every=1,
    )
    info = {
        "device": args.device,
        "device_name": device_name,
        "device_id": args.device_id if args.device == "cuda" else None,
        "dataset": args.dataset,
        "data_shape": list(values.shape),
    }
    snapshot = system_snapshot(gpu=args.device == "cuda")
    device_memory = cuda_device_memory() if args.device == "cuda" else None
    allocator_memory = ({args.allocator_framework: cuda_allocator_memory(args.allocator_framework)}
                        if args.device == "cuda" and args.allocator_framework else {})

    with ProfileSession(config, name="profiling-smoke", metadata=info) as profile:
        with profile.span("data_prepare", cuda=False):
            x = xp.asarray(values)
            # Synthetic regression target derived from the selected dataset.
            target = (x[:, :1] * 0.25 + 0.1)
            weights = xp.zeros((x.shape[1], 1), dtype=x.dtype)
        for iteration in range(args.iterations):
            start = (iteration * args.batch_size) % max(1, x.shape[0] - args.batch_size + 1)
            stop = min(start + args.batch_size, x.shape[0])
            xb, yb = x[start:stop], target[start:stop]
            if not len(xb):
                xb, yb = x, target
            with profile.span("forward", cuda=args.device == "cuda"):
                prediction = xb @ weights
            with profile.span("loss", cuda=args.device == "cuda"):
                error = prediction - yb
                loss = xp.mean(error * error)
            with profile.span("backward", cuda=args.device == "cuda"):
                gradient = (xb.T @ error) / max(1, len(xb))
            with profile.span("optimizer", cuda=args.device == "cuda"):
                weights -= 0.01 * gradient
            profile.count("examples", int(len(xb)))
            if args.device == "cuda":
                profile.sample_cupy_pool()
            profile.mark("iteration", index=iteration + 1)

        if args.device == "cuda":
            # Synchronize once at the end so the smoke run validates completed GPU work.
            cp.cuda.get_current_stream().synchronize()
            profile.observe("cupy_device_memory", cp.cuda.Device().mem_info)

    report = profile.report
    nsight = discover_nsight()
    try:
        nsys_plan = nsight_command("nsys", [sys.executable, "--version"], output="smoke-check")
    except Exception as exc:
        nsys_plan = {"error": str(exc)}
    try:
        ncu_plan = nsight_command("ncu", [sys.executable, "--version"], output="smoke-check.ncu-rep")
    except Exception as exc:
        ncu_plan = {"error": str(exc)}

    summary = {
        "status": report["status"],
        "run_id": report["run_id"],
        "report_path": str(profile.report_path) if profile.report_path else None,
        "samples_path": str(profile.samples_path) if profile.samples_path else None,
        "sample_count": report["sample_count"],
        "timings": report["timings"],
        "peaks": report["peaks"],
        "errors": report["errors"],
        "nvtx_backend": report["nvtx_backend"],
        "cuda_device_name": device_name if args.device == "cuda" else None,
        "system_snapshot": snapshot,
        "cuda_device_memory": device_memory,
        "framework_allocator_memory": allocator_memory,
        "nsight": nsight,
        "nsys_plan_example": nsys_plan,
        "ncu_plan_example": ncu_plan,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if report["status"] != "completed":
        return 1
    if args.device == "cuda" and ("cuda_timing" in report["errors"] or device_memory is None):
        return 2
    if args.nvtx and report["nvtx_backend"] is None:
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        raise SystemExit(main())
    except Exception as exc:
        print("profiling smoke check failed: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
