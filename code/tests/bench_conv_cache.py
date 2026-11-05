# -*- coding: utf-8 -*-
"""Conv2d CPU cache benchmark (time + optional perf counters)."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_SHAPE = (8, 16, 128, 128)
DEFAULT_KERNEL = (32, 3, 3)
DEFAULT_STRIDE = (1, 1)
DEFAULT_PAD = (1, 1)
DEFAULT_DILATION = (1, 1)
DEFAULT_CACHE_LEVEL = "l2"

CACHE_EVENT_CANDIDATES = {
    "l1": [
        ("L1-dcache-loads", "L1-dcache-load-misses"),
    ],
    "l2": [
        ("L2-dcache-loads", "L2-dcache-load-misses"),
        ("L2_RQSTS:references", "L2_RQSTS:misses"),
        ("L2_RQSTS.REFERENCES", "L2_RQSTS.MISS"),
        ("l2_rqsts.references", "l2_rqsts.miss"),
    ],
    "llc": [
        ("LLC-loads", "LLC-load-misses"),
    ],
}


def _parse_tuple(text, size):
    parts = [int(p.strip()) for p in text.split(",") if p.strip()]
    if len(parts) != size:
        raise ValueError(f"Expected {size} values, got {len(parts)}")
    return tuple(parts)


def _configure_env(mode):
    if mode == "baseline":
        os.environ["ENE_CONV_CPU_TILED"] = "0"
    else:
        os.environ["ENE_CONV_CPU_TILED"] = "1"


def _run_conv_case(iters, warmup, shape, kernel, stride, pad, dilation, seed):
    from eneuro.base import Tensor, as_Tensor  # noqa: E402
    from eneuro.base.functions import conv2d  # noqa: E402

    np.random.seed(seed)

    n, c, h, w_in = shape
    oc, kh, kw = kernel

    x = as_Tensor(np.random.randn(n, c, h, w_in).astype(np.float32))
    w = Tensor(np.random.randn(oc, c, kh, kw).astype(np.float32))
    b = Tensor(np.random.randn(oc).astype(np.float32))

    for _ in range(warmup):
        _ = conv2d(x, w, b, stride=stride, pad=pad, dilation=dilation)

    t0 = time.perf_counter()
    y = None
    for _ in range(iters):
        y = conv2d(x, w, b, stride=stride, pad=pad, dilation=dilation)
    t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = (total * 1000.0) / iters

    return {
        "avg_ms": avg_ms,
        "total_s": total,
        "iters": iters,
        "output_shape": tuple(y.shape) if y is not None else None,
    }


def _perf_stat(cmd, events):
    perf = shutil.which("perf")
    if perf is None:
        raise RuntimeError("perf not found in PATH")

    event_arg = ",".join(events)
    perf_cmd = [perf, "stat", "-x,", "-e", event_arg] + cmd
    result = subprocess.run(perf_cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "perf stat failed")

    counters = {}
    for line in result.stderr.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        count, _, event = parts[0], parts[1], parts[2]
        if count in ("<not supported>", "<not counted>"):
            continue
        count = count.replace(",", "")
        if not re.match(r"^-?\d+$", count):
            continue
        counters[event] = int(count)
    return counters, result.stdout


def _calc_miss_rate(counters, load_key, miss_key):
    loads = counters.get(load_key)
    misses = counters.get(miss_key)
    if loads is None or misses is None or loads == 0:
        return None
    return misses / loads


def _calc_hit_rate(counters, load_key, miss_key):
    miss_rate = _calc_miss_rate(counters, load_key, miss_key)
    if miss_rate is None:
        return None
    return 1.0 - miss_rate


def _calc_delta(base_value, tiled_value):
    if base_value is None or tiled_value is None:
        return None
    return tiled_value - base_value


def _calc_pct_change(base_value, tiled_value):
    if base_value is None or tiled_value is None or base_value == 0:
        return None
    return (tiled_value - base_value) / base_value


def _resolve_cache_events(cache_level):
    candidates = CACHE_EVENT_CANDIDATES.get(cache_level)
    if not candidates:
        raise ValueError(f"Unsupported cache level: {cache_level}")

    perf = shutil.which("perf")
    if perf is None:
        raise RuntimeError("perf not found in PATH")

    probe_cmd = [sys.executable, "-c", "pass"]
    for load_key, miss_key in candidates:
        probe = subprocess.run(
            [perf, "stat", "-x,", "-e", f"{load_key},{miss_key}"] + probe_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return load_key, miss_key

    load_key, miss_key = candidates[0]
    return load_key, miss_key


def _run_subprocess(args, mode, metrics, events):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--shape",
        ",".join(str(v) for v in args.shape),
        "--kernel",
        ",".join(str(v) for v in args.kernel),
        "--stride",
        ",".join(str(v) for v in args.stride),
        "--pad",
        ",".join(str(v) for v in args.pad),
        "--dilation",
        ",".join(str(v) for v in args.dilation),
        "--seed",
        str(args.seed),
        "--json",
    ]

    if metrics == "perf":
        counters, stdout = _perf_stat(cmd, events)
        data = json.loads(stdout.strip().splitlines()[-1])
        data["counters"] = counters
        return data

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"🔴 子进程 ({mode}) 运行失败，错误信息如下：")
        print(result.stderr)
        raise RuntimeError(f"Subprocess failed with code {result.returncode}")

    # 解析标准输出中的 JSON (取最后一行以防有警告混入)
    output_lines = result.stdout.strip().splitlines()
    if not output_lines:
        raise ValueError("子进程没有输出任何内容")

    try:
        data = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Subprocess ({mode}) produced invalid JSON: {output_lines[-1]}"
        ) from exc

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "tiled"], default="tiled")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--metrics", choices=["time", "perf"], default="time")
    parser.add_argument("--cache-level", choices=["l1", "l2", "llc"], default=DEFAULT_CACHE_LEVEL)
    parser.add_argument("--perf-events", default=None)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--shape", type=lambda s: _parse_tuple(s, 4), default=DEFAULT_SHAPE)
    parser.add_argument("--kernel", type=lambda s: _parse_tuple(s, 3), default=DEFAULT_KERNEL)
    parser.add_argument("--stride", type=lambda s: _parse_tuple(s, 2), default=DEFAULT_STRIDE)
    parser.add_argument("--pad", type=lambda s: _parse_tuple(s, 2), default=DEFAULT_PAD)
    parser.add_argument("--dilation", type=lambda s: _parse_tuple(s, 2), default=DEFAULT_DILATION)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.compare:
        events = []
        selected_level = args.cache_level
        load_key = miss_key = None

        if args.metrics == "perf":
            if args.perf_events:
                events = [e.strip() for e in args.perf_events.split(",") if e.strip()]
                if len(events) < 2:
                    raise ValueError("--perf-events needs at least two events: load,miss")
                load_key, miss_key = events[0], events[1]
            else:
                load_key, miss_key = _resolve_cache_events(selected_level)
                events = [load_key, miss_key]

        baseline = _run_subprocess(args, "baseline", args.metrics, events)
        tiled = _run_subprocess(args, "tiled", args.metrics, events)
        speedup = baseline["avg_ms"] / tiled["avg_ms"] if tiled["avg_ms"] else None

        report = {
            "baseline": baseline,
            "tiled": tiled,
            "speedup": speedup,
            "cache_level": selected_level,
        }

        if args.metrics == "perf":
            base_counters = baseline.get("counters", {})
            tiled_counters = tiled.get("counters", {})
            report["baseline"][f"{selected_level}_miss_rate"] = _calc_miss_rate(
                base_counters, load_key, miss_key
            )
            report["baseline"][f"{selected_level}_hit_rate"] = _calc_hit_rate(
                base_counters, load_key, miss_key
            )
            report["tiled"][f"{selected_level}_miss_rate"] = _calc_miss_rate(
                tiled_counters, load_key, miss_key
            )
            report["tiled"][f"{selected_level}_hit_rate"] = _calc_hit_rate(
                tiled_counters, load_key, miss_key
            )

        report["summary"] = {
            "avg_ms_delta": _calc_delta(baseline["avg_ms"], tiled["avg_ms"]),
            "avg_ms_pct_change": _calc_pct_change(baseline["avg_ms"], tiled["avg_ms"]),
            "speedup": speedup,
            "speedup_over_baseline_pct": None if speedup is None else (speedup - 1.0),
            "interpretation": (
                "tiled is faster" if (speedup is not None and speedup > 1.0) else "tiled is not faster"
            ),
        }

        if args.metrics == "perf":
            base_miss = report["baseline"][f"{selected_level}_miss_rate"]
            tiled_miss = report["tiled"][f"{selected_level}_miss_rate"]
            base_hit = report["baseline"][f"{selected_level}_hit_rate"]
            tiled_hit = report["tiled"][f"{selected_level}_hit_rate"]

            report["summary"].update({
                "cache_level": selected_level,
                f"{selected_level}_miss_rate_base": base_miss,
                f"{selected_level}_miss_rate_tiled": tiled_miss,
                f"{selected_level}_miss_rate_delta": _calc_delta(base_miss, tiled_miss),
                f"{selected_level}_miss_rate_pct_change": _calc_pct_change(base_miss, tiled_miss),
                f"{selected_level}_hit_rate_base": base_hit,
                f"{selected_level}_hit_rate_tiled": tiled_hit,
                f"{selected_level}_hit_rate_delta": _calc_delta(base_hit, tiled_hit),
                f"{selected_level}_hit_rate_pct_change": _calc_pct_change(base_hit, tiled_hit),
                "interpretation": (
                    f"{selected_level.upper()} hit rate improved"
                    if (base_hit is not None and tiled_hit is not None and tiled_hit > base_hit)
                    else f"{selected_level.upper()} hit rate did not improve"
                ),
            })

        print(json.dumps(report, indent=2))
        return

    _configure_env(args.mode)

    # Delay imports until env is set.
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    stats = _run_conv_case(
        iters=args.iters,
        warmup=args.warmup,
        shape=args.shape,
        kernel=args.kernel,
        stride=args.stride,
        pad=args.pad,
        dilation=args.dilation,
        seed=args.seed,
    )

    result = {
        "mode": args.mode,
        "shape": args.shape,
        "kernel": args.kernel,
        "stride": args.stride,
        "pad": args.pad,
        "dilation": args.dilation,
        **stats,
    }

    if args.json:
        print(json.dumps(result))
        return

    print("==== conv2d cpu cache bench ====")
    print(f"mode: {args.mode}")
    print(f"shape: {args.shape} kernel: {args.kernel}")
    print(f"stride: {args.stride} pad: {args.pad} dilation: {args.dilation}")
    print(f"iters: {args.iters} warmup: {args.warmup}")
    print(f"avg: {stats['avg_ms']:.4f} ms")
    print(f"total: {stats['total_s']:.4f} s")


if __name__ == "__main__":
    main()
