"""三方验收 benchmark：NumPy CPU、CuPy、CUDA C RawModule。

运行：
    $env:PYTHONPATH = "code"
    python code/bench_cuda_three_way.py --warmup 5 --iters 30

计时范围包含算子本身，不包含首次 RawModule 编译；每次 GPU 计时前后同步。
脚本同时检查三方结果，避免把错误实现的高速度当成优化收益。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="artifacts/cuda_stage1_probe/three_way_benchmark.json")
    ap.add_argument("--plot", default=None, help="柱状图输出路径，默认与 JSON 同目录")
    ap.add_argument("--backends", nargs="+", choices=("numpy", "cupy", "rawmodule"),
                    default=("numpy", "cupy", "rawmodule"),
                    help="Select measured backends; rawmodule alone reuses the fixed NumPy oracle")
    args = ap.parse_args()

    import cupy as cp
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CUDA device unavailable")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eneuro.base.cuda import dispatch

    rng = np.random.default_rng(20260914)
    x_np = rng.normal(size=(1, 3, 28, 28)).astype(np.float32)
    w_np = rng.normal(size=(6, 3, 5, 5)).astype(np.float32)
    b_np = rng.normal(size=(6,)).astype(np.float32)
    v_np = rng.normal(size=(1 << 20)).astype(np.float32)
    results = {}
    # The numerical oracle is computed once outside the timed loop. This
    # allows a RawModule-only rerun without timing the unchanged CPU path.
    ref_z, ref_y = None, None

    def sync(kind):
        if kind != "numpy":
            cp.cuda.Stream.null.synchronize()

    def workload(kind):
        if kind == "numpy":
            z = np.maximum(np.exp(v_np), 0)
            # 与 GPU 路径相同的卷积数学流程，使用已有 NumPy im2col 基线。
            from eneuro.base.functions import im2col_array
            col = im2col_array(x_np, (5, 5), (1, 1), (2, 2), True, xp=np)
            y = col.dot(w_np.reshape(6, -1).T).reshape(1, 28, 28, 6).transpose(0, 3, 1, 2)
            return z, y + b_np.reshape(1, 6, 1, 1)
        dispatch.set_backend(kind)
        # 实际计时复用显存中的输入，排除每次迭代的 host→device 拷贝。
        if not hasattr(workload, "gpu_inputs"):
            workload.gpu_inputs = (cp.asarray(v_np), cp.asarray(x_np), cp.asarray(w_np), cp.asarray(b_np))
        v, x, w, b = workload.gpu_inputs
        z = dispatch.relu_forward(dispatch.exp_forward(v))
        y = dispatch.conv2d_forward(x, w, b, stride=1, pad=2)
        return z, y

    # 先预热，RawModule 编译和 CuPy kernel cache 不进入稳定计时。
    ref_z, ref_y = workload("numpy")
    for kind in args.backends:
        try:
            for _ in range(args.warmup):
                workload(kind)
            sync(kind)
        except Exception as exc:
            results[kind] = {"status": "unavailable", "error": repr(exc)}
            continue

        start = time.perf_counter()
        if kind == "rawmodule":
            dispatch.launch_counts(reset=True)
        for _ in range(args.iters):
            out = workload(kind)
        sync(kind)
        elapsed = (time.perf_counter() - start) / args.iters * 1000.0
        results[kind] = {"status": "ok", "mean_ms": elapsed}
        if kind == "rawmodule":
            results[kind]["cuda_kernel_launches"] = dispatch.launch_counts()
        if kind != "numpy":
            z, y = out
            results[kind]["max_abs_error"] = max(
                float(np.max(np.abs(cp.asnumpy(z) - ref_z))),
                float(np.max(np.abs(cp.asnumpy(y) - ref_y))),
            )

    cpu_ms = results.get("numpy", {}).get("mean_ms")
    cupy_ms = results.get("cupy", {}).get("mean_ms")
    raw_ms = results.get("rawmodule", {}).get("mean_ms")
    if cpu_ms and cupy_ms:
        results["cupy"]["speedup_vs_numpy"] = cpu_ms / cupy_ms
    if cpu_ms and raw_ms:
        results["rawmodule"]["speedup_vs_numpy"] = cpu_ms / raw_ms
    if cupy_ms and raw_ms:
        results["rawmodule"]["speedup_vs_cupy"] = cupy_ms / raw_ms

    report = {
        "schema_version": 1,
        "device": cp.cuda.Device().compute_capability,
        "cupy": cp.__version__,
        "warmup": args.warmup,
        "iterations": args.iters,
        "results": results,
        "acceptance": {
            "numerical_match": all(
                r.get("status") != "ok" or r.get("max_abs_error", 0.0) <= 2e-4
                for k, r in results.items() if k in ("cupy", "rawmodule")
            ),
            "rawmodule_faster_than_cupy": bool(raw_ms and cupy_ms and raw_ms < cupy_ms)
                if "cupy" in results and "rawmodule" in results else None,
        },
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = Path(args.plot) if args.plot else path.with_suffix(".png")
    try:
        # 使用 Agg 后端，适合 Windows 终端和无桌面环境，直接生成 PNG。
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels, values, colors = [], [], []
        for key, label, color in (("numpy", "CPU NumPy", "#7f8c8d"),
                                  ("cupy", "CuPy", "#3498db"),
                                  ("rawmodule", "CUDA C RawModule", "#e67e22")):
            value = results.get(key, {}).get("mean_ms")
            if value is not None:
                labels.append(label); values.append(value); colors.append(color)
        if values:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            bars = ax.bar(labels, values, color=colors)
            ax.set_ylabel("Mean latency (ms)")
            ax.set_title("EnNeuro three-way CUDA benchmark")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value,
                        f"{value:.3f} ms", ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            report["plot"] = str(plot_path)
    except Exception as exc:
        report["plot_error"] = repr(exc)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["acceptance"]["numerical_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
