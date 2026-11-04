# -*- coding: utf-8 -*-
"""CUDA 基础算子 vs EnNeuro(CPU / GPU) 等价过程 耗时对比

对比三方：
  1. CUDA(dll)      —— code/eneuro/cuda/cu/*.cu 编译出的算子动态库（每次调用都做
                       cudaMalloc → H2D → kernel → D2H → cudaFree）
  2. EnNeuro(CPU)   —— eneuro.base.Tensor 跑在 numpy 后端
  3. EnNeuro(GPU)   —— eneuro.base.Tensor 跑在 cupy 后端（数据常驻显存，不含拷贝）

用法:
    python code/tests/bench_cu_vs_eneuro.py
    python code/tests/bench_cu_vs_eneuro.py --sizes 100000,1000000,10000000
    python code/tests/bench_cu_vs_eneuro.py --iters 30 --rounds 7 --csv bench.csv

指标说明:
    - 单位毫秒(ms)，取 rounds 轮中每轮平均耗时最小的那一轮（min-of-mean），
      以降低调度抖动影响。
    - "加速比" = 对方耗时 / CUDA 耗时，>1 表示 CUDA 更快。
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import os
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # .../code
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIB_DIR = ROOT / "eneuro" / "cuda" / ("dll" if os.name == "nt" else "so")
EXT = "dll" if os.name == "nt" else "so"

FP = ctypes.POINTER(ctypes.c_float)
F = ctypes.c_float
I = ctypes.c_int

ALPHA = 1.7
BETA = -0.35

# ----------------------------------------------------------------------------
# 后端可用性探测
# ----------------------------------------------------------------------------
cp: Any = None
HAS_CUPY = False
GPU_NAME = "-"
try:
    import cupy as _cupy

    cp = _cupy
    HAS_CUPY = True
    try:
        GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        GPU_NAME = "unknown"
except Exception as _exc:                            # pragma: no cover
    print(f"[warn] cupy 不可用，将跳过 GPU 对比: {_exc}", file=sys.stderr)

# 注意：下面这些符号统一先声明为 Any，避免“未安装时回退为 None”触发静态类型误报
Tensor: Any = None
as_Tensor: Any = None
ene_relu: Any = None
HAS_ENEURO = False
ENEURO_ERR = None
try:
    from eneuro.base import Tensor as _Tensor, as_Tensor as _as_Tensor
    from eneuro.base.functions import relu as _ene_relu

    Tensor, as_Tensor, ene_relu = _Tensor, _as_Tensor, _ene_relu
    HAS_ENEURO = True
except Exception as _exc:                            # pragma: no cover
    ENEURO_ERR = _exc
    print(f"[warn] eneuro 导入失败，将跳过框架对比: {_exc}", file=sys.stderr)


# ----------------------------------------------------------------------------
# CUDA 算子库
# ----------------------------------------------------------------------------
CUDA_SIGS = {
    "vector_add": {"launch_add": ([FP, FP, FP, I], None)},
    "vector_sub": {"launch_sub": ([FP, FP, FP, I], None)},
    "vector_mul": {"launch_mul": ([FP, FP, FP, I], None)},
    "vector_div": {"launch_div": ([FP, FP, FP, I], None)},
    "vector_maximum": {"launch_maximum": ([FP, FP, FP, I], None)},
    "vector_minimum": {"launch_minimum": ([FP, FP, FP, I], None)},
    "vector_neg": {"launch_neg": ([FP, FP, I], None)},
    "vector_relu": {"launch_relu": ([FP, FP, I], None)},
    "vector_scale": {"launch_scale": ([FP, FP, F, I], None)},
    "vector_add_scalar": {"launch_add_scalar": ([FP, FP, F, I], None)},
    "axpy": {"launch_axpy": ([FP, FP, F, I], None)},
    "vector_sum": {"launch_sum": ([FP, I], F)},
    "vector_dot": {"launch_dot": ([FP, FP, I], F)},
}


def load_cuda_libs():
    libs, missing = {}, []
    for name, funcs in CUDA_SIGS.items():
        path = LIB_DIR / f"{name}.{EXT}"
        if not path.exists():
            missing.append(name)
            continue
        lib = ctypes.CDLL(str(path))
        for fname, (argtypes, restype) in funcs.items():
            fn = getattr(lib, fname)
            fn.argtypes = argtypes
            fn.restype = restype
        libs[name] = lib
    return libs, missing


# ----------------------------------------------------------------------------
# 计时
# ----------------------------------------------------------------------------
def bench_group(tasks, iters, rounds, warmup=5):
    """交错(round-robin)测量一组任务，返回 {key: (min_avg_ms, median_avg_ms)}

    tasks: [(key, fn, sync), ...]

    先对所有任务做 warmup（涵盖首次 CUDA 上下文初始化），再按轮次轮流测量。
    对比试验表明：若把同一个算子连续测一长串再换下一个，会引入约 2x 的
    位置/温度漂移（先测/后测的算子吃亏），交错测量可基本消除该偏差。
    """
    for _, fn, _ in tasks:
        for _ in range(warmup):
            fn()
    for _, _, sync in tasks:
        if sync:
            sync()

    samples = {key: [] for key, _, _ in tasks}
    for _ in range(rounds):
        for key, fn, sync in tasks:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            if sync:
                sync()
            samples[key].append((time.perf_counter() - t0) * 1000.0 / iters)
    return {k: (min(v), statistics.median(v)) for k, v in samples.items()}


# ----------------------------------------------------------------------------
# 表格输出（按东亚字符宽度对齐）
# ----------------------------------------------------------------------------
def _dwidth(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text, width, align="l"):
    gap = max(0, width - _dwidth(text))
    return text + " " * gap if align == "l" else " " * gap + text


def print_table(header, rows):
    widths = [_dwidth(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _dwidth(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("|" + "|".join(f" {_pad(h, widths[i])} " for i, h in enumerate(header)) + "|")
    print(sep)
    for row in rows:
        cells = [
            f" {_pad(c, widths[i], 'r' if i else 'l')} "
            for i, c in enumerate(row)
        ]
        print("|" + "|".join(cells) + "|")
    print(sep)


def fmt_ms(v):
    if v is None:
        return "-"
    return f"{v:.4f}" if v < 1 else f"{v:.3f}"


def fmt_ratio(base, other):
    """>1 表示 CUDA 比对方快"""
    if base is None or other is None or base <= 0:
        return "-"
    return f"{other / base:.2f}x"


# ----------------------------------------------------------------------------
# 单个规模下的对比
# ----------------------------------------------------------------------------
def run_size(n, iters, rounds, libs):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    b_pos = (np.abs(b) + 0.5).astype(np.float32)      # 避免除零
    out = np.zeros_like(a)

    pa = a.ctypes.data_as(FP)
    pb = b.ctypes.data_as(FP)
    pb_pos = b_pos.ctypes.data_as(FP)
    pout = out.ctypes.data_as(FP)

    has_cuda = bool(libs)
    cuda_it = {k: v for k, v in libs.items()}         # 局部别名，闭包更快

    # ---- CPU 侧 ----
    ta: Any = None
    tb: Any = None
    tb_pos: Any = None
    if HAS_ENEURO:
        ta = as_Tensor(a)
        tb = as_Tensor(b)
        tb_pos = as_Tensor(b_pos)

    # ---- GPU 侧 ----
    ga: Any = None
    gb: Any = None
    gb_pos: Any = None
    na: Any = None
    nb: Any = None
    if HAS_ENEURO and HAS_CUPY:
        na = cp.asarray(a)
        nb = cp.asarray(b)
        ga = Tensor(a, device="cuda")
        gb = Tensor(b, device="cuda")
        gb_pos = Tensor(b_pos, device="cuda")
    sync_gpu = (lambda: cp.cuda.Stream.null.synchronize()) if HAS_CUPY else None

    def cuda(name: str) -> Any:
        """取已加载的算子库；未加载则返回 None（闭包内已做非 None 判断）"""
        return cuda_it.get(name) if has_cuda else None

    # 每个条目: (名称, cuda_fn, cpu_fn, gpu_fn)；设备不支持时置 None
    ops = []

    def add_op(name, cuda_fn, cpu_fn, gpu_fn):
        ops.append((name, cuda_fn, cpu_fn, gpu_fn))

    add_op(
        "add",
        (lambda: cuda("vector_add").launch_add(pa, pb, pout, n)) if cuda("vector_add") else None,
        (lambda: ta + tb) if ta is not None else None,
        (lambda: ga + gb) if ga is not None else None,
    )
    add_op(
        "sub",
        (lambda: cuda("vector_sub").launch_sub(pa, pb, pout, n)) if cuda("vector_sub") else None,
        (lambda: ta - tb) if ta is not None else None,
        (lambda: ga - gb) if ga is not None else None,
    )
    add_op(
        "mul",
        (lambda: cuda("vector_mul").launch_mul(pa, pb, pout, n)) if cuda("vector_mul") else None,
        (lambda: ta * tb) if ta is not None else None,
        (lambda: ga * gb) if ga is not None else None,
    )
    add_op(
        "div",
        (lambda: cuda("vector_div").launch_div(pa, pb_pos, pout, n)) if cuda("vector_div") else None,
        (lambda: ta / tb_pos) if ta is not None else None,
        (lambda: ga / gb_pos) if ga is not None else None,
    )
    add_op(
        "neg",
        (lambda: cuda("vector_neg").launch_neg(pa, pout, n)) if cuda("vector_neg") else None,
        (lambda: -ta) if ta is not None else None,
        (lambda: -ga) if ga is not None else None,
    )
    add_op(
        "scale(a*α)",
        (lambda: cuda("vector_scale").launch_scale(pa, pout, F(ALPHA), n)) if cuda("vector_scale") else None,
        (lambda: ta * ALPHA) if ta is not None else None,
        (lambda: ga * ALPHA) if ga is not None else None,
    )
    add_op(
        "add_scalar(a+β)",
        (lambda: cuda("vector_add_scalar").launch_add_scalar(pa, pout, F(BETA), n)) if cuda("vector_add_scalar") else None,
        (lambda: ta + BETA) if ta is not None else None,
        (lambda: ga + BETA) if ga is not None else None,
    )
    add_op(
        "axpy(αx+y)",
        (lambda: cuda("axpy").launch_axpy(pa, pout, F(ALPHA), n)) if cuda("axpy") else None,
        (lambda: ta * ALPHA + tb) if ta is not None else None,
        (lambda: ga * ALPHA + gb) if ga is not None else None,
    )
    add_op(
        "relu",
        (lambda: cuda("vector_relu").launch_relu(pa, pout, n)) if cuda("vector_relu") else None,
        (lambda: ene_relu(ta)) if HAS_ENEURO else None,
        (lambda: ene_relu(ga)) if HAS_ENEURO and ga is not None else None,
    )
    add_op(
        "maximum",
        (lambda: cuda("vector_maximum").launch_maximum(pa, pb, pout, n)) if cuda("vector_maximum") else None,
        (lambda: Tensor(np.maximum(ta.data, tb.data))) if ta is not None else None,
        (lambda: Tensor(cp.maximum(na, nb), device="cuda")) if na is not None else None,
    )
    add_op(
        "minimum",
        (lambda: cuda("vector_minimum").launch_minimum(pa, pb, pout, n)) if cuda("vector_minimum") else None,
        (lambda: Tensor(np.minimum(ta.data, tb.data))) if ta is not None else None,
        (lambda: Tensor(cp.minimum(na, nb), device="cuda")) if na is not None else None,
    )
    add_op(
        "sum(归约)",
        (lambda: cuda("vector_sum").launch_sum(pa, n)) if cuda("vector_sum") else None,
        (lambda: ta.sum()) if ta is not None else None,
        (lambda: ga.sum()) if ga is not None else None,
    )
    add_op(
        "dot(归约)",
        (lambda: cuda("vector_dot").launch_dot(pa, pb, n)) if cuda("vector_dot") else None,
        (lambda: (ta * tb).sum()) if ta is not None else None,
        (lambda: (ga * gb).sum()) if ga is not None else None,
    )

    # ---- 交错测量：把全部 (算子 × 后端) 放进同一个 round-robin 队列 ----
    tasks = []
    for name, c_fn, p_fn, g_fn in ops:
        if c_fn is not None:
            tasks.append((f"{name}\tcuda", c_fn, None))
        if p_fn is not None:
            tasks.append((f"{name}\tcpu", p_fn, None))
        if g_fn is not None:
            tasks.append((f"{name}\tgpu", g_fn, sync_gpu))

    # 纯拷贝基线（cupy H2D + D2H），用于解释 CUDA 耗时构成
    xfer_key = None
    if HAS_CUPY:
        xfer_key = "__transfer__\tgpu"
        tasks.append((xfer_key, lambda: cp.asnumpy(cp.asarray(a)), sync_gpu))

    res = bench_group(tasks, iters, rounds)

    def pick(name, backend):
        hit = res.get(f"{name}\t{backend}")
        return hit[0] if hit else None

    rows, csv_rows = [], []
    for name, c_fn, p_fn, g_fn in ops:
        c_ms = pick(name, "cuda")
        p_ms = pick(name, "cpu")
        g_ms = pick(name, "gpu")

        rows.append([
            name,
            fmt_ms(c_ms),
            fmt_ms(p_ms),
            fmt_ms(g_ms),
            fmt_ratio(c_ms, p_ms),
            fmt_ratio(c_ms, g_ms),
        ])
        csv_rows.append({"n": n, "op": name, "cuda_ms": c_ms, "cpu_ms": p_ms, "gpu_ms": g_ms})

    xfer_ms = res[xfer_key][0] if xfer_key else None
    if xfer_ms is not None:
        rows.append(["传输基线(H2D+D2H)", "-", "-", fmt_ms(xfer_ms), "-", "-"])
        csv_rows.append({"n": n, "op": "__transfer__", "cuda_ms": None,
                         "cpu_ms": None, "gpu_ms": xfer_ms})

    return rows, csv_rows, xfer_ms


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CUDA 算子 vs EnNeuro(CPU/GPU) 耗时对比")
    ap.add_argument("--sizes", default="100000,1000000,10000000",
                    help="向量长度列表，逗号分隔 (默认 100000,1000000,10000000)")
    ap.add_argument("--iters", type=int, default=20, help="每轮迭代次数 (默认 20)")
    ap.add_argument("--rounds", type=int, default=7, help="重复轮数 (默认 7)")
    ap.add_argument("--csv", default=None, help="将结果另存为 CSV")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    libs, missing = load_cuda_libs()

    print("=" * 100)
    print("CUDA 算子 vs EnNeuro(CPU/GPU) 耗时对比")
    print("=" * 100)
    print(f"numpy      : {np.__version__}")
    print(f"cupy       : {cp.__version__ if HAS_CUPY else 'N/A'}")
    print(f"GPU        : {GPU_NAME}")
    print(f"EnNeuro    : {'可用' if HAS_ENEURO else f'不可用 ({ENEURO_ERR})'}")
    print(f"CUDA 算子库: {LIB_DIR}  (已加载 {len(libs)}/{len(CUDA_SIGS)})")
    if missing:
        print(f"  缺失: {', '.join(missing)}  -> 请先运行 code\\eneuro\\cuda\\build.bat")
    print()

    all_csv = []
    for n in sizes:
        # 控制单轮总元素量，避免大数组上跑太久
        iters = max(3, min(args.iters, max(3, 20_000_000 // n)))
        print(f"\nn = {n:,}   (iters={iters}, rounds={args.rounds}, 单位: ms)")
        rows, csv_rows, xfer_ms = run_size(n, iters, args.rounds, libs)
        print_table(
            ["算子", "CUDA(dll)", "CPU(EnNeuro)", "GPU(EnNeuro)", "CPU/CUDA", "GPU/CUDA"],
            rows,
        )
        all_csv.extend(csv_rows)

        # 直观结论：CUDA(dll) 耗时里有多少是数据搬运
        if xfer_ms:
            cuda_times = [r["cuda_ms"] for r in csv_rows if r["cuda_ms"] is not None]
            if cuda_times:
                avg_cuda = statistics.mean(cuda_times)
                bw = (2 * n * 4) / (xfer_ms / 1000.0) / 1e9       # 往返 H2D+D2H
                print(f"  → CUDA(dll) 均值 {avg_cuda:.3f} ms；纯拷贝基线 {xfer_ms:.3f} ms"
                      f"（占 {xfer_ms / avg_cuda * 100:.0f}%），拷贝带宽 ≈ {bw:.1f} GB/s")
            cpu_best = min((r for r in csv_rows if r["cpu_ms"] is not None),
                           key=lambda r: r["cpu_ms"], default=None)
            gpu_times = [r["gpu_ms"] for r in csv_rows if r["gpu_ms"] is not None]
            if cpu_best and gpu_times:
                print(f"  → 最快算子: CPU {cpu_best['op']} {cpu_best['cpu_ms']:.4f} ms / "
                      f"GPU {min(gpu_times):.4f} ms，而 CUDA(dll) 最好的也要 {min(cuda_times):.4f} ms")

    print()
    print("说明:")
    print("  1. CUDA(dll) 每次调用都包含 cudaMalloc/cudaFree + H2D + kernel + D2H，")
    print("     当前 .cu 是「朴素实现」，小数组时分配与拷贝开销占主导；")
    print("     EnNeuro(GPU/cupy) 的数据常驻显存，不计拷贝，故并不完全对等。")
    print("  2. 「传输基线(H2D+D2H)」给出同规模数据一次往返拷贝的耗时，")
    print("     CUDA(dll) 耗时 ≈ 传输基线 + kernel，可据此判断是否值得改成常驻显存版本。")
    print("  3. maximum/minimum 在 EnNeuro 中无对应算子，按 Tensor 的 data 层实现。")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=["n", "op", "cuda_ms", "cpu_ms", "gpu_ms"])
            writer.writeheader()
            writer.writerows(all_csv)
        print(f"\n结果已写入 {args.csv}")


if __name__ == "__main__":
    main()
