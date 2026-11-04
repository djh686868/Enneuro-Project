# -*- coding: utf-8 -*-
"""13 个 CUDA 算子：host 接口 vs dev 零拷贝接口 vs cupy 原生 —— 正确性 + 性能 + 柱状图

对应 `code/eneuro/cuda/cu/*.cu` 中的两套导出：
    launch_xxx(...)                host 版：主机指针，每次 malloc + H2D + kernel + D2H + free
    launch_xxx_dev(..., stream)    dev  版：只借 cupy 的显存指针，零拷贝（方案B）

用法：
    python code/tests/bench_cu_dev_ops.py
    python code/tests/bench_cu_dev_ops.py --n 4000000 --out doc/CUDAC/cu_dev_ops_bar.png
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eneuro.utils.cuda_ops import OPS, CudaOps, ELEMENTWISE_KINDS  # noqa: E402

# 中文字体（Windows 自带；缺失则自动退回默认字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 输出对齐工具
# ---------------------------------------------------------------------------
def _dwidth(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, align: str = "l") -> str:
    gap = max(0, width - _dwidth(text))
    return text + " " * gap if align == "l" else " " * gap + text


def print_table(header, rows) -> None:
    widths = [_dwidth(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _dwidth(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("|" + "|".join(f" {_pad(h, widths[i])} " for i, h in enumerate(header)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(
            f" {_pad(c, widths[i], 'r' if i else 'l')} " for i, c in enumerate(row)) + "|")
    print(sep)


# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------
class Data:
    def __init__(self, n: int):
        rng = np.random.default_rng(0)
        self.n = n
        self.a = rng.standard_normal(n).astype(np.float32)
        self.b = rng.standard_normal(n).astype(np.float32)
        self.b_safe = (np.abs(self.b) + 0.5).astype(np.float32)   # 除法用

        self.out_np = np.zeros(n, dtype=np.float32)

        self.ga = cp.asarray(self.a)
        self.gb = cp.asarray(self.b)
        self.gb_safe = cp.asarray(self.b_safe)
        self.out_gpu = cp.empty(n, dtype=cp.float32)
        self.d_out = cp.zeros(1, dtype=cp.float32)                # 归约结果的常驻缓冲

    def b_for(self, spec):
        return self.b_safe if spec.safe_b else self.b

    def gb_for(self, spec):
        return self.gb_safe if spec.safe_b else self.gb

    def np_b_for(self, spec):
        return self.b_safe if spec.safe_b else self.b


# ---------------------------------------------------------------------------
# 调用构造：host / dev / cupy 三种跑法
# ---------------------------------------------------------------------------
def make_host_call(ops: CudaOps, spec, data: Data):
    a, b, out = data.a, data.np_b_for(spec), data.out_np
    if spec.kind in ("binary",):
        return lambda: ops.host(spec, a, b, out)
    if spec.kind == "unary":
        return lambda: ops.host(spec, a, out=out)
    if spec.kind == "scalar":
        return lambda: ops.host(spec, a, out=out)
    if spec.kind == "axpy":
        y = data.b.copy()
        return lambda: ops.host(spec, a, y)
    if spec.kind == "reduce1":
        return lambda: ops.host(spec, a)
    return lambda: ops.host(spec, a, b)


def make_dev_call(ops: CudaOps, spec, data: Data, stream):
    ga, gb, out = data.ga, data.gb_for(spec), data.out_gpu
    if spec.kind == "binary":
        return lambda: ops.dev(spec, ga, gb, out, stream=stream)
    if spec.kind in ("unary", "scalar"):
        return lambda: ops.dev(spec, ga, out=out, stream=stream)
    if spec.kind == "axpy":
        gy = cp.empty_like(ga)
        return lambda: ops.dev(spec, ga, gy, stream=stream)
    if spec.kind == "reduce1":
        return lambda: ops.dev(spec, ga, d_out=data.d_out, stream=stream)
    return lambda: ops.dev(spec, ga, gb, d_out=data.d_out, stream=stream)


def make_cupy_call(ops: CudaOps, spec, data: Data):
    ga, gb = data.ga, data.gb_for(spec)
    if spec.kind == "axpy":
        gy = cp.empty_like(ga)
        return lambda: ops.cupy_ref(spec, ga, gy)
    return lambda: ops.cupy_ref(spec, ga, gb)


# ---------------------------------------------------------------------------
# 正确性
# ---------------------------------------------------------------------------
def check_correctness(ops: CudaOps, data: Data) -> list[tuple[str, bool, float, str]]:
    results = []
    for spec in OPS:
        # ---- host 版 vs numpy ----
        out = data.out_np.copy()
        b = data.np_b_for(spec)
        if spec.kind == "binary":
            ops.host(spec, data.a, b, out)
            got, exp = out, ops.numpy_ref(spec, data.a, b)
        elif spec.kind == "unary":
            ops.host(spec, data.a, out=out)
            got, exp = out, ops.numpy_ref(spec, data.a)
        elif spec.kind == "scalar":
            ops.host(spec, data.a, out=out)
            got, exp = out, ops.numpy_ref(spec, data.a)
        elif spec.kind == "axpy":
            y = data.b.copy()
            ops.host(spec, data.a, y)
            got, exp = y, ops.numpy_ref(spec, data.a, data.b)
        else:
            got = np.float32(ops.host(spec, data.a, b))
            exp = np.float32(ops.numpy_ref(spec, data.a, b))
        host_ok = bool(np.allclose(got, exp, rtol=1e-4, atol=1e-3, equal_nan=True))

        # ---- dev 版 vs cupy ----
        out_gpu = data.out_gpu
        out_gpu.fill(0)
        ga, gb = data.ga, data.gb_for(spec)
        if spec.kind == "binary":
            ops.dev(spec, ga, gb, out_gpu)
            got, exp = out_gpu, ops.cupy_ref(spec, ga, gb)
        elif spec.kind in ("unary", "scalar"):
            ops.dev(spec, ga, out=out_gpu)
            got, exp = out_gpu, ops.cupy_ref(spec, ga)
        elif spec.kind == "axpy":
            gy = cp.array(data.gb)
            ops.dev(spec, ga, gy)
            got, exp = gy, ops.cupy_ref(spec, ga, data.gb)
        else:
            ops.dev(spec, ga, gb, d_out=data.d_out)
            cp.cuda.Stream.null.synchronize()
            got = np.float32(data.d_out.item())
            exp = np.float32(ops.cupy_ref(spec, ga, gb).item())   # cupy 0-d 需 .item() 才能回主机
        cp.cuda.Stream.null.synchronize()
        if spec.kind in ELEMENTWISE_KINDS:
            dev_diff = float(cp.max(cp.abs(got - exp)))
            dev_ok = dev_diff < 1e-3                       # 逐元素：绝对误差
        else:
            # 归约：1e6 个 float32 累加顺序不同，绝对误差会有 ~1e-3 量级，只能看相对误差
            exp_f = float(exp)
            dev_diff = abs(float(got) - exp_f) / max(abs(exp_f), 1e-6)
            dev_ok = dev_diff < 1e-5                       # 相对误差

        results.append((spec.name, host_ok and dev_ok, dev_diff, spec.kind))
    return results


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------
def bench(ops: CudaOps, data: Data, iters: int, rounds: int):
    stream = cp.cuda.get_current_stream()
    sync = cp.cuda.Stream.null.synchronize

    cases = []          # (op_name, backend, fn, need_sync)
    for spec in OPS:
        # host 版内部就有 D2H + cudaDeviceSynchronize，天然同步；
        # dev / cupy 是异步的，必须在计时区间内同步，否则只量到 kernel 入队开销。
        cases.append((spec.name, "host", make_host_call(ops, spec, data), False))
        cases.append((spec.name, "dev", make_dev_call(ops, spec, data, stream), True))
        cases.append((spec.name, "cupy", make_cupy_call(ops, spec, data), True))

    # 统一预热（含首次上下文/模块加载）
    for _, _, fn, _ in cases:
        for _ in range(3):
            fn()
    sync()

    samples = {(op, be): [] for op, be, _, _ in cases}
    for _ in range(rounds):
        for op, be, fn, need_sync in cases:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            if need_sync:
                sync()
            samples[(op, be)].append((time.perf_counter() - t0) * 1000.0 / iters)

    return {k: min(v) for k, v in samples.items()}


# ---------------------------------------------------------------------------
# 柱状图
# ---------------------------------------------------------------------------
def draw_chart(times, out_path: Path, n: int, iters: int, rounds: int) -> None:
    names = [s.name for s in OPS]
    host = [times[(nm, "host")] for nm in names]
    dev = [times[(nm, "dev")] for nm in names]
    cupy = [times[(nm, "cupy")] for nm in names]

    x = np.arange(len(names))
    w = 0.27

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))

    # ---- 左：绝对耗时（对数轴，否则 dev 根本看不见）----
    b1 = ax1.bar(x - w, host, w, label="host 接口（malloc + 拷贝）", color="#d9534f")
    b2 = ax1.bar(x, dev, w, label="dev 接口（零拷贝，方案B）", color="#5cb85c")
    b3 = ax1.bar(x + w, cupy, w, label="cupy 原生", color="#4a90d9")
    ax1.set_yscale("log")
    ax1.set_ylabel("单次调用耗时 (ms, log)")
    ax1.set_title(f"各算子耗时对比  (n = {n:,}, min-of-{rounds}×{iters})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.3, which="both")
    ax1.legend(loc="upper right", fontsize=9)
    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax1.annotate(f"{h:.3g}", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="bottom", fontsize=6.5, rotation=90)

    # ---- 右：加速比 ----
    sp_host = [h / d for h, d in zip(host, dev)]
    sp_cupy = [c / d for c, d in zip(cupy, dev)]
    r1 = ax2.bar(x - w / 2, sp_host, w, label="host / dev（方案B 提速）", color="#5cb85c")
    r2 = ax2.bar(x + w / 2, sp_cupy, w, label="cupy / dev（vs cupy 原生）", color="#f0ad4e")
    ax2.axhline(1.0, color="#666", lw=1, ls="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("加速比 (x, log)")
    ax2.set_title("方案B 相对提速")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right")
    ax2.grid(axis="y", alpha=0.3, which="both")
    ax2.legend(loc="upper left", fontsize=9)
    for bars in (r1, r2):
        for rect in bars:
            h = rect.get_height()
            ax2.annotate(f"{h:.1f}x", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="bottom", fontsize=6.5)

    fig.suptitle("CUDA 算子：host 接口 vs dev 零拷贝接口（复用 cupy data.ptr）"
                 f"   |   GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n柱状图已保存: {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CUDA host vs dev 零拷贝接口 对比")
    ap.add_argument("--n", type=int, default=1_000_000, help="向量长度 (默认 1e6)")
    ap.add_argument("--iters", type=int, default=30, help="每轮迭代次数 (默认 30)")
    ap.add_argument("--rounds", type=int, default=5, help="重复轮数 (默认 5)")
    ap.add_argument("--out", default=str((ROOT.parent / "doc" / "CUDAC" / "cu_dev_ops_bar.png").resolve()),
                    help="柱状图输出路径")
    args = ap.parse_args()

    data = Data(args.n)
    ops = CudaOps()

    print("=" * 92)
    print("CUDA 算子 host 接口 vs dev 零拷贝接口（方案B）")
    print("=" * 92)
    print(f"GPU      : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"库目录   : {ops.lib_dir}  ({len(ops.ops)} 个算子)")
    print(f"n        : {args.n:,}\n")

    # ---- 1. 正确性 ----
    print("[1] 正确性（host vs numpy / dev vs cupy）")
    checks = check_correctness(ops, data)
    for name, ok, diff, kind in checks:
        label = "相对误差" if kind not in ELEMENTWISE_KINDS else "max|diff|"
        print(f"    [{'PASS' if ok else 'FAIL'}] {_pad(name, 12)} {label} = {diff:.3e}")
    passed = sum(1 for _, ok, _, _ in checks if ok)
    print(f"    -> {passed}/{len(OPS)} 通过" + ("" if passed == len(OPS) else "  （存在失败项）"))

    # ---- 2. 性能 ----
    print(f"\n[2] 性能（n={args.n:,}，交错测量 min-of-{args.rounds}，单位 ms）")
    times = bench(ops, data, args.iters, args.rounds)
    rows = []
    for spec in OPS:
        h = times[(spec.name, "host")]
        d = times[(spec.name, "dev")]
        c = times[(spec.name, "cupy")]
        rows.append([spec.name, f"{h:.4f}", f"{d:.4f}", f"{c:.4f}",
                     f"{h / d:.1f}x", f"{c / d:.2f}x"])
    print_table(["算子", "host", "dev(零拷贝)", "cupy", "host/dev", "cupy/dev"], rows)

    geo = statistics.geometric_mean([times[(s.name, "host")] / times[(s.name, "dev")] for s in OPS])
    print(f"\n    host/dev 提速 geometric mean = {geo:.1f}x "
          f"(min {min(times[(s.name, 'host')] / times[(s.name, 'dev')] for s in OPS):.1f}x, "
          f"max {max(times[(s.name, 'host')] / times[(s.name, 'dev')] for s in OPS):.1f}x)")

    # ---- 3. 柱状图 ----
    draw_chart(times, Path(args.out), args.n, args.iters, args.rounds)


if __name__ == "__main__":
    main()
