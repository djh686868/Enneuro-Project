# -*- coding: utf-8 -*-
"""同步 vs 异步、串行 vs 流水线 —— 效果对比 + 柱状图

两组互相独立的对照：

  实验 A：传输方式对带宽的影响（纯传输，不含计算）
      ① 同步 / pageable          逐次同步
      ② 同步 / pinned            逐次同步
      ③ 异步 / pageable          逐次同步（含主机侧 staging 中转）
      ④ 异步 / pinned            逐次同步
      ⑤ 异步 / pinned            背靠背发送，只同步一次（异步的最优用法口径）

  实验 B：流水线对端到端的影响（拷贝 + 计算，6 × 8 MB）
      · 串行（单流 + 同步传输）
      · 流水线（双缓冲 + 2 流，pinned 源）
      · 流水线（双缓冲 + 2 流，pageable 源 —— 预期负优化）

与其它测试脚本一样，**在导入 eneuro 前屏蔽 cupy**，跑通即证明不依赖 cupy。

运行：
    python code/tests/bench_dev_memory_pipeline.py
    python code/tests/bench_dev_memory_pipeline.py --out doc/CUDAC/cu_async_pipeline_bar.png
"""
from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ModuleNotFoundError(f"{name} 被测试脚本屏蔽（验证不依赖 cupy）")
        return None


sys.meta_path.insert(0, _BlockCupy())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eneuro.utils.cuda_ops import CudaOps  # noqa: E402
from eneuro.utils.dev_memory import DeviceAllocator  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_SYNC = "#d9534f"
C_ASYNC = "#5cb85c"
C_CUPY = "#4a90d9"


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


def bench(fn, iters: int, rounds: int = 3, warmup: int = 1) -> float:
    """返回最快一轮的平均毫秒数"""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0 / iters)
    return best


# ---------------------------------------------------------------------------
# 实验 A：传输方式
# ---------------------------------------------------------------------------
def experiment_transfer(iters: int) -> list[tuple[str, float, float]]:
    """返回 [(名称, 单次耗时 ms, 带宽 GB/s)]"""
    n = 4_000_000                                   # 16 MB，超出 L2 以免假象
    nbytes = n * 4
    host = np.random.randn(n).astype(np.float32)

    out: list[tuple[str, float, float]] = []
    with DeviceAllocator(name="xfer") as dev:
        d = dev.alloc_like(host, name="d")
        s = dev.new_stream()
        pin = dev.pinned(n, name="pin")
        pin.array[:] = host

        cases = [
            ("① 同步 / pageable", lambda: d.upload(host), C_SYNC),
            ("② 同步 / pinned", lambda: d.upload(pin.array), C_SYNC),
            ("③ 异步 / pageable（含 staging）",
             lambda: (d.upload_async(host, stream=s), dev.sync(s)), C_ASYNC),
            ("④ 异步 / pinned（逐次同步）",
             lambda: (d.upload_async(pin.array, stream=s), dev.sync(s)), C_ASYNC),
            ("⑤ 异步 / pinned（背靠背）",
             lambda: (d.upload_async(pin.array, stream=s), dev.sync(s)), C_ASYNC),
        ]
        # ⑤ 的口径不同：连续发 4 次才同步一次，这里手工折算
        back_to_back = 4

        for name, fn, _color in cases[:-1]:
            ms = bench(fn, iters=iters)
            out.append((name, ms, nbytes / (ms / 1000) / 1e9))

        def five() -> None:
            for _ in range(back_to_back):
                d.upload_async(pin.array, stream=s)
            dev.sync(s)

        ms = bench(five, iters=max(2, iters // 2), warmup=1) / back_to_back
        out.append((cases[-1][0], ms, nbytes / (ms / 1000) / 1e9))
        dev.destroy_stream(s)
    return out


# ---------------------------------------------------------------------------
# 实验 B：端到端
# ---------------------------------------------------------------------------
def experiment_end_to_end(ops: CudaOps) -> tuple[list[tuple[str, float]], int, int]:
    """返回 ([(名称, 总耗时 ms)], chunk 字节数, 每 chunk kernel 次数)"""
    chunks, n = 6, 2_000_000
    nbytes = n * 4
    rng = np.random.default_rng(7)
    data = [rng.standard_normal(n).astype(np.float32) for _ in range(chunks)]
    results: list[tuple[str, float]] = []

    with DeviceAllocator(name="e2e") as dev:
        db = dev.alloc(n, name="accum")
        db.upload(data[0])

        # ---- 标定：让每 chunk 的计算耗时约等于拷贝耗时 ----
        s = dev.new_stream()
        # 必须先预热：首次 launch 会加载 kernel 模块（约 20 ms），
        # 不预热的话这个一次性开销会被平均进后面的测量，得到假数据
        ops.dev("add", db, db, db, stream=s)
        dev.sync(s)

        t0 = time.perf_counter()
        db.upload(data[0], stream=s)
        dev.sync(s)
        t_copy = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(100):
            ops.dev("add", db, db, db, stream=s)
        dev.sync(s)
        t_kernel = (time.perf_counter() - t0) / 100
        reps = max(1, int(round(t_copy / max(t_kernel, 1e-9))))
        dev.destroy_stream(s)
        print(f"    标定: 拷贝 {t_copy * 1000:.3f} ms/chunk，kernel {t_kernel * 1e6:.1f} us "
              f"-> 每 chunk 跑 {reps} 次 kernel")

        pins = [dev.pinned(n, name=f"chunk{i}") for i in range(chunks)]
        for pin, chunk in zip(pins, data):
            pin.array[:] = chunk

        def run_serial() -> None:
            src = dev.alloc(n, name="serial-src")
            ss = dev.new_stream()
            for pin in pins:
                src.upload(pin.array, stream=ss)      # 同步传输
                for _ in range(reps):
                    ops.dev("add", src, db, db, stream=ss)
            dev.sync(ss)
            dev.destroy_stream(ss)
            src.release()

        def run_pipeline(pinned: bool) -> None:
            with dev.double_buffer(n, slots=2, name="pipe") as pipe:
                for i in range(chunks):
                    slot = pipe.next()
                    slot.upload_async(pins[i].array if pinned else data[i])
                    for _ in range(reps):
                        ops.dev("add", slot.buffer, db, db, stream=slot.compute_stream)
                    slot.done()
                pipe.sync()

        results.append(("串行（单流 + 同步传输）", bench(run_serial, iters=1, rounds=3, warmup=1)))
        results.append(("流水线（pinned 源）", bench(lambda: run_pipeline(True), 1, 3, 1)))
        results.append(("流水线（pageable 源）", bench(lambda: run_pipeline(False), 1, 3, 1)))

        db.release()
        for pin in pins:
            pin.release()
    return results, nbytes, reps


# ---------------------------------------------------------------------------
# 柱状图
# ---------------------------------------------------------------------------
def draw(xfer, e2e, out_path: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.5))

    # ---- 面板1：传输带宽 ----
    names = [r[0] for r in xfer]
    short = ["① 同步\npageable", "② 同步\npinned", "③ 异步\npageable",
             "④ 异步\npinned", "⑤ 异步 pinned\n背靠背"]
    bw = [r[2] for r in xfer]
    colors = [C_SYNC, C_SYNC, C_ASYNC, C_ASYNC, C_ASYNC]
    bars = ax1.bar(range(len(names)), bw, 0.62, color=colors)
    bars[-1].set_hatch("//")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(short, fontsize=9)
    ax1.set_ylabel("带宽 (GB/s)")
    ax1.set_title("实验 A：传输方式对带宽的影响\n(16 MB 单次 H2D，越高越好)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(bw) * 1.22)
    for rect, v in zip(bars, bw):
        ax1.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    base_bw = bw[0]
    ax1.axhline(base_bw, color="#888", ls="--", lw=1)
    ax1.annotate(f"基线 {base_bw:.2f}", (len(names) - 0.55, base_bw), fontsize=8,
                 color="#666", ha="right", va="bottom")
    ax1.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=C_SYNC, label="同步（逐次等完成）"),
        plt.Rectangle((0, 0), 1, 1, color=C_ASYNC, label="异步（逐次等完成）"),
        plt.Rectangle((0, 0), 1, 1, color=C_ASYNC, hatch="//",
                      label="异步·背靠背（只同步一次）"),
    ], fontsize=8, loc="lower right")

    # ---- 面板2：端到端耗时 ----
    names2 = [r[0] for r in e2e]
    times = [r[1] for r in e2e]
    colors2 = [C_SYNC, C_ASYNC, "#f0ad4e"]
    bars2 = ax2.bar(range(len(names2)), times, 0.55, color=colors2)
    ax2.set_xticks(range(len(names2)))
    ax2.set_xticklabels([n.replace("（", "\n（") for n in names2], fontsize=9)
    ax2.set_ylabel("总耗时 (ms)")
    ax2.set_title("实验 B：端到端（6 × 8 MB 拷贝 + 等量计算）\n越低越好")
    ax2.grid(axis="y", alpha=0.3)
    for rect, v in zip(bars2, times):
        ax2.annotate(f"{v:.2f} ms\n({times[0] / v:.2f}x)",
                     (rect.get_x() + rect.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(times) * 1.22)

    # ---- 面板3：加速比汇总 ----
    labels = [
        "传输: 同步pinned\n/ 同步pageable",
        "传输: 异步pinned\n(背靠背)/ 同步pageable",
        "端到端: 流水线\n(pinned) / 串行",
        "端到端: 流水线\n(pageable) / 串行",
    ]
    ratios = [
        xfer[1][2] / xfer[0][2],
        xfer[4][2] / xfer[0][2],
        e2e[0][1] / e2e[1][1],
        e2e[0][1] / e2e[2][1],
    ]
    colors3 = [C_ASYNC, C_ASYNC, C_ASYNC, "#f0ad4e"]
    bars3 = ax3.bar(range(len(labels)), ratios, 0.55, color=colors3)
    ax3.axhline(1.0, color="#666", ls="--", lw=1.2)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=8.5)
    ax3.set_ylabel("加速比 (x)，> 1 表示更快")
    ax3.set_title("效果汇总（相对各自基线）")
    ax3.grid(axis="y", alpha=0.3)
    for rect, v in zip(bars3, ratios):
        ax3.annotate(f"{v:.2f}x", (rect.get_x() + rect.get_width() / 2, v),
                     ha="center", va="bottom" if v >= 1 else "top",
                     fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(max(ratios) * 1.2, 1.3))

    fig.suptitle("CUDA 显存层：同步 vs 异步、串行 vs 流水线   |   RTX 4060 Laptop",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n柱状图已保存: {out_path}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="同步/异步、串行/流水线 对比")
    ap.add_argument("--iters", type=int, default=8, help="实验 A 每轮迭代次数")
    ap.add_argument("--out", default=str((ROOT.parent / "doc" / "CUDAC"
                                          / "cu_async_pipeline_bar.png").resolve()))
    args = ap.parse_args()

    print("=" * 92)
    print("同步 vs 异步、串行 vs 流水线")
    print("=" * 92)
    ops = CudaOps()

    print("\n[实验 A] 传输方式对带宽的影响（16 MB 单次 H2D）")
    xfer = experiment_transfer(args.iters)
    print_table(["传输方式", "单次 (ms)", "带宽 (GB/s)", "相对基线"], [
        [n, f"{ms:.3f}", f"{bw:.2f}", f"{bw / xfer[0][2]:.2f}x"]
        for n, ms, bw in xfer
    ])

    print("\n[实验 B] 端到端（6 × 8 MB 拷贝 + 等量计算）")
    e2e, nbytes, reps = experiment_end_to_end(ops)
    print_table(["方式", "总耗时 (ms)", "相对串行"], [
        [n, f"{t:.2f}", f"{e2e[0][1] / t:.2f}x"] for n, t in e2e
    ])

    draw(xfer, e2e, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
