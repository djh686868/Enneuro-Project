# -*- coding: utf-8 -*-
"""方案A 阶段2 验收测试：size-class 内存池

与阶段1 测试一样，**在导入任何 eneuro 模块之前屏蔽 cupy**，
所以跑通即证明内存层不依赖 cupy（NFR-6）。

覆盖：
    [1] 池命中与指针复用
    [2] 稳态零 cudaMalloc（NFR-3）
    [3] 分配/归还开销（NFR-1）
    [4] size-class 分桶正确性
    [5] 大块按精确大小分桶
    [6] 复用后旧句柄必须失效（安全关键）
    [7] trim 归还驱动
    [8] pool_max_bytes 驱逐
    [9] pool=False 等价阶段1
    [10] 长时间稳定性 / 空闲块收敛（NFR-8）
    [11] 13 个算子跑在池化显存上
    [12] close 归还池中空闲块

运行：
    python code/tests/test_dev_memory_pool.py
    python code/tests/test_dev_memory_pool.py --stress 1000000
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


# ---------------------------------------------------------------------------
# 0. 屏蔽 cupy —— 必须早于 import eneuro
# ---------------------------------------------------------------------------
class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ModuleNotFoundError(f"{name} 被测试脚本屏蔽（验证不依赖 cupy）")
        return None


sys.meta_path.insert(0, _BlockCupy())

from eneuro.utils.cuda_ops import OPS, CudaOps                  # noqa: E402
from eneuro.utils.dev_memory import (                           # noqa: E402
    BufferReleasedError,
    DeviceAllocator,
    _size_class,
)

_FAILED: list[str] = []


# ---------------------------------------------------------------------------
# 输出工具
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


def check(label: str, ok: bool, detail: str = "") -> bool:
    if not ok:
        _FAILED.append(label)
    print(f"    [{'PASS' if ok else 'FAIL'}] {_pad(label, 36)} {detail}")
    return ok


def expect_raises(label: str, exc_type, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return check(label, True, f"{type(exc).__name__}: {str(exc)[:48]}")
    except Exception as exc:                                    # noqa: BLE001
        return check(label, False, f"意料外的 {type(exc).__name__}: {exc}")
    return check(label, False, f"没有抛出 {exc_type.__name__}")


def native_allocs(mem: DeviceAllocator) -> int:
    return mem.stats().get("native_alloc_calls", -1)


# ---------------------------------------------------------------------------
# [1] 池命中与指针复用
# ---------------------------------------------------------------------------
def test_hit(mem: DeviceAllocator) -> None:
    print("\n[1] 池命中与指针复用")

    c0 = native_allocs(mem)
    b1 = mem.alloc(1024, name="x")
    ptr1 = b1.ptr
    b1.release()
    b2 = mem.alloc(1024, name="x2")

    check("复用同一块显存", b2.ptr == ptr1, f"0x{ptr1:x} -> 0x{b2.ptr:x}")
    check("只发生过 1 次 cudaMalloc", native_allocs(mem) - c0 == 1,
          f"dC={native_allocs(mem) - c0}")
    b2.release()

    st = mem.stats()
    check("命中计数增加", st["pool_hits"] >= 1, f"hits={st['pool_hits']}")
    check("空闲块在池中", st["free_blocks"] >= 1, f"free_blocks={st['free_blocks']}")


# ---------------------------------------------------------------------------
# [2] 稳态零 cudaMalloc（NFR-3）
# ---------------------------------------------------------------------------
def test_steady_state(mem: DeviceAllocator, cycles: int) -> None:
    print(f"\n[2] 稳态零 cudaMalloc（NFR-3，{cycles:,} 次固定尺寸 alloc/release）")

    def run(k: int) -> None:
        for _ in range(k):
            b = mem.alloc(4096, name="steady")
            b.release()

    run(1)                                     # 首次必然 cudaMalloc
    c1 = native_allocs(mem)
    run(cycles - 1)
    delta = native_allocs(mem) - c1

    check("预热后 cudaMalloc 增量为 0", delta == 0, f"dC={delta}")
    st = mem.stats()
    check("命中率 > 99%", st["hit_rate"] > 0.99, f"hit_rate={st['hit_rate'] * 100:.2f}%")
    check("空闲块收敛", st["free_blocks"] <= 2, f"free_blocks={st['free_blocks']}")


# ---------------------------------------------------------------------------
# [3] 分配开销（NFR-1）
# ---------------------------------------------------------------------------
def test_overhead(mem: DeviceAllocator, cycles: int) -> None:
    print("\n[3] 分配/归还开销（NFR-1）")

    def loop(k: int) -> None:
        for _ in range(k):
            b = mem.alloc(4096, name="bench")
            b.release()

    loop(2000)                                  # 预热
    t0 = time.perf_counter()
    loop(cycles)
    per_us = (time.perf_counter() - t0) / cycles * 1e6

    # 对照：同样的循环，但关掉池（每次真 cudaMalloc + cudaFree）
    n_raw = max(50, cycles // 100)
    with DeviceAllocator(name="raw-bench", pool=False) as raw:
        def loop_raw(k: int) -> None:
            for _ in range(k):
                b = raw.alloc(4096, name="bench")
                b.release()

        loop_raw(5)
        t0 = time.perf_counter()
        loop_raw(n_raw)
        raw_us = (time.perf_counter() - t0) / n_raw * 1e6

    print(f"    池命中  一次「分配 + 归还」= {per_us:.2f} us"
          f"（NFR-1 的 <5us 针对分配单项，合计都达标则单项必然达标）")
    print(f"    对照 pool=False（真 cudaMalloc/cudaFree）= {raw_us:.2f} us"
          f"  ->  池提速 {raw_us / per_us:.0f}x")
    check("池循环开销 < 5 us", per_us < 5.0, f"{per_us:.2f} us")

    # ---- 分配开销 vs 块大小：drivers 的 caching allocator 对小块的缓存效果远好于大块 ----
    def pair_cost(sz: int, pooled: bool, n: int) -> float:
        with DeviceAllocator(name=f"{'pool' if pooled else 'raw'}{sz}",
                             pool=pooled) as m:
            def one() -> None:
                b = m.alloc_bytes(sz, name="cost")
                b.release()
            for _ in range(3):
                one()
            t0 = time.perf_counter()
            for _ in range(n):
                one()
            return (time.perf_counter() - t0) / n * 1e6

    rows = []
    for sz in (4096, 65536, 1 << 20, 4 << 20):
        raw_c = pair_cost(sz, False, 50)
        pool_c = pair_cost(sz, True, 2000)
        rows.append([f"{sz / 1024:,.0f} KB", f"{raw_c:.2f}", f"{pool_c:.2f}",
                     f"{raw_c / pool_c:.1f}x"])
    print_table(["块大小", "pool=False (us)", "池命中 (us)", "提速"], rows)
    print("    说明: 驱动的 caching allocator 对小块缓存效果很好（~2us），"
          "大块要贵两个数量级；池把两者都压到与大小无关的常数。")


# ---------------------------------------------------------------------------
# [4][5] size-class 分桶
# ---------------------------------------------------------------------------
def test_size_classes(mem: DeviceAllocator) -> None:
    print("\n[4] size-class 分桶正确性")

    cases = [
        (1, ("s", 256), 256),
        (256, ("s", 256), 256),
        (257, ("s", 512), 512),
        (1000, ("s", 1024), 1024),
        (4096, ("s", 4096), 4096),
        (4097, ("p", 8192), 8192),
        (65536, ("p", 65536), 65536),
        (1 << 20, ("p", 1 << 20), 1 << 20),
        ((1 << 20) + 1, ("e", (1 << 20) + 1), (1 << 20) + 1),
        (4_000_000, ("e", 4_000_000), 4_000_000),
    ]
    rows = []
    ok_all = True
    for nbytes, exp_key, exp_cap in cases:
        key, cap = _size_class(nbytes)
        buf = mem.alloc_bytes(nbytes, name="cls")
        ok = (key == exp_key and cap == exp_cap and buf.capacity == exp_cap)
        ok_all &= ok
        rows.append([f"{nbytes:,}", f"{key[0]}/{key[1]:,}", f"{exp_cap:,}",
                     f"{buf.capacity:,}", "PASS" if ok else "FAIL"])
        buf.release()
    print_table(["请求字节", "实际桶", "预期容量", "实际容量", "结果"], rows)
    check("全部分桶符合预期", ok_all, f"{len(cases)} 个尺寸")

    print("\n[5] 大块按精确大小分桶（不取整、不浪费）")
    big = mem.alloc_bytes(4_000_000, name="big")
    check("容量 == 请求字节", big.capacity == 4_000_000,
          f"cap={big.capacity:,} 请求=4,000,000")
    big.release()
    check("大块同样进池", mem.stats()["free_blocks"] >= 1,
          f"free_blocks={mem.stats()['free_blocks']}")


# ---------------------------------------------------------------------------
# [6] 复用后旧句柄必须失效（安全关键）
# ---------------------------------------------------------------------------
def test_handle_safety(mem: DeviceAllocator) -> None:
    print("\n[6] 复用后旧句柄必须失效（安全关键）")

    a = mem.alloc(2048, name="old")
    old_ptr = a.ptr
    a.release()
    b = mem.alloc(2048, name="new")

    check("新块复用了同一地址", b.ptr == old_ptr, f"0x{old_ptr:x}")

    # 核心断言：地址被复用后，旧句柄绝不能又「活过来」
    expect_raises("旧句柄取指针仍报错", BufferReleasedError, lambda: a.ptr)
    expect_raises("旧句柄再释放仍报错", BufferReleasedError, a.release)
    check("新句柄可正常使用", not b.released, f"ptr=0x{b.ptr:x}")

    # 数据隔离：新块写入的内容必须与旧数据无关
    src = np.arange(2048, dtype=np.float32)
    b.upload(src)
    got = b.to_numpy()
    check("复用块数据隔离正确", np.array_equal(src, got),
          f"max|diff|={np.max(np.abs(src - got)):.1e}")
    b.release()


# ---------------------------------------------------------------------------
# [7] trim
# ---------------------------------------------------------------------------
def test_trim(mem: DeviceAllocator) -> None:
    print("\n[7] trim 归还驱动（FR-2.5）")

    tmp = DeviceAllocator(name="trim")
    for size in (1024, 4096, 65536):
        b = tmp.alloc_bytes(size, name="t")
        b.release()

    st = tmp.stats()
    check("池中有空闲块", st["free_blocks"] == 3, f"free_blocks={st['free_blocks']}")
    n0 = native_allocs(tmp)

    out = tmp.trim(older_than=3600.0)          # 只归还「空闲 1 小时以上」的
    check("older_than 生效（不归还）", out["blocks"] == 0, f"trim -> {out}")
    check("空闲块仍在", tmp.stats()["free_blocks"] == 3, "")

    out = tmp.trim()                           # 全部归还
    st = tmp.stats()
    check("trim() 归还全部（3 块）", out["blocks"] == 3, f"trim -> {out}")
    check("池已清空", st["free_blocks"] == 0 and st["free_bytes"] == 0,
          f"free_blocks={st['free_blocks']}")
    check("trim 后 cudaFree 了 3 次", st["native_free_calls"] >= 3,
          f"native_free_calls={st['native_free_calls']}")
    check("trim 不影响 cudaMalloc 计数", native_allocs(tmp) == n0, "")
    tmp.close()


# ---------------------------------------------------------------------------
# [8] pool_max_bytes 驱逐
# ---------------------------------------------------------------------------
def test_eviction() -> None:
    print("\n[8] pool_max_bytes 驱逐（FR-2.6）")

    cap = 64 * 1024
    with DeviceAllocator(name="capped", pool_max_bytes=cap) as mem:
        c0 = native_allocs(mem)
        f0 = mem.stats()["native_free_calls"]
        # 必须**同时持有**多块再统一释放，池里才会真的堆积（逐个 alloc/release 会被直接复用）
        held = [mem.alloc_bytes(32 * 1024, name="big") for _ in range(10)]
        for buf in held:
            buf.release()
        st = mem.stats()
        check(f"空闲字节 ≤ 上限 {cap:,}", st["free_bytes"] <= cap,
              f"free_bytes={st['free_bytes']:,}")
        check("发生了 FIFO 驱逐", st["evicted_blocks"] > 0,
              f"evicted={st['evicted_blocks']} / 10")
        check("驱逐的块真的被 free", st["native_free_calls"] - f0 == st["evicted_blocks"],
              f"dFree={st['native_free_calls'] - f0}")
        check("cudaMalloc 只发生在首次", native_allocs(mem) - c0 == 10,
              f"dC={native_allocs(mem) - c0}（10 块首次都要 cudaMalloc）")


# ---------------------------------------------------------------------------
# [9] pool=False 等价阶段1
# ---------------------------------------------------------------------------
def test_pool_disabled() -> None:
    print("\n[9] pool=False 等价阶段1（每次直通）")

    with DeviceAllocator(name="raw", pool=False) as mem:
        c0 = native_allocs(mem)
        for _ in range(5):
            b = mem.alloc(1024, name="raw")
            b.release()
        st = mem.stats()
        check("每次都 cudaMalloc", native_allocs(mem) - c0 == 5,
              f"dC={native_allocs(mem) - c0}")
        check("池为空", st["free_blocks"] == 0, f"free_blocks={st['free_blocks']}")
        check("pool_enabled = False", st["pool_enabled"] is False, "")


# ---------------------------------------------------------------------------
# [10] 长时间稳定性（NFR-8）
# ---------------------------------------------------------------------------
def test_stability(cycles: int) -> None:
    print(f"\n[10] 长时间稳定性（NFR-8，{cycles:,} 次 alloc/release）")

    with DeviceAllocator(name="stress") as mem:
        def cycle(k: int) -> None:
            for _ in range(k):
                b = mem.alloc(1024, name="s")
                b.release()

        cycle(1)                                    # 首次必然 cudaMalloc
        c0 = native_allocs(mem)
        t0 = time.perf_counter()
        cycle(cycles - 1)
        elapsed = time.perf_counter() - t0
        st = mem.stats()
        check("无活跃块泄漏", st["live_blocks"] == 0, f"live_blocks={st['live_blocks']}")
        check("空闲块数收敛（不随次数增长）", st["free_blocks"] <= 2,
              f"free_blocks={st['free_blocks']}")
        check("预热后 cudaMalloc 增量为 0", native_allocs(mem) - c0 == 0,
              f"dC={native_allocs(mem) - c0}")
        print(f"    {cycles - 1:,} 次循环耗时 {elapsed:.2f} s "
              f"（{elapsed / (cycles - 1) * 1e6:.2f} us/次）")


# ---------------------------------------------------------------------------
# [11] 13 个算子跑在池化显存上
# ---------------------------------------------------------------------------
def test_all_ops(ops: CudaOps) -> None:
    n = 200_000
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    b_safe = (np.abs(b) + 0.5).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    scalar = np.zeros(1, dtype=np.float32)
    print(f"\n[11] 13 个算子跑在池化显存上（n={n:,}，每块 {a.nbytes:,} B）")

    with DeviceAllocator(name="ops") as mem:
        da = mem.alloc_like(a, name="a")
        db = mem.alloc_like(b, name="b")
        db_safe = mem.alloc_like(b_safe, name="b_safe")
        dy = mem.alloc_like(b, name="y")
        dout = mem.alloc_like(a, name="out")
        dscalar = mem.alloc(1, name="scalar")

        check("大张量也进了池", da._info.cls is not None,
              f"桶={da._info.cls}")

        da.upload(a)
        db.upload(b)
        db_safe.upload(b_safe)

        failed = []
        for spec in OPS:
            gb = db_safe if spec.safe_b else db
            np_b = b_safe if spec.safe_b else b
            if spec.kind == "binary":
                ops.dev(spec, da, gb, dout)
                dout.download(out)
                ok = np.allclose(out, spec.ref(np, a, np_b, spec.scalar), rtol=1e-5, atol=0)
            elif spec.kind in ("unary", "scalar"):
                ops.dev(spec, da, out=dout)
                dout.download(out)
                ok = np.allclose(out, spec.ref(np, a, np_b, spec.scalar), rtol=1e-5, atol=0)
            elif spec.kind == "axpy":
                dy.upload(b)
                ops.dev(spec, da, dy)
                dy.download(out)
                ok = np.allclose(out, spec.ref(np, a, b, spec.scalar), rtol=1e-4, atol=1e-4)
            else:
                dscalar.upload(scalar)
                ops.dev(spec, da, gb, d_out=dscalar)
                dscalar.download(scalar)
                exp = float(spec.ref(np, a, np_b, spec.scalar))
                ok = abs(float(scalar[0]) - exp) / max(abs(exp), 1e-6) < 1e-5
            if not ok:
                failed.append(spec.name)

        check("13 个算子全部正确", not failed, f"{len(OPS) - len(failed)}/{len(OPS)}")
        if failed:
            print(f"    失败: {', '.join(failed)}")

        # 先归还一块，再申请同尺寸 → 必须命中池（池里没有空闲块时当然要新分配）
        dout.release()
        c0 = native_allocs(mem)
        again = mem.alloc_like(a, name="reuse")
        check("归还后同尺寸再分配命中池", native_allocs(mem) == c0,
              f"dC={native_allocs(mem) - c0}")
        check("复用块容量与桶一致", again.capacity == dout.capacity,
              f"cap={again.capacity:,}")
        again.release()


# ---------------------------------------------------------------------------
# [12] close 归还池块
# ---------------------------------------------------------------------------
def test_close_frees_pool() -> None:
    print("\n[12] close 归还池中空闲块")

    mem = DeviceAllocator(name="closepool")
    b = mem.alloc(1024, name="p")
    b.release()
    st = mem.stats()
    check("释放后块留在池中", st["free_blocks"] == 1, f"free_blocks={st['free_blocks']}")
    n0 = st["native_free_calls"]

    mem.close()
    st = mem.stats()
    check("close 后池已清空", st["free_blocks"] == 0, f"free_blocks={st['free_blocks']}")
    check("close 真的归还了驱动", st["native_free_calls"] - n0 == 1,
          f"dFree={st['native_free_calls'] - n0}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="方案A 阶段2 验收：size-class 内存池")
    ap.add_argument("--steady", type=int, default=20_000, help="[2] 稳态循环次数")
    ap.add_argument("--overhead", type=int, default=200_000, help="[3] 开销测量次数")
    ap.add_argument("--stress", type=int, default=200_000, help="[10] 稳定性循环次数")
    args = ap.parse_args()

    print("=" * 90)
    print("方案A 阶段2 验收：size-class 内存池")
    print("=" * 90)

    with DeviceAllocator(name="main") as mem:
        print(f"模式   : {mem.mode}")
        print(f"池     : 启用（上限 无限制）")
        print(f"cupy   : 已屏蔽（本测试不依赖 cupy）")
        ops = CudaOps()
        print(f"算子库 : {ops.lib_dir}  ({len(ops.ops)} 个算子)")

        test_hit(mem)
        test_steady_state(mem, args.steady)
        test_overhead(mem, args.overhead)
        test_size_classes(mem)
        test_handle_safety(mem)
        test_trim(mem)

    test_eviction()
    test_pool_disabled()
    test_stability(args.stress)
    test_all_ops(ops)
    test_close_frees_pool()

    print("\n" + "=" * 90)
    if _FAILED:
        print(f"失败 {len(_FAILED)} 项: {', '.join(_FAILED)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
