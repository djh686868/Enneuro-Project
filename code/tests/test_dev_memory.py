# -*- coding: utf-8 -*-
"""方案A 阶段1 验收测试：自管理显存（直通分配）+ 现有 13 个 `*_dev` 算子

**本脚本在导入任何 eneuro 模块之前屏蔽了 cupy**，
所以它能跑通本身就证明了 NFR-6（内存层不依赖 cupy）。

覆盖：
    [1] 分配 / 释放 / 对齐 / 上传下载往返
    [2] 错误路径（双重释放、释放后使用、OOM、设备越界、参数非法、已关闭）
    [3] 13 个算子全流程（纯 numpy 输入）
    [4] 统计交叉验证（Python 计数 vs C 计数）与泄漏报告
    [5] 无 CUDA 环境的主机回退（FR-10.1）

运行：
    python code/tests/test_dev_memory.py
"""
from __future__ import annotations

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
    """让 `import cupy` 抛 ImportError，用于验证内存层不依赖 cupy"""

    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ModuleNotFoundError(f"{name} 被测试脚本屏蔽（验证不依赖 cupy）")
        return None


sys.meta_path.insert(0, _BlockCupy())

from eneuro.utils.cuda_ops import OPS, CudaOps                      # noqa: E402
from eneuro.utils.dev_memory import (                               # noqa: E402
    AllocatorClosedError,
    BufferReleasedError,
    DeviceAllocator,
    DeviceError,
    DevMemError,
    InvalidArgumentError,
    OutOfMemoryError,
)


def _assert_cupy_blocked() -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return
    raise AssertionError("cupy 未被屏蔽，本测试的前提不成立")


# ---------------------------------------------------------------------------
# 输出工具
# ---------------------------------------------------------------------------
_FAILED: list[str] = []


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
    mark = "PASS" if ok else "FAIL"
    if not ok:
        _FAILED.append(label)
    print(f"    [{mark}] {_pad(label, 34)} {detail}")
    return ok


def expect_raises(label: str, exc_type, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return check(label, True, f"{type(exc).__name__}: {str(exc)[:56]}")
    except Exception as exc:                                   # noqa: BLE001
        return check(label, False, f"抛出了意料外的 {type(exc).__name__}: {exc}")
    return check(label, False, f"没有抛出 {exc_type.__name__}")


# ---------------------------------------------------------------------------
# [1] 基础功能
# ---------------------------------------------------------------------------
def test_basic(mem: DeviceAllocator) -> None:
    print("\n[1] 分配 / 释放 / 对齐 / 往返")
    check("cupy 已被屏蔽", _cupy_blocked(), "import cupy -> ImportError")

    buf = mem.alloc(1024, name="basic")
    check("元素数正确", buf.size == 1024, f"size={buf.size}")
    check("dtype 默认 float32", buf.dtype == np.float32, f"{buf.dtype}")
    check("字节数正确", buf.nbytes == 4096, f"nbytes={buf.nbytes}")
    check("256B 对齐 (FR-1.3)", buf.ptr % 256 == 0, f"ptr=0x{buf.ptr:x}")

    src = np.arange(1024, dtype=np.float32)
    dst = np.zeros(1024, dtype=np.float32)
    buf.upload(src)
    buf.download(dst)
    check("上传/下载往返一致", np.array_equal(src, dst), f"max|diff|={np.max(np.abs(src - dst)):.1e}")

    buf.release()
    check("释放后标记为 released", buf.released, f"released={buf.released}")

    # 上下文管理器自动回收
    with mem.alloc(16, name="ctx") as tmp:
        tmp_ptr = tmp.ptr
    check("with 退出后自动释放", tmp.released, f"被释放 ptr=0x{tmp_ptr:x}")


def _cupy_blocked() -> bool:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return True
    return False


# ---------------------------------------------------------------------------
# [2] 错误路径
# ---------------------------------------------------------------------------
def test_errors(mem: DeviceAllocator) -> None:
    print("\n[2] 错误路径")

    buf = mem.alloc(64, name="err")
    buf.release()
    expect_raises("双重释放 (FR-4.4)", BufferReleasedError, buf.release)
    expect_raises("释放后取指针", BufferReleasedError, lambda: buf.ptr)
    expect_raises("释放后上传", BufferReleasedError, buf.upload, np.zeros(64, np.float32))

    expect_raises("alloc(0) 非法参数", InvalidArgumentError, mem.alloc, 0)
    expect_raises("alloc(-1) 非法参数", InvalidArgumentError, mem.alloc, -1)

    live = mem.alloc(64, name="chk")
    expect_raises("dtype 不匹配", InvalidArgumentError,
                  live.upload, np.zeros(64, np.float64))
    expect_raises("字节数不匹配", InvalidArgumentError,
                  live.upload, np.zeros(32, np.float32))
    big = np.zeros((64, 2), dtype=np.float32)
    expect_raises("非连续数组", InvalidArgumentError, live.upload, big[:, 0])
    expect_raises("传入非 ndarray", InvalidArgumentError, live.upload, [1, 2, 3])
    live.release()

    expect_raises("设备号越界 (FR-8.2)", DeviceError, DeviceAllocator, 99)
    expect_raises("显存不足 (FR-1.2)", OutOfMemoryError, mem.alloc_bytes, 1 << 50)

    closed = DeviceAllocator(name="closed")
    closed.close()
    expect_raises("已关闭的分配器", AllocatorClosedError, closed.alloc, 16)
    check("重复 close 幂等", _close_twice_ok(closed), "no-op")


def _close_twice_ok(alloc: DeviceAllocator) -> bool:
    try:
        alloc.close()
        return True
    except Exception:                                          # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# [3] 13 个算子全流程（纯 numpy）
# ---------------------------------------------------------------------------
def test_all_ops(mem: DeviceAllocator, ops: CudaOps) -> None:
    n = 1_000_000
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    b_safe = (np.abs(b) + 0.5).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    scalar = np.zeros(1, dtype=np.float32)

    print(f"\n[3] 13 个算子全流程（纯 numpy，n={n:,}）")

    # ---- 一次分配 + 一次上传，之后反复算（这就是方案A 的正确用法）----
    with DeviceAllocator(name="ops") as dev:
        da = dev.alloc_like(a, name="a")
        db = dev.alloc_like(b, name="b")
        db_safe = dev.alloc_like(b_safe, name="b_safe")
        dy = dev.alloc_like(b, name="y")
        dout = dev.alloc_like(a, name="out")
        dscalar = dev.alloc(1, name="scalar")                  # sum/dot 的常驻结果缓冲

        da.upload(a)
        db.upload(b)
        db_safe.upload(b_safe)

        rows = []
        for spec in OPS:
            gb = db_safe if spec.safe_b else db
            np_b = b_safe if spec.safe_b else b

            if spec.kind == "binary":
                ops.dev(spec, da, gb, dout)
                dout.download(out)
                exp = spec.ref(np, a, np_b, spec.scalar)
                got, tol_kind = out, "abs"
            elif spec.kind in ("unary", "scalar"):
                ops.dev(spec, da, out=dout)
                dout.download(out)
                exp = spec.ref(np, a, np_b, spec.scalar)
                got, tol_kind = out, "abs"
            elif spec.kind == "axpy":
                dy.upload(b)                                   # 原地算子，每轮先还原 y
                ops.dev(spec, da, dy)
                dy.download(out)
                exp = spec.ref(np, a, b, spec.scalar)
                got, tol_kind = out, "abs"
            else:                                              # reduce1 / reduce2
                dscalar.upload(scalar)
                ops.dev(spec, da, gb, d_out=dscalar)
                dscalar.download(scalar)
                exp = np.float32(spec.ref(np, a, np_b, spec.scalar))
                got, tol_kind = np.float32(scalar[0]), "rel"

            if tol_kind == "abs":
                diff = float(np.max(np.abs(got - exp)))
                ok = diff < 1e-3
                shown = f"max|diff|={diff:.2e}"
            else:
                diff = abs(float(got) - float(exp)) / max(abs(float(exp)), 1e-6)
                ok = diff < 1e-5
                shown = f"相对误差={diff:.2e}"

            rows.append([spec.name, "PASS" if ok else "FAIL",
                         f"{spec.kind}", shown])
            if not ok:
                _FAILED.append(f"op:{spec.name}")

        print_table(["算子", "结果", "类型", "误差"], rows)
        passed = sum(1 for r in rows if r[1] == "PASS")
        check("13 个算子全部正确", passed == len(OPS), f"{passed}/{len(OPS)}")

        # ---- 性能：单次流程 vs host 接口；以及复用常驻缓冲的多次计算 ----
        def host_once() -> None:
            ops.host("add", a, b, out)

        def stage1_once() -> None:
            da.upload(a)
            db.upload(b)
            ops.dev("add", da, db, dout)
            dout.download(out)

        def kernel_only() -> None:
            ops.dev("add", da, db, dout)

        def _bench(fn, iters=10, warmup=2):
            for _ in range(warmup):
                fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) * 1000 / iters

        t_host = _bench(host_once)
        t_stage1 = _bench(stage1_once, iters=5)
        t_kernel = _bench(kernel_only, iters=50)

        print()
        print_table(["场景", "单次耗时 (ms)", "说明"], [
            ["host 接口（malloc + 3 次拷贝）", f"{t_host:.4f}", "每次都要进出显存"],
            ["方案A 单次（上传+算+下载）", f"{t_stage1:.4f}", "同样受 H2D/D2H 带宽限制"],
            ["方案A 复用常驻缓冲（只算）", f"{t_kernel:.4f}", "数据已在显存，只跑 kernel"],
        ])
        check("复用常驻缓冲显著快于 host", t_kernel < t_host / 20,
              f"{t_host / t_kernel:.0f}x")


# ---------------------------------------------------------------------------
# [4] 统计与泄漏
# ---------------------------------------------------------------------------
def test_stats_and_leak(mem: DeviceAllocator) -> None:
    print("\n[4] 统计与泄漏报告")

    before = mem.stats()
    for _ in range(10):
        with mem.alloc(1024, name="cycle"):
            pass
    after = mem.stats()

    check("分配计数 +10", after["alloc_calls"] - before["alloc_calls"] == 10,
          f"{before['alloc_calls']} -> {after['alloc_calls']}")
    check("释放计数 +10", after["free_calls"] - before["free_calls"] == 10,
          f"{before['free_calls']} -> {after['free_calls']}")
    check("无残留块", after["live_blocks"] == 0, f"live_blocks={after['live_blocks']}")

    # 阶段2 起分配器默认启用内存池：10 次 alloc/release 会被池复用，
    # 所以「alloc 与 cudaMalloc 一一对应」这条断言要用 pool=False 的分配器验证。
    if "native_alloc_calls" in after:
        d_native = after["native_alloc_calls"] - before["native_alloc_calls"]
        check("池复用后 cudaMalloc 增量远小于 10", d_native < 10,
              f"dC={d_native}（命中 {after['pool_hits']} 次）")

    with DeviceAllocator(name="no-pool", pool=False) as raw:
        c0 = raw.stats()["native_alloc_calls"]
        for _ in range(10):
            with raw.alloc(1024, name="cycle"):
                pass
        d_raw = raw.stats()["native_alloc_calls"] - c0
        check("pool=False: 10 次 alloc = 10 次 cudaMalloc", d_raw == 10, f"dC={d_raw}")

    check("泄漏报告为空", "无未释放块" in mem.leak_report(), mem.leak_report()[:40])

    # 注意：不保留返回的句柄时，对象会被 GC 并由 __del__ 自动归还，
    # 真正的泄漏来自「仍然持有引用」。这里故意持有。
    leaky = DeviceAllocator(name="leaky")
    kept = [leaky.alloc(4096, name="forgot-this"), leaky.alloc(2048, name="and-this")]
    report = leaky.leak_report()
    check("能检出泄漏 (FR-4.5)", "未释放块 2 个" in report, f"共 {len(leaky.live_blocks)} 块")
    print("      " + report.replace("\n", "\n      "))

    # 句柄被回收后泄漏应自动消失
    kept.clear()
    check("GC 后自动归还", len(leaky.live_blocks) == 0, f"live_blocks={len(leaky.live_blocks)}")
    leaky.close()
    check("close 后清空", len(leaky.live_blocks) == 0, "0 块")

    print(f"      统计快照: {after}")


# ---------------------------------------------------------------------------
# [5] 主机回退
# ---------------------------------------------------------------------------
def test_host_fallback() -> None:
    print("\n[5] 无 CUDA 环境回退（FR-10.1）")

    with DeviceAllocator(force_host=True, name="fallback") as host_mem:
        check("mode = host-fallback", host_mem.mode == "host-fallback", host_mem.mode)
        check("available = False", host_mem.available is False, str(host_mem.available))

        buf = host_mem.alloc(256, name="hbuf")
        src = np.arange(256, dtype=np.float32)
        buf.upload(src)
        got = buf.to_numpy()
        check("主机模式往返一致", np.array_equal(src, got), f"max|diff|={np.max(np.abs(src - got)):.1e}")
        check("主机模式无 CUDA 调用", "native_alloc_calls" not in host_mem.stats(),
              "未加载 dev_memory.dll")
        buf.release()

    # 库不存在时也应回退而不是崩
    missing = DeviceAllocator(lib_dir=ROOT / "cuda" / "not-exist", name="nolib")
    check("库缺失时回退不崩", missing.mode == "host-fallback", missing.mode)
    with missing.alloc(32, name="x") as b:
        b.upload(np.ones(32, dtype=np.float32))
    missing.close()


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 88)
    print("方案A 阶段1 验收：自管理显存（直通分配）+ 13 个 *_dev 算子")
    print("=" * 88)
    _assert_cupy_blocked()

    with DeviceAllocator(name="main") as mem:
        print(f"模式        : {mem.mode}")
        print(f"设备数      : {_device_count()}")
        print(f"cupy        : 已屏蔽（本测试不依赖 cupy）")

        ops = CudaOps()
        print(f"算子库      : {ops.lib_dir}  ({len(ops.ops)} 个算子)")

        test_basic(mem)
        test_errors(mem)
        test_all_ops(mem, ops)
        test_stats_and_leak(mem)

    test_host_fallback()

    print("\n" + "=" * 88)
    if _FAILED:
        print(f"失败 {len(_FAILED)} 项: {', '.join(_FAILED)}")
        return 1
    print("全部通过")
    return 0


def _device_count() -> int:
    from eneuro.utils.dev_memory import load_library
    lib = load_library()
    return int(lib.dll.ene_mem_device_count()) if lib else 0


if __name__ == "__main__":
    sys.exit(main())
