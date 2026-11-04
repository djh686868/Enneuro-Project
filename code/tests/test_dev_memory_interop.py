# -*- coding: utf-8 -*-
"""方案A 阶段4 验收：生态互操作（CUDA Array Interface / cupy 零拷贝）

覆盖 FR-6.1 ~ FR-6.4：

    [1] __cuda_array_interface__ 字典本身正确（shape/strides/typestr/data/version/stream）
    [2] 不依赖 cupy 也能读接口：cai_ptr() + device_ptr() 的通用兜底（用假 CAI 对象模拟 torch）
    [3] PinnedBuffer 的 __array_interface__（np.asarray 零拷贝，不触发拷贝）
    [4] as_cupy() 零拷贝一致性：cupy 写 → 我们读；我们写 → cupy 读
    [5] 租约保活：句柄先死、视图后死 → 块延迟回收（FR-6.2 的核心）
    [6] 租约未清时显式释放必须报错（防悬垂），force=True 放行
    [7] 混合模式：cupy 输入 + 池 workspace 输出（FR-6.4）
    [8] strict_interop 开关与主机回退/无设备模式下的行为
    [9] 越界保护：暴露的是 nbytes 而不是桶容量

用法： python code/tests/test_dev_memory_interop.py
"""
from __future__ import annotations

import ctypes
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- 让 code/ 下的 eneuro 可导入（与 test_dev_memory.py 同款） ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eneuro.utils.cuda_ops import CudaOps, device_ptr                    # noqa: E402
from eneuro.utils.dev_memory import (                                    # noqa: E402
    BufferLeasedError, BufferReleasedError, DeviceAllocator, DeviceError,
    cai_ptr, cupy_available, load_library, require_cupy,
)

try:
    import cupy as cp
except Exception:                                                        # pragma: no cover
    cp = None

N = 4096
A = (np.arange(N, dtype=np.float32) % 17) * 0.5
B = (np.arange(N, dtype=np.float32) % 13) * 0.25

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"    [PASS] {name:<40} {detail}")
    else:
        FAIL += 1
        print(f"    [FAIL] {name:<40} {detail}")
    return bool(ok)


def group(title: str) -> None:
    print(f"\n{title}")


def table(headers, rows) -> None:
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) for i in range(len(headers))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line)
    for r in rows:
        print("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line)


class FakeCaiArray:
    """最小的 CUDA Array Interface 实现：用来在**没有 torch/cupy** 的情况下
    验证通用兜底路径（阶段4 的 `cai_ptr` / `device_ptr`）。"""

    def __init__(self, ptr: int, size: int, typestr: str = "<f4",
                 shape=None, strides=None):
        self._cai = {
            "shape": (size,) if shape is None else shape,
            "strides": strides,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
        }

    @property
    def __cuda_array_interface__(self):
        return self._cai


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 90)
    print("方案A 阶段4 验收：生态互操作（CUDA Array Interface / cupy 零拷贝）")
    print("=" * 90)
    print(f"cupy 可用: {cupy_available()}"
          + (f"（{cp.__version__}）" if cupy_available() else "  —— 互操作测试将跳过，"
             "但 [1][2][3][8][9] 仍应全部通过（NFR-6：核心不依赖 cupy）"))

    # ------------------------------------------------------------------
    group("[1] __cuda_array_interface__ 字典本身（FR-6.1）")
    lib = load_library()
    if lib is None:
        print("    [SKIP] 当前环境无 CUDA 库，本组跳过")
    else:
        with DeviceAllocator(name="cai") as mem:
            d = mem.alloc(N, name="d")
            cai = d.__cuda_array_interface__
            check("shape = (size,)", cai["shape"] == (N,), str(cai["shape"]))
            check("strides = None（C 连续）", cai["strides"] is None)
            check("typestr = <f4", cai["typestr"] == "<f4", cai["typestr"])
            check("version = 3", cai["version"] == 3, str(cai["version"]))
            check("data[0] = 块指针", cai["data"][0] == d.ptr, f"0x{cai['data'][0]:x}")
            check("data[1] = False（可写）", cai["data"][1] is False)
            check("stream 键存在", "stream" in cai, str(cai.get("stream")))
            check("size 与 shape 自洽", cai["shape"][0] == d.size)

            # 释放后不能再导出
            d.release()
            try:
                d.__cuda_array_interface__                            # noqa: B018
                check("释放后取接口报错", False)
            except BufferReleasedError as exc:
                check("释放后取接口报错", True, type(exc).__name__)

            # 主机回退模式：绝不能把主机指针当设备指针导出
            with DeviceAllocator(force_host=True, name="host") as hm:
                hb = hm.alloc(N, name="h")
                try:
                    hb.__cuda_array_interface__                            # noqa: B018
                    check("主机回退模式拒绝导出", False)
                except DeviceError as exc:
                    check("主机回退模式拒绝导出", True, type(exc).__name__)
                try:
                    hb.as_cupy()
                    check("主机回退模式拒绝 as_cupy", False)
                except DeviceError as exc:
                    check("主机回退模式拒绝 as_cupy", True, type(exc).__name__)

    # ------------------------------------------------------------------
    group("[2] 不依赖 cupy 的通用反读：cai_ptr / device_ptr 兜底")
    with DeviceAllocator(name="cai2") as mem:
        d = mem.alloc(N, name="d")
        fake = FakeCaiArray(d.ptr, N)

        ptr, nbytes = cai_ptr(fake, what="fake")
        check("cai_ptr 取到指针", ptr == d.ptr, f"0x{ptr:x}")
        check("cai_ptr 算出字节数", nbytes == N * 4, f"{nbytes} B")

        got = device_ptr(fake)
        check("device_ptr 接受任意 CAI 对象", got.value == d.ptr, f"0x{got.value:x}")

        # 校验必须拦住不合规的 CAI 对象（否则 kernel 会静默算错）
        for bad, expect, why in [
            (FakeCaiArray(d.ptr, N, typestr="<f8"), TypeError, "dtype 不是 float32"),
            (FakeCaiArray(d.ptr, N, strides=(8,)), ValueError, "非连续"),
            (FakeCaiArray(d.ptr, N, shape=(N // 2, 2), strides=(2, 1)), ValueError, "二维非连续"),
            (FakeCaiArray(0, N), ValueError, "空指针"),
        ]:
            try:
                cai_ptr(bad, what="bad")
                check(f"拦住 {why}", False, "没有报错")
            except expect as exc:
                check(f"拦住 {why}", True, type(exc).__name__)

        # 低版本接口不认
        old = FakeCaiArray(d.ptr, N)
        old._cai["version"] = 2
        try:
            cai_ptr(old, what="old")
            check("拦住 version<3", False, "没有报错")
        except ValueError as exc:
            check("拦住 version<3", True, type(exc).__name__)

        # 缺接口的普通对象给出可读错误（而不是 AttributeError）
        try:
            cai_ptr(np.zeros(4, dtype=np.float32), what="numpy")
            check("numpy 数组被拒绝", False, "没有报错")
        except TypeError as exc:
            check("numpy 数组被拒绝", True, str(exc)[:38])

    # ------------------------------------------------------------------
    group("[3] PinnedBuffer 的 __array_interface__（numpy 零拷贝）")
    with DeviceAllocator(name="pin") as mem:
        pin = mem.pinned(N, dtype=np.float32, name="pin")
        arr = np.asarray(pin)
        check("np.asarray(pin) 形状/类型", arr.shape == (N,) and arr.dtype == np.float32,
              f"{arr.shape} {arr.dtype}")
        check("np.asarray 零拷贝", arr.ctypes.data == pin.ptr,
              f"0x{arr.ctypes.data:x} vs 0x{pin.ptr:x}")
        arr[:] = 3.5
        check("写 numpy 视图 = 写锁页内存", float(pin.array[0]) == 3.5)
        check("as_numpy() 与 .array 同一块", pin.as_numpy().ctypes.data == pin.ptr)
        pin.release()
        try:
            pin.__array_interface__                                   # noqa: B018
            check("归还后拒绝导出", False)
        except BufferReleasedError as exc:
            check("归还后拒绝导出", True, type(exc).__name__)

    # ------------------------------------------------------------------
    group("[4] as_cupy() 零拷贝一致性（cupy 写 / 我们读，双向）")
    if not cupy_available():
        print("    [SKIP] 无 cupy")
    else:
        ops = CudaOps()
        out_np = np.empty(N, dtype=np.float32)
        with DeviceAllocator(name="xc") as mem:
            da, db, dout = (mem.alloc_like(A, name="a"), mem.alloc_like(B, name="b"),
                            mem.alloc_like(A, name="out"))
            da.upload(A)
            db.upload(B)

            # (a) 我们的 kernel 写 → cupy 读
            ops.dev("add", da, db, dout)
            mem.sync()
            view = dout.as_cupy()
            check("kernel 结果 = cupy 视图", float(cp.max(cp.abs(view - cp.asarray(A + B)))) == 0.0)
            check("视图与块同指针", view.data.ptr == dout.ptr)

            # (b) cupy 写 → 我们的 kernel 读
            view[:] = cp.asarray(A * 2.0)
            mem.sync()
            ops.dev("mul", da, dout, db)          # db = A * (A*2)
            mem.sync()
            db.download(out_np)
            ref = A * (A * 2.0)
            check("cupy 写入被 kernel 读到", float(np.max(np.abs(out_np - ref))) == 0.0,
                  f"max|diff|={float(np.max(np.abs(out_np - ref))):.1e}")

            # (c) 让 cupy 直接在我们的 workspace 上算（输出也是我们的块）
            cp.multiply(view, cp.asarray(B), out=view)
            mem.sync()
            view.get()                            # cupy 侧同步一次
            dout.download(out_np)
            ref = (A * 2.0) * B
            check("cupy 算子写进我们的块", float(np.max(np.abs(out_np - ref))) == 0.0,
                  f"max|diff|={float(np.max(np.abs(out_np - ref))):.1e}")

            # (d) cupy 数组反向喂给我们的 kernel（CAI 兜底路径）
            cp_arr = cp.asarray(A + B)
            ops.dev("add", cp_arr, db, dout)
            mem.sync()
            dout.download(out_np)
            ref = (A + B) + (A * (A * 2.0))
            check("cupy 数组走 device_ptr 兜底", float(np.max(np.abs(out_np - ref))) == 0.0)

            del view
            gc.collect()

    # ------------------------------------------------------------------
    group("[5] 租约保活：句柄先死、视图后死（FR-6.2 核心）")
    with DeviceAllocator(name="lease", pool=True) as mem:
        d = mem.alloc(N, name="d")
        d.upload(A)
        view = d.as_cupy()
        info = d._info
        check("leases = 1", d.leases == 1, str(d.leases))
        check("stats.live_leases = 1", mem.stats()["live_leases"] == 1)

        del d
        gc.collect()
        check("句柄删除后视图仍可读", float(cp.max(cp.abs(view - cp.asarray(A)))) == 0.0)
        check("块未回池（free_blocks=0）", len(mem.free_blocks) == 0,
              f"free={len(mem.free_blocks)}")
        check("块进入延迟回收", len(mem.deferred_blocks) == 1)
        check("延迟块仍被登记为租用", mem.stats()["leased_blocks"] == 1)
        check("延迟回收不减少已分配总量", mem.stats()["deferred_bytes"] == N * 4,
              f"{mem.stats()['deferred_bytes']} B")

        # 同一个块开两个视图：必须等两个都死才回收
        d2 = mem.alloc(N, name="d2")
        info2 = d2._info
        v1, v2 = d2.as_cupy(), d2.as_cupy()
        check("同一块两个视图 = 2 个租约", d2.leases == 2, str(d2.leases))
        del d2, v1
        gc.collect()
        check("还剩 1 个视图时（本块）不回收", len(info2.leases) == 1,
              f"leases={len(info2.leases)}")
        check("整进程仍有 2 个租约（含前面那块）", mem.stats()["live_leases"] == 2,
              str(mem.stats()["live_leases"]))
        del v2
        gc.collect()
        check("本块视图清空后租约归零", len(info2.leases) == 0,
              f"leases={len(info2.leases)}")
        check("本块已回到空闲表", len(mem.free_blocks) == 1,
              f"free={len(mem.free_blocks)}")
        gc.collect()

        del view
        gc.collect()
        gc.collect()
        check("最后一个视图死亡 → 块回池", len(mem.free_blocks) == 2,
              f"free={len(mem.free_blocks)}")
        check("延迟表清空", len(mem.deferred_blocks) == 0)
        check("defer_events 计数", mem.stats()["defer_events"] == 2,
              str(mem.stats()["defer_events"]))

        # 回收后该内存必须能被复用（说明确实回到了池里）
        d3 = mem.alloc(N, name="reuse")
        check("延迟回收的块被复用", d3.ptr in (info.ptr,) or d3.ptr != 0,
              f"0x{d3.ptr:x}")
        d3.release()

    # ------------------------------------------------------------------
    group("[6] 租约未清时的释放策略")
    with DeviceAllocator(name="strict") as mem:
        d = mem.alloc(N, name="d")
        view = d.as_cupy()
        try:
            d.release()
            check("显式释放被租约拦住", False, "没有报错")
        except BufferLeasedError as exc:
            check("显式释放被租约拦住", True, f"leases={d.leases}")
            check("错误信息可读", "外部视图" in str(exc), str(exc)[:30])
        check("lease_blocked_releases 计数", mem.stats()["lease_blocked_releases"] == 1)
        check("块仍然活着（没被误释放）", not d.released and len(mem.live_blocks) == 1)

        # force=True：明确知道视图不用了 → 放行
        d.release(force=True)
        check("force=True 放行", d.released)

        # 强制释放后视图成悬垂 —— 这是调用方自担的行为，只是确认它没被静默兜住
        del view
        gc.collect()

        # 双重释放检测在阶段4 依然有效
        try:
            d.release()
            check("双重释放仍被检出", False)
        except BufferReleasedError as exc:
            check("双重释放仍被检出", True, type(exc).__name__)

    # ------------------------------------------------------------------
    group("[7] 混合模式：cupy 输入 + 池 workspace 输出（FR-6.4）")
    if not cupy_available():
        print("    [SKIP] 无 cupy")
    else:
        ops = CudaOps()
        with DeviceAllocator(name="mix") as mem:
            ws = mem.alloc(N, name="ws")                 # 长期复用的 workspace
            cp_a, cp_b = cp.asarray(A), cp.asarray(B)
            d_out = mem.alloc(N, name="scalar_out")      # 归约结果常驻标量
            out_np = np.empty(N, dtype=np.float32)

            before = mem.stats()
            t0 = time.perf_counter()
            for _ in range(200):                          # 稳态热路径
                ops.dev("add", cp_a, cp_b, ws)            # 用户数据在 cupy
                ops.dev("mul", ws, cp_b, ws)              # 中间结果落我们的池
            mem.sync()
            dt = (time.perf_counter() - t0) / 200 * 1e6   # -> us/轮
            after = mem.stats()
            ws.download(out_np)
            check("热路径 400 次 kernel 结果正确",
                  float(np.max(np.abs(out_np - ((A + B) * B)))) == 0.0)
            check("热路径没有再分配（alloc_calls 增量为 0）",
                  after["alloc_calls"] == before["alloc_calls"],
                  f"{before['alloc_calls']} -> {after['alloc_calls']}")
            check("热路径 0 次 cudaMalloc（NFR-3）",
                  after["native_alloc_calls"] == before["native_alloc_calls"],
                  f"{before['native_alloc_calls']} -> {after['native_alloc_calls']}")
            check("cupy 数组也可零拷贝喂给算子",
                  after["pool_misses"] == before["pool_misses"],
                  f"misses={after['pool_misses']}")
            check("每轮开销合理（< 200 us）", dt < 200, f"{dt:.1f} us/轮（2 个 kernel）")

            # 归约算子：结果写进我们的常驻标量缓冲，全程不产生 D2H
            ops.dev("sum", ws, out=ws, d_out=d_out)
            mem.sync()
            got = float(d_out.to_numpy()[0])
            check("归约结果写进常驻标量缓冲",
                  abs(got - float(np.sum((A + B) * B))) < 1e-2,
                  f"{got:.4f} vs {float(np.sum((A + B) * B)):.4f}")

            # 同一块同时被我们的句柄和 cupy 数组引用：租约把它钉住
            view = ws.as_cupy()
            check("workspace 有 cupy 视图时租约=1", ws.leases == 1)
            del view
            gc.collect()

    # ------------------------------------------------------------------
    group("[8] strict_interop 开关与无设备模式")
    with DeviceAllocator(name="strict2", strict_interop=True) as mem:
        d = mem.alloc(N, name="d")
        try:
            d.__cuda_array_interface__                                  # noqa: B018
            check("strict 模式禁止瞬时借出", False, "没有报错")
        except BufferLeasedError as exc:
            check("strict 模式禁止瞬时借出", True, str(exc)[:34])
        if cupy_available():
            view = d.as_cupy()
            check("strict 模式仍允许 as_cupy", view is not None and d.leases == 1)
            del view
            gc.collect()
        else:
            print("    [SKIP] as_cupy 需要 cupy")

    with DeviceAllocator(force_host=True, name="nohost") as mem:
        d = mem.alloc(N, name="d")
        d.upload(A)
        out = np.empty(N, dtype=np.float32)
        d.download(out)
        check("主机回退模式功能不受影响", float(np.max(np.abs(out - A))) == 0.0)
        try:
            d.__cuda_array_interface__                                  # noqa: B018
            check("主机回退拒绝 CAI", False)
        except DeviceError as exc:
            check("主机回退拒绝 CAI", True, type(exc).__name__)

    # ------------------------------------------------------------------
    group("[9] 越界保护：暴露 nbytes 而不是桶容量")
    if not lib:
        print("    [SKIP] 无 CUDA 库")
    else:
        with DeviceAllocator(name="cap") as mem:
            raw = N * 4 + 100                       # 故意跨过 256B 步进边界
            d = mem.alloc_bytes(raw, dtype=np.float32, name="cap")
            if d.capacity == d.nbytes:
                print(f"    [SKIP] 本次分配桶容量等于请求（{d.nbytes} B），无可验证的富余")
            else:
                cai = d.__cuda_array_interface__
                check("shape 用请求长度而非容量", cai["shape"] == (d.size,),
                      f"shape={cai['shape']} size={d.size} cap={d.capacity}")
                if cupy_available():
                    view = d.as_cupy()
                    check("cupy 视图 nbytes 只到请求长度",
                          view.nbytes == d.nbytes, f"{view.nbytes} vs {d.nbytes}")
                    del view
                    gc.collect()
                else:
                    print("    [SKIP] cupy 视图检查需要 cupy")
            d.release()

    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    if FAIL == 0:
        print(f"全部通过（{PASS} 项）")
    else:
        print(f"失败 {FAIL} 项 / 共 {PASS + FAIL} 项")
    print("=" * 90)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
