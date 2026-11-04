# -*- coding: utf-8 -*-
"""方案A 阶段5 验收：多设备与并发

覆盖 FR-7.1 / FR-7.3 / FR-8.1 / FR-8.2 / FR-8.3 / FR-9.2 / FR-11.1：

    [1] 线程级设备绑定（ene_mem_set_device / current_device / cached_device）
    [2] 跨设备误用必须报错（不隐式拷贝）
    [3] 池按分配器隔离 + 非法设备号报错
    [4] 跨流复用默认关闭：行为与阶段4 完全一致
    [5] 跨流复用开启：释放进守卫、复用时插事件等待、数据正确
    [6] 守卫清扫：流同步后块自动回池
    [7] debug 越界哨兵：越界写被抓出，损坏块不回池
    [8] debug 哨兵：块复用时重新写哨兵
    [9] 多线程并发分配：无泄漏、C 侧计数与 Python 一致
    [10] 不回归：13 算子、分配开销、稳态零 cudaMalloc

用法： python code/tests/test_dev_memory_multidev.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eneuro.utils.cuda_ops import CudaOps                       # noqa: E402
from eneuro.utils.dev_memory import (                           # noqa: E402
    DeviceAllocator, DeviceError, MemoryCorruptionError, load_library,
)

N = 4096
A = ((np.arange(N, dtype=np.float32) % 19) * 0.5 + 1.0)
B = ((np.arange(N, dtype=np.float32) % 11) * 0.25 + 0.5)

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"    [PASS] {name:<44} {detail}")
    else:
        FAIL += 1
        print(f"    [FAIL] {name:<44} {detail}")
    return bool(ok)


def group(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    print("=" * 92)
    print("方案A 阶段5 验收：多设备与并发")
    print("=" * 92)

    lib = load_library()
    if lib is None:
        print("\n[SKIP] 当前环境没有 dev_memory 动态库，无法验收")
        return 0
    device_count = int(lib.dll.ene_mem_device_count())
    print(f"可见 CUDA 设备数: {device_count}")

    ops = CudaOps()

    # ------------------------------------------------------------------
    group("[1] 线程级设备绑定（FR-8）")
    with DeviceAllocator(name="bind") as mem:
        check("allocator.device 记录正确", mem.device == 0, f"device={mem.device}")
        check("current_device() = 0", mem.current_device() == 0, str(mem.current_device()))
        check("cached_device() = 0", mem.cached_device() == 0, str(mem.cached_device()))

        # 幂等：重复绑定不报错、不改变状态
        for _ in range(3):
            mem._bind_device()
        check("重复 _bind_device 幂等", mem.cached_device() == mem.device)

        # 单卡机无法真的交错两张卡，但可以验证「每次都必须向 C 层确认」：
        # 若用 Python 侧的 _bound 做短路，另一个 allocator 把线程绑到别的卡后
        # 这里就会跳过绑定，退回静默分错卡。
        calls: list[int] = []
        real_set = mem._lib.dll.ene_mem_set_device
        mem._lib.dll.ene_mem_set_device = lambda d: (calls.append(int(d)), real_set(d))[1]
        try:
            mem._bind_device()
            mem._bind_device()
        finally:
            mem._lib.dll.ene_mem_set_device = real_set
        check("每次都向 C 层确认绑定（不用 Python 侧缓存短路）",
              calls == [mem.device, mem.device], str(calls))

        set_dev = lib.dll.ene_mem_set_device
        check("set_device(0) 返回 ENE_OK", int(set_dev(0)) == 0)
        check("set_device(-1) 非法", int(set_dev(-1)) != 0,
              f"code={int(set_dev(-1))}")
        bad = int(set_dev(device_count + 99))
        check("set_device(越界) 非法", bad != 0, f"code={bad}")
        check("失败后缓存设备未被污染", mem.cached_device() == mem.device,
              str(mem.cached_device()))

    # ------------------------------------------------------------------
    group("[2] 跨设备误用必须报错（FR-8.1 / FR-8.2）")
    with DeviceAllocator(name="xd") as mem:
        d = mem.alloc(N, name="x")
        d.upload(A)
        cur = mem.current_device()

        d.assert_current_device()                       # 同设备：不报错
        check("同设备不报错", True, f"current={cur}")

        real = d._info.device
        d._info.device = 1 if real != 1 else 0          # 伪造「块在另一张卡」
        try:
            d.assert_current_device(what="test")
            check("跨设备被拦住", False, "没有报错")
        except DeviceError as exc:
            msg = str(exc)
            check("跨设备被拦住", True, type(exc).__name__)
            check("错误信息含两个设备号", str(real) in msg and str(d._info.device) in msg,
                  msg[:60])

        # 算子下发路径同样要拦（_prepare_dev_args）
        try:
            ops.dev("add", d, d, d)
            check("ops.dev 也拦跨设备", False, "没有报错")
        except DeviceError as exc:
            check("ops.dev 也拦跨设备", True, type(exc).__name__)
        d._info.device = real

        # 关掉校验后应放行（默认开，这里只验证开关有效）
        with DeviceAllocator(name="xd2", strict_device=False) as lax:
            e = lax.alloc(N, name="e")
            e._info.device = 1 if real != 1 else 0
            try:
                e.assert_current_device()
                check("strict_device=False 放行", True)
            except DeviceError:
                check("strict_device=False 放行", False, "仍然报错")
            e._info.device = real

    # ------------------------------------------------------------------
    group("[3] 池隔离与非法设备号（FR-8.3）")
    if device_count < 2:
        try:
            DeviceAllocator(device=1, name="bad")
            check("超出设备数时创建报错", False, "没有报错")
        except DeviceError as exc:
            check("超出设备数时创建报错", True, type(exc).__name__)
            check("错误信息含错误码/设备号", "code=" in str(exc) or "1" in str(exc),
                  str(exc)[:52])
    else:                                              # pragma: no cover - 双卡机器
        print("    [SKIP] 本机有 2 张以上卡，留给真实双卡用例")

    with DeviceAllocator(name="pa") as m1, DeviceAllocator(name="pb") as m2:
        a1 = m1.alloc(N, name="a1")
        b1 = m2.alloc(N, name="b1")
        check("两个分配器各自独立分配", a1.ptr != b1.ptr,
              f"0x{a1.ptr:x} vs 0x{b1.ptr:x}")
        a1.release()
        check("归还只进入自己的池", len(m1.free_blocks) == 1 and len(m2.free_blocks) == 0,
              f"m1={len(m1.free_blocks)} m2={len(m2.free_blocks)}")
        b1.release()

    # ------------------------------------------------------------------
    group("[4] 跨流复用默认关闭（FR-7.3）")
    with DeviceAllocator(name="off") as mem:
        s = mem.new_stream()
        d = mem.alloc(N, name="d")
        d.upload(A)
        ops.dev("add", d, d, d, stream=s)
        check("默认不记录 last_stream", d._info.last_stream == 0,
              str(d._info.last_stream))
        d.release()
        check("默认释放即回池", len(mem.free_blocks) == 1)
        check("默认不产生跨流守卫",
              mem.stats()["stream_guarded_blocks"] == 0
              and mem.stats()["stream_guards"] == 0)
        mem.sync(s)
        mem.destroy_stream(s)

    # ------------------------------------------------------------------
    group("[5] 跨流复用开启（FR-7.1 / FR-7.3）")
    out_np = np.empty(N, dtype=np.float32)
    with DeviceAllocator(name="on", cross_stream_reuse=True) as mem:
        s1, s2 = mem.new_stream(), mem.new_stream()

        x = mem.alloc(N, name="x", stream=s1)
        y = mem.alloc(N, name="y", stream=s1)
        big = mem.alloc(1 << 24, name="big", stream=s1)  # 64 MB：单 kernel 约 0.9 ms
        x.upload(A)
        y.upload(B)

        ops.dev("mul", x, x, x, stream=s1)              # x = A*A，排到 s1
        # 把 s1 真正占住：小 kernel 受 L2（24 MB）加速，单次只要 2 us，
        # 入队反而比执行慢，队列会被排空 —— 那样守卫事件会在 alloc 前的清扫里
        # 已完成，走另一条同样正确的路径。64 MB 的 kernel 受显存带宽限制，
        # 8 个就够让事件在 alloc 时仍处于未完成状态。
        for _ in range(8):
            ops.dev("mul", big, big, big, stream=s1)
        check("记录了最后使用的流", x._info.last_stream == s1,
              f"0x{x._info.last_stream:x}")
        px = x.ptr
        x.release()

        st = mem.stats()
        check("释放后进入跨流守卫", st["stream_guarded_blocks"] == 1,
              f"guarded={st['stream_guarded_blocks']}")
        check("守卫中的块不在空闲池", len(mem.free_blocks) == 0,
              f"free={len(mem.free_blocks)}")

        # 在另一条流上申请同样大小 —— 应复用同一块，并插入事件等待
        z = mem.alloc(N, name="z", stream=s2)
        check("复用到同一块显存", z.ptr == px, f"0x{z.ptr:x} vs 0x{px:x}")
        check("流仍在跑时插入事件等待（主机未阻塞）",
              mem.stats()["cross_stream_waits"] == 1,
              str(mem.stats()["cross_stream_waits"]))
        check("守卫已清空", mem.stats()["stream_guarded_blocks"] == 0)

        # 正确性：s2 上读到的必须是 s1 上算出来的 A*A
        ops.dev("add_scalar", z, out=z, scalar=1.0, stream=s2)
        mem.sync(s2)
        z.download(out_np)
        expect = A * A + 1.0
        err = float(np.max(np.abs(out_np - expect)))
        check("跨流复用后数据正确", err == 0.0, f"max|diff|={err:.1e}")

        y.release()
        z.release()
        big.release()
        mem.sync(s1)
        mem.sync(s2)
        mem.destroy_stream(s1)
        mem.destroy_stream(s2)

    # ------------------------------------------------------------------
    group("[6] 守卫清扫（流跑完后块自动回池）")
    with DeviceAllocator(name="sweep", cross_stream_reuse=True) as mem:
        s = mem.new_stream()
        d = mem.alloc(N, name="d", stream=s)
        d.upload(A)
        ops.dev("add", d, d, d, stream=s)
        d.release()
        check("先进入守卫", mem.stats()["stream_guarded_blocks"] == 1)
        mem.sync(s)                                    # sync 内部会清扫
        check("sync 后回池", len(mem.free_blocks) == 1,
              f"free={len(mem.free_blocks)}")
        check("守卫清空", mem.stats()["stream_guarded_blocks"] == 0)

        # ---- 另一条清扫路径：alloc() 开头（不调 sync）----
        d2 = mem.alloc(N, name="d2", stream=s)          # 复用刚回池的块
        ops.dev("add", d2, d2, d2, stream=s)
        d2.release()
        check("再次进入守卫", mem.stats()["stream_guarded_blocks"] == 1)
        guard_ev = mem._stream_guard[0].guard_event
        mem.event_sync(guard_ev)                        # 只等事件，不触发清扫
        check("事件已完成但守卫仍在（等待本身不清扫）",
              mem.stats()["stream_guarded_blocks"] == 1)
        other = mem.alloc(64, name="other")             # 不同桶 → 走不到 _take_guarded
        check("alloc 前会把已完成的守卫块放回池",
              mem.stats()["stream_guarded_blocks"] == 0 and len(mem.free_blocks) == 1,
              f"guarded={mem.stats()['stream_guarded_blocks']} "
              f"free={len(mem.free_blocks)}")
        other.release()
        mem.destroy_stream(s)

    # ------------------------------------------------------------------
    group("[7] debug 越界哨兵（FR-9.2）")
    with DeviceAllocator(name="dbg", debug=True) as mem:
        d = mem.alloc(N, name="d")
        check("用户指针仍是 256B 对齐", d.ptr % 256 == 0, f"0x{d.ptr:x}")
        check("物理占用 = 桶容量 + 2x256",
              d._info.physical == d.capacity + 512,
              f"{d._info.physical} vs {d.capacity}+512")
        check("base_ptr 在用户指针之前",
              d._info.base_ptr == d.ptr - 256,
              f"0x{d._info.base_ptr:x} vs 0x{d.ptr:x}")

        d.upload(A)
        d.download(out_np)
        check("debug 模式下读写正常", float(np.max(np.abs(out_np - A))) == 0.0)
        d.release()
        check("正常释放不报错", True)
        check("未产生误报", mem.stats()["guard_failures"] == 0)

        # 尾部越界写
        d2 = mem.alloc(N, name="d2")
        mem._raw_copy(d2._info.base_ptr + 256 + d2._info.capacity, b"\x00" * 16, up=True)
        try:
            d2.release()
            check("尾部越界被抓住", False, "没有报错")
        except MemoryCorruptionError as exc:
            check("尾部越界被抓住", True, type(exc).__name__)
            check("错误信息指出尾部", "尾部" in str(exc), str(exc)[:44])
        check("损坏块不回池", len(mem.free_blocks) == 0,
              f"free={len(mem.free_blocks)}")
        check("guard_failures 计数 = 1", mem.stats()["guard_failures"] == 1)

        # 头部越界写
        d3 = mem.alloc(N, name="d3")
        mem._raw_copy(d3._info.base_ptr, b"\x01" * 16, up=True)
        try:
            d3.release()
            check("头部越界被抓住", False, "没有报错")
        except MemoryCorruptionError as exc:
            check("头部越界被抓住", "头部" in str(exc), str(exc)[:44])
        check("guard_failures 计数 = 2", mem.stats()["guard_failures"] == 2)

    # ------------------------------------------------------------------
    group("[8] debug 哨兵：块复用时重新写入")
    with DeviceAllocator(name="dbg2", debug=True) as mem:
        d = mem.alloc(N, name="d")
        info = d._info
        d.release()                                     # 正常回池
        check("正常块回到池", len(mem.free_blocks) == 1)
        # 直接破坏池中块的哨兵（模拟上一任使用者越界，且在 release 后被写脏）
        mem._raw_copy(info.base_ptr + 256 + info.capacity, b"\x7f" * 16, up=True)
        d2 = mem.alloc(N, name="d2")                    # 取回同一块
        check("复用到同一块", d2.ptr == info.ptr, f"0x{d2.ptr:x}")
        try:
            d2.release()
            check("复用时哨兵被重写", True)
        except MemoryCorruptionError as exc:
            check("复用时哨兵被重写", False, str(exc)[:50])

    # ------------------------------------------------------------------
    group("[9] 多线程并发分配（FR-11.1 / FR-11.3）")
    with DeviceAllocator(name="mt") as mem:
        warm = mem.alloc(1024, name="warm")
        warm.release()                                 # 先预热：池里已有 1 块
        before = mem.stats()
        errors: list[BaseException] = []
        rounds = 1500
        nthreads = 8

        def worker(k: int) -> None:
            try:
                for _ in range(rounds):
                    b = mem.alloc(1024, name=f"w{k}")
                    b.release()
                    _ = mem.stats()["free_blocks"]      # 顺带读统计
            except BaseException as exc:                # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(nthreads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.perf_counter() - t0
        after = mem.stats()

        check("并发无异常", not errors, str(errors[:1]))
        check("并发后无泄漏", len(mem.live_blocks) == 0,
              f"live={len(mem.live_blocks)}")
        check("Python 分配计数正确",
              after["alloc_calls"] - before["alloc_calls"] == nthreads * rounds,
              f"+{after['alloc_calls'] - before['alloc_calls']}")
        check("池吸收并发请求（native 增量 <= 线程数）",
              after["native_alloc_calls"] - before["native_alloc_calls"] <= nthreads,
              f"native +{after['native_alloc_calls'] - before['native_alloc_calls']}"
              f"（线程同时未命中时最多各 malloc 一次）")
        check("空闲块数有界收敛", 1 <= after["free_blocks"] <= nthreads,
              f"free={after['free_blocks']}")
        print(f"    {nthreads} 线程 x {rounds} 次 alloc/release: {dt * 1000:.1f} ms")

    # 原子计数的强验证：关掉池，让每次 alloc 都是一次真实 cudaMalloc，
    # 多线程并发下 C 侧计数必须**一个不少**（非原子实现会丢计数）
    with DeviceAllocator(name="mt-atomic", pool=False) as mem:
        before = mem.stats()
        errors = []
        rounds, nthreads = 300, 8

        def worker_raw(k: int) -> None:
            try:
                for _ in range(rounds):
                    b = mem.alloc(1024, name=f"r{k}")
                    b.release()
            except BaseException as exc:                # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker_raw, args=(k,)) for k in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        after = mem.stats()
        expect = nthreads * rounds
        check("直通模式并发无异常", not errors, str(errors[:1]))
        check("C 侧 alloc 计数一个不少（原子）",
              after["native_alloc_calls"] - before["native_alloc_calls"] == expect,
              f"+{after['native_alloc_calls'] - before['native_alloc_calls']} / 期望 {expect}")
        check("C 侧 free 计数一个不少（原子）",
              after["native_free_calls"] - before["native_free_calls"] == expect,
              f"+{after['native_free_calls'] - before['native_free_calls']} / 期望 {expect}")
        check("直通模式无泄漏", len(mem.live_blocks) == 0)

    # ------------------------------------------------------------------
    group("[10] 不回归：算子 / 开销 / 稳态零分配")
    with DeviceAllocator(name="reg") as mem:
        da, db, dout = mem.alloc_like(A, name="a"), mem.alloc_like(B, name="b"), \
            mem.alloc_like(A, name="out")
        da.upload(A)
        db.upload(B)
        ops.dev("add", da, db, dout)
        mem.sync()
        got = dout.to_numpy()
        check("13 算子仍可用（add）", float(np.max(np.abs(got - (A + B)))) == 0.0)

        dsum = mem.alloc(1, name="s")
        ops.dev("sum", da, out=da, d_out=dsum)
        mem.sync()
        got = float(dsum.to_numpy()[0])
        check("sum 常驻标量缓冲仍可用",
              abs(got - float(np.sum(A))) / abs(float(np.sum(A))) < 1e-5,
              f"{got:.3f} vs {np.sum(A):.3f}")

        # 热路径分配开销
        for _ in range(200):                            # 预热
            b = mem.alloc(N, name="h")
            b.release()
        cyc = 20000
        t0 = time.perf_counter()
        for _ in range(cyc):
            b = mem.alloc(N, name="h")
            b.release()
        us = (time.perf_counter() - t0) / cyc * 1e6
        check("分配开销 < 5 us（NFR-1）", us < 5.0, f"{us:.2f} us/次")

        before = mem.stats()["native_alloc_calls"]
        for _ in range(2000):
            b = mem.alloc(N, name="h")
            b.release()
        after = mem.stats()["native_alloc_calls"]
        check("稳态 cudaMalloc 增量为 0（NFR-3）", after == before,
              f"{before} -> {after}")

        da.release(); db.release(); dout.release(); dsum.release()

    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    if FAIL == 0:
        print(f"全部通过（{PASS} 项）")
    else:
        print(f"失败 {FAIL} 项 / 共 {PASS + FAIL} 项")
    print("=" * 92)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
