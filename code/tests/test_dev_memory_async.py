# -*- coding: utf-8 -*-
"""方案A 阶段3 验收测试：pinned staging + 异步拷贝 + 事件 + 双缓冲

与阶段1/2 测试一样，**在导入任何 eneuro 模块之前屏蔽 cupy**，跑通即证明不依赖 cupy。

覆盖：
    [1] pinned 主机内存：分配 / 视图读写 / 池复用
    [2] 异步上传/下载正确性（pageable 源走 staging；pinned 源走直接 DMA）
    [3] 传输带宽：同步/异步 × pageable/pinned（NFR-4）
    [4] 大块自动分片（FR-3.3）
    [5] 事件：record / wait / elapsed / done
    [6] 在飞 pinned 有界（不会随调用次数增长）
    [7] download_async 目标必须锁页（否则报错而不是假装异步）
    [8] 双缓冲流水线：H2D 与 kernel 重叠（FR-3.4）
    [9] 13 个算子跑在自定义流上（回归）
    [10] 主机回退模式下异步 API 不崩

运行：
    python code/tests/test_dev_memory_async.py
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


class _BlockCupy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ModuleNotFoundError(f"{name} 被测试脚本屏蔽（验证不依赖 cupy）")
        return None


sys.meta_path.insert(0, _BlockCupy())

from eneuro.utils.cuda_ops import OPS, CudaOps                  # noqa: E402
from eneuro.utils.dev_memory import (                           # noqa: E402
    DeviceAllocator,
    DevMemError,
    InvalidArgumentError,
)

_FAILED: list[str] = []


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
        return check(label, True, f"{type(exc).__name__}: {str(exc)[:44]}")
    except Exception as exc:                                    # noqa: BLE001
        return check(label, False, f"意料外的 {type(exc).__name__}: {exc}")
    return check(label, False, f"没有抛出 {exc_type.__name__}")


def bandwidth_gbs(nbytes: int, seconds: float) -> float:
    return nbytes / seconds / 1e9 if seconds > 0 else 0.0


# ---------------------------------------------------------------------------
# [1] pinned 主机内存
# ---------------------------------------------------------------------------
def test_pinned(mem: DeviceAllocator) -> None:
    print("\n[1] pinned 主机内存")

    n = 4096
    pin = mem.pinned(n, name="p1")
    check("视图元素数正确", pin.array.size == n, f"{pin.array.size}")
    check("视图 dtype 正确", pin.array.dtype == np.float32, f"{pin.array.dtype}")
    check("视图可写", pin.array.flags.writeable, "")

    pin.array[:] = np.arange(n, dtype=np.float32)
    check("写入后能读回", pin.array[-1] == n - 1, f"last={pin.array[-1]}")
    check("指针非空", pin.ptr != 0, f"0x{pin.ptr:x}")

    pinned_before = mem.stats()["native_host_alloc_calls"]
    pin.release()
    check("归还后标记 released", pin.released, "")

    pin2 = mem.pinned(n, name="p2")
    after = mem.stats()
    check("归还后复用池中块", after["pinned_hits"] >= 1, f"hits={after['pinned_hits']}")
    check("没有新增 cudaHostAlloc", after["native_host_alloc_calls"] == pinned_before,
          f"dC={after['native_host_alloc_calls'] - pinned_before}")
    pin2.release()
    expect_raises("重复归还报错", DevMemError, pin2.release)
    check("旧句柄不会复活", pin is not pin2 and pin.released,
          f"pin.released={pin.released}")


# ---------------------------------------------------------------------------
# [2][3] 异步正确性
# ---------------------------------------------------------------------------
def test_async_transfer(mem: DeviceAllocator, ops: CudaOps) -> None:
    print("\n[2] 异步上传/下载正确性")

    n = 100_000
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)

    with DeviceAllocator(name="async") as dev:
        da = dev.alloc_like(a, name="a")
        db = dev.alloc_like(b, name="b")
        dout = dev.alloc_like(a, name="out")
        s = dev.new_stream()
        check("拿到非阻塞流", s != 0, f"stream=0x{s:x}")

        # pageable 源 -> 内部 pinned staging
        da.upload_async(a, stream=s)
        db.upload_async(b, stream=s)
        ops.dev("add", da, db, dout, stream=s)
        dev.sync(s)
        dout.download(out)
        check("pageable 源异步上传正确", np.allclose(out, a + b, atol=0, rtol=0),
              f"max|diff|={np.max(np.abs(out - (a + b))):.1e}")

        # pinned 源 -> 直接 DMA（无 staging）
        pin_src = dev.pinned(n, name="src")
        pin_src.array[:] = a
        pin_dst = dev.pinned(n, name="dst")
        da.upload_async(pin_src.array, stream=s)
        dev.sync(s)
        ops.dev("add", da, db, dout, stream=s)
        dout.download_async(pin_dst.array, stream=s)
        dev.sync(s)
        check("pinned 源上传 + pinned 目标下载", np.allclose(pin_dst.array, a + b, atol=0, rtol=0),
              f"max|diff|={np.max(np.abs(pin_dst.array - (a + b))):.1e}")

        st = dev.stats()
        check("异步调用计数递增", st["async_upload_calls"] >= 3 and st["async_download_calls"] >= 1,
              f"up={st['async_upload_calls']} down={st['async_download_calls']}")
        dev.destroy_stream(s)


# ---------------------------------------------------------------------------
# [4] 传输带宽（NFR-4）
# ---------------------------------------------------------------------------
def test_bandwidth(mem: DeviceAllocator) -> None:
    print("\n[3] 传输带宽：同步(pageable) vs 异步(pinned)（NFR-4）")

    n = 4_000_000                                   # 16 MB，避开 L2 假象（L2 24MB）
    nbytes = n * 4
    rng = np.random.default_rng(1)
    host = rng.standard_normal(n).astype(np.float32)

    with DeviceAllocator(name="bw") as dev:
        d = dev.alloc_like(host, name="bw")
        s = dev.new_stream()

        def timed(fn, iters: int = 8, rounds: int = 3, warmup: int = 2) -> float:
            """取多轮中最快一轮的平均值（单次均值会被捼动与首达开销抬高）"""
            for _ in range(warmup):
                fn()
            best = float("inf")
            for _ in range(rounds):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                best = min(best, (time.perf_counter() - t0) / iters)
            return best

        pin = dev.pinned(n, name="bw-pin")
        pin.array[:] = host

        t_pageable = timed(lambda: d.upload(host))
        t_pinned = timed(lambda: d.upload(pin.array))
        t_async = timed(lambda: (d.upload_async(pin.array, stream=s), dev.sync(s)))

        bw_pageable = bandwidth_gbs(nbytes, t_pageable)
        bw_pinned = bandwidth_gbs(nbytes, t_pinned)
        bw_async = bandwidth_gbs(nbytes, t_async)
        print_table(["传输方式", "耗时 (ms)", "带宽 (GB/s)"], [
            ["同步 / pageable          ", f"{t_pageable * 1000:.3f}", f"{bw_pageable:.2f}"],
            ["同步 / pinned            ", f"{t_pinned * 1000:.3f}", f"{bw_pinned:.2f}"],
            ["异步 / pinned（逐次同步）", f"{t_async * 1000:.3f}", f"{bw_async:.2f}"],
        ])
        check("pinned 不低于 pageable", bw_pinned >= bw_pageable * 0.95,
              f"{bw_pinned:.2f} vs {bw_pageable:.2f} GB/s")
        check("异步不降低纯传输吞吐", bw_async >= bw_pinned * 0.9,
              f"{bw_async:.2f} vs {bw_pinned:.2f} GB/s（异步的价值在重叠，不在带宽）")
        check("带宽 ≥ 基线 5 GB/s 的 85%", bw_pinned >= 4.25, f"{bw_pinned:.2f} GB/s")


# ---------------------------------------------------------------------------
# [5] 分片
# ---------------------------------------------------------------------------
def test_chunking() -> None:
    print("\n[4] 大块自动分片（FR-3.3）")

    n = 500_000                                     # 2 MB
    rng = np.random.default_rng(2)
    host = rng.standard_normal(n).astype(np.float32)

    with DeviceAllocator(name="chunk", chunk_bytes=256 * 1024) as dev:
        d = dev.alloc_like(host, name="c")
        d.upload(host)
        got = d.to_numpy()
        check("分片后数据仍正确", np.array_equal(host, got), f"max|diff|={np.max(np.abs(host - got)):.1e}")

        s = dev.new_stream()
        d.upload_async(host, stream=s)
        dev.sync(s)
        d.download(got)
        check("异步分片也正确", np.array_equal(host, got),
              f"2 MB / 256 KB = {-(-host.nbytes // (256 * 1024))} 片")
        dev.destroy_stream(s)


# ---------------------------------------------------------------------------
# [6] 事件
# ---------------------------------------------------------------------------
def test_events() -> None:
    print("\n[5] 事件：record / wait / elapsed / done")

    with DeviceAllocator(name="ev") as dev:
        s = dev.new_stream()
        ev = dev.create_event()
        ev_t0 = dev.create_event(timed=True)
        ev_t1 = dev.create_event(timed=True)

        dev.record_event_on(ev_t0, s)
        n = 1_000_000
        host = np.random.randn(n).astype(np.float32)
        d = dev.alloc_like(host, name="e")
        for _ in range(20):
            d.upload_async(host, stream=s)
        dev.record_event_on(ev_t1, s)

        check("事件尚未完成", not dev.event_done(ev_t1), "done=False")
        dev.event_sync(ev_t1)
        ms = dev.event_elapsed(ev_t0, ev_t1)
        check("GPU 侧计时为正", ms > 0, f"{ms:.3f} ms")

        dev.record_event_on(ev, s)
        dev.event_sync(ev)
        check("同步后事件已完成", dev.event_done(ev), "done=True")

        # 跨流等待：compute 流等 copy 流的进度
        s2 = dev.new_stream()
        dev.wait_event(ev, s2)
        dev.sync(s2)
        check("跨流 wait 不报错", True, "wait_event ok")

        for e in (ev, ev_t0, ev_t1):
            dev.destroy_event(e)
        dev.destroy_stream(s)
        dev.destroy_stream(s2)


# ---------------------------------------------------------------------------
# [7] 双缓冲重叠（FR-3.4）
# ---------------------------------------------------------------------------
def test_double_buffer(ops: CudaOps) -> None:
    print("\n[8] 双缓冲流水线：H2D 与 kernel 重叠（FR-3.4）")

    chunks, n = 6, 2_000_000
    nbytes = n * 4
    rng = np.random.default_rng(3)
    data = [rng.standard_normal(n).astype(np.float32) for _ in range(chunks)]

    with DeviceAllocator(name="pipe") as dev:
        db = dev.alloc(n, name="b-const")
        db.upload(data[0])
        s = dev.new_stream()

        # ---- 标定：让每个 chunk 的计算耗时 ≈ 拷贝耗时，否则看不出重叠 ----
        # 先预热一次：首次 launch 会加载 kernel 模块（~20 ms），不预热会污染标定
        ops.dev("add", db, db, db, stream=s)
        dev.sync(s)
        t0 = time.perf_counter()
        db.upload(data[0])
        t_copy = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(50):
            ops.dev("add", db, db, db, stream=s)
        dev.sync(s)
        t_kernel = (time.perf_counter() - t0) / 50
        reps = max(1, int(round(t_copy / max(t_kernel, 1e-9))))
        print(f"    标定: 一次 {nbytes / 1e6:.0f} MB 拷贝 {t_copy * 1000:.2f} ms，"
              f"一次 kernel {t_kernel * 1000:.3f} ms -> 每 chunk 跑 {reps} 次 kernel")
        dev.destroy_stream(s)

        # 数据预先落在 pinned 内存里（真实场景是「生产者直接写进 staging」）——
        # 这样 upload_async 走直接 DMA 路径，没有主机侧 memmove 干扰重叠
        pins = [dev.pinned(n, name=f"chunk{i}") for i in range(chunks)]
        for pin, chunk in zip(pins, data):
            pin.array[:] = chunk

        # ---- 串行：单流，同步传输 + 计算 ----
        src = dev.alloc(n, name="src")
        ser_stream = dev.new_stream()
        t0 = time.perf_counter()
        for pin in pins:
            src.upload(pin.array, stream=ser_stream)
            for _ in range(reps):
                ops.dev("add", src, db, db, stream=ser_stream)
        dev.sync(ser_stream)
        t_serial = time.perf_counter() - t0
        dev.destroy_stream(ser_stream)

        # ---- 流水线 A：pinned 源（无主机侧中转）----
        src2 = dev.alloc(n, name="src2")
        t0 = time.perf_counter()
        with dev.double_buffer(n, slots=2, name="pipe") as pipe:
            for pin in pins:
                slot = pipe.next()
                slot.upload_async(pin.array)
                for _ in range(reps):
                    ops.dev("add", slot.buffer, db, db, stream=slot.compute_stream)
                slot.done()
            pipe.sync()
        t_pipe = time.perf_counter() - t0

        # ---- 流水线 B：pageable 源（每次多一次主机侧 memmove 到 staging）----
        src3 = dev.alloc(n, name="src3")
        t0 = time.perf_counter()
        with dev.double_buffer(n, slots=2, name="pipe2") as pipe2:
            for chunk in data:
                slot = pipe2.next()
                slot.upload_async(chunk)
                for _ in range(reps):
                    ops.dev("add", slot.buffer, db, db, stream=slot.compute_stream)
                slot.done()
            pipe2.sync()
        t_pipe_host = time.perf_counter() - t0

        print_table(["方式", "总耗时 (ms)", "相对串行"], [
            ["串行（单流 + 同步传输）", f"{t_serial * 1000:.2f}", "1.00x"],
            ["流水线（pinned 源）", f"{t_pipe * 1000:.2f}", f"{t_serial / t_pipe:.2f}x"],
            ["流水线（pageable 源，需 staging）", f"{t_pipe_host * 1000:.2f}",
             f"{t_serial / t_pipe_host:.2f}x"],
        ])
        check("流水线（pinned 源）快于串行", t_pipe < t_serial * 0.9,
              f"{t_serial * 1000:.1f} -> {t_pipe * 1000:.1f} ms ({t_serial / t_pipe:.2f}x)")
        src.release()
        src2.release()
        src3.release()
        db.release()
        for pin in pins:
            pin.release()


# ---------------------------------------------------------------------------
# [8] 在飞 pinned 有界
# ---------------------------------------------------------------------------
def test_inflight_bounded(mem: DeviceAllocator) -> None:
    print("\n[6] 在飞 pinned 有界（不随调用次数增长）")

    n = 200_000
    host = np.random.randn(n).astype(np.float32)

    with DeviceAllocator(name="inflight") as dev:
        d = dev.alloc_like(host, name="d")
        s = dev.new_stream()
        for _ in range(50):
            d.upload_async(host, stream=s)
            dev.sync(s)                              # 每轮同步 → staging 应被回收
        st = dev.stats()
        check("pinned 分配次数很少", st["pinned_allocs"] <= 3,
              f"pinned_allocs={st['pinned_allocs']}（50 次调用）")
        check("在飞数量归零", st["pinned_inflight"] == 0, f"inflight={st['pinned_inflight']}")

        # 不同步地连续发 10 次：在飞数量不会超过「尚未同步的提交数」，池会自动扩容到够用
        for _ in range(10):
            d.upload_async(host + 1.0, stream=s)
        inflight = dev.stats()["pinned_inflight"]
        check("在飞数量 ≤ 未同步提交数", 1 <= inflight <= 10, f"inflight={inflight}")
        dev.sync(s)
        st = dev.stats()
        check("同步后在飞归零", st["pinned_inflight"] == 0, f"inflight={st['pinned_inflight']}")
        check("pinned 总分配仍很少", st["pinned_allocs"] <= 5,
              f"pinned_allocs={st['pinned_allocs']}（累计 60 次异步上传）")
        dev.destroy_stream(s)


# ---------------------------------------------------------------------------
# [9] download_async 目标必须锁页
# ---------------------------------------------------------------------------
def test_download_requires_pinned(mem: DeviceAllocator) -> None:
    print("\n[7] download_async 目标必须锁页")

    n = 1000
    host = np.zeros(n, dtype=np.float32)
    with DeviceAllocator(name="dl") as dev:
        d = dev.alloc_like(host, name="d")
        s = dev.new_stream()
        d.upload(host)
        expect_raises("pageable 目标被拒", InvalidArgumentError,
                      d.download_async, host, stream=s)
        pin = dev.pinned(n, name="ok")
        d.download_async(pin.array, stream=s)
        dev.sync(s)
        check("pinned 目标可用", np.array_equal(pin.array, host), "")
        dev.destroy_stream(s)


# ---------------------------------------------------------------------------
# [10] 13 个算子跑在自定义流上
# ---------------------------------------------------------------------------
def test_all_ops_on_stream(ops: CudaOps) -> None:
    n = 100_000
    rng = np.random.default_rng(4)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    b_safe = (np.abs(b) + 0.5).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    scalar = np.zeros(1, dtype=np.float32)
    print(f"\n[9] 13 个算子跑在自定义流上（n={n:,}）")

    with DeviceAllocator(name="ops") as dev:
        da = dev.alloc_like(a, name="a")
        db = dev.alloc_like(b, name="b")
        db_safe = dev.alloc_like(b_safe, name="b_safe")
        dy = dev.alloc_like(b, name="y")
        dout = dev.alloc_like(a, name="out")
        dscalar = dev.alloc(1, name="scalar")
        s = dev.new_stream()

        # 一次性把输入异步传上去
        da.upload_async(a, stream=s)
        db.upload_async(b, stream=s)
        db_safe.upload_async(b_safe, stream=s)
        dev.sync(s)

        failed = []
        for spec in OPS:
            gb = db_safe if spec.safe_b else db
            np_b = b_safe if spec.safe_b else b
            if spec.kind == "binary":
                ops.dev(spec, da, gb, dout, stream=s)
                dev.sync(s)
                dout.download(out)
                ok = np.allclose(out, spec.ref(np, a, np_b, spec.scalar), rtol=1e-5, atol=0)
            elif spec.kind in ("unary", "scalar"):
                ops.dev(spec, da, out=dout, stream=s)
                dev.sync(s)
                dout.download(out)
                ok = np.allclose(out, spec.ref(np, a, np_b, spec.scalar), rtol=1e-5, atol=0)
            elif spec.kind == "axpy":
                dy.upload(b, stream=s)
                ops.dev(spec, da, dy, stream=s)
                dev.sync(s)
                dy.download(out)
                ok = np.allclose(out, spec.ref(np, a, b, spec.scalar), rtol=1e-4, atol=1e-4)
            else:
                dscalar.upload(scalar, stream=s)
                ops.dev(spec, da, gb, d_out=dscalar, stream=s)
                dev.sync(s)
                dscalar.download(scalar)
                exp = float(spec.ref(np, a, np_b, spec.scalar))
                ok = abs(float(scalar[0]) - exp) / max(abs(exp), 1e-6) < 1e-5
            if not ok:
                failed.append(spec.name)

        check("13 个算子全部正确", not failed, f"{len(OPS) - len(failed)}/{len(OPS)}")
        if failed:
            print(f"    失败: {', '.join(failed)}")
        dev.destroy_stream(s)


# ---------------------------------------------------------------------------
# [11] 主机回退
# ---------------------------------------------------------------------------
def test_host_fallback() -> None:
    print("\n[10] 主机回退模式下异步 API 不崩")

    with DeviceAllocator(force_host=True, name="fallback") as dev:
        n = 256
        host = np.arange(n, dtype=np.float32)
        d = dev.alloc_like(host, name="h")
        d.upload_async(host)
        got = np.zeros(n, dtype=np.float32)
        d.download_async(got)
        check("回退模式异步往返正确", np.array_equal(host, got),
              f"max|diff|={np.max(np.abs(host - got)):.1e}")

        pin = dev.pinned(n, name="hp")
        pin.array[:] = host
        check("回退模式 pinned 视图可用", pin.array[-1] == n - 1, f"last={pin.array[-1]}")
        pin.release()
        check("回退模式 new_stream 返回 0", dev.new_stream() == 0, "")
        expect_raises("回退模式不支持事件", DevMemError, dev.create_event)
        expect_raises("回退模式不支持双缓冲", DevMemError, dev.double_buffer, n)


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 90)
    print("方案A 阶段3 验收：pinned staging + 异步拷贝 + 事件 + 双缓冲")
    print("=" * 90)

    with DeviceAllocator(name="main") as mem:
        print(f"模式   : {mem.mode}")
        print(f"cupy   : 已屏蔽（本测试不依赖 cupy）")
        ops = CudaOps()
        print(f"算子库 : {ops.lib_dir}  ({len(ops.ops)} 个算子)")

        test_pinned(mem)
        test_async_transfer(mem, ops)
        test_bandwidth(mem)
        test_chunking()
        test_events()
        test_inflight_bounded(mem)
        test_download_requires_pinned(mem)

    test_double_buffer(ops)
    test_all_ops_on_stream(ops)
    test_host_fallback()

    print("\n" + "=" * 90)
    if _FAILED:
        print(f"失败 {len(_FAILED)} 项: {', '.join(_FAILED)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
