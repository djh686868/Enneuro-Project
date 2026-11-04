# -*- coding: utf-8 -*-
"""方案B（零拷贝 / 显存常驻）最小实现示例 —— 以 vector_add.cu 为例

对应 `code/eneuro/cuda/cu/vector_add.cu` 里的两个导出：

    launch_add(a, b, c, n)                # 旧接口：主机指针，每次调用都 malloc + H2D + D2H
    launch_add_dev(a, b, c, n, stream)    # 方案B ：设备指针，只借指针，不分配不拷贝

本文件演示方案B 落地时必须处理的 5 件事：

    1. 如何把 cupy 的 `ndarray.data.ptr` 传给 nvcc 编译出的 kernel
    2. 三项前置校验（dtype / 连续性 / 设备）—— 必须在 Python 侧拦住，不能让 kernel 去猜
    3. stream 透传，保证与 cupy 的非默认流有序
    4. 与 `eneuro.base.Tensor(device='cuda')` 对接
    5. 正确性与性能对比

运行：
    python code/tests/demo_cu_cupy_zerocopy.py
"""
from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402

from eneuro.base import Tensor as EneTensor  # noqa: E402

LIB_PATH = ROOT / "eneuro" / "cuda" / ("dll" if os.name == "nt" else "so") / f"vector_add.{'dll' if os.name == 'nt' else 'so'}"

FP = ctypes.POINTER(ctypes.c_float)
I = ctypes.c_int

# ---------------------------------------------------------------------------
# 1. 加载库：两个入口的签名
# ---------------------------------------------------------------------------
_lib = ctypes.CDLL(str(LIB_PATH))

# 旧接口（主机指针）
_lib.launch_add.argtypes = [FP, FP, FP, I]
_lib.launch_add.restype = None

# 方案B（设备指针）。用 c_void_p 而不是 POINTER(c_float)：接的是裸地址，不是主机数组。
# stream 用 c_void_p 接，Python 侧传 cupy stream 的 .ptr 即可。
_lib.launch_add_dev.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, I, ctypes.c_void_p]
_lib.launch_add_dev.restype = None


# ---------------------------------------------------------------------------
# 2. 前置校验：三种非法输入必须在进 kernel 之前挡住
# ---------------------------------------------------------------------------
def device_ptr(x) -> ctypes.c_void_p:
    """取出设备指针。接受 cupy.ndarray 或 device='cuda' 的 eneuro Tensor。"""
    # 必须显式判断类型：numpy 2.x 的 ndarray 也有 .device（字符串 'cpu'）和 .data（memoryview），
    # 用 hasattr(x, "device") 这类鸭子判断会把 numpy 数组错认成 Tensor
    arr: Any = x
    if isinstance(arr, EneTensor):
        if arr.device not in ("cuda", "gpu"):
            raise ValueError(f"Tensor 在 {arr.device} 上；方案B 只适用于显存数据")
        arr = arr.data

    if not isinstance(arr, cp.ndarray):
        raise TypeError(f"需要 cupy.ndarray（或 device='cuda' 的 Tensor），得到 {type(x).__name__}")

    if arr.dtype != cp.float32:
        # kernel 是按 float 解释这段内存的，dtype 不对会算出垃圾值而不是报错
        raise TypeError(f"kernel 按 float32 解释内存，实际 dtype = {arr.dtype}")
    if not arr.flags.c_contiguous:
        # 非连续时 .ptr 只指向第一段，size 会骗人；必须先 ascontiguousarray（那是隐式拷贝）
        raise ValueError("需要 C 连续数组；请先 cp.ascontiguousarray（注意会触发隐式拷贝）")

    # 同一张卡是硬前提：nvcc 的运行时 API 与 cupy 共用 primary context
    if arr.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"数组在第 {arr.device.id} 张卡，当前设备是 {cp.cuda.runtime.getDevice()}")

    return ctypes.c_void_p(arr.data.ptr)


# ---------------------------------------------------------------------------
# 3. 算子封装：不分配、不拷贝、不同步（异步语义与 cupy 一致）
# ---------------------------------------------------------------------------
def cuda_add(a, b, out=None, stream=None):
    """out 省略时在 cupy 的显存池里分配；同步交给调用方（和 cupy 一样）"""
    if out is None:
        out = cp.empty_like(a)
    if stream is None:
        stream = cp.cuda.get_current_stream()
    _lib.launch_add_dev(device_ptr(a), device_ptr(b), device_ptr(out),
                        int(out.size), ctypes.c_void_p(stream.ptr))
    return out


# ---------------------------------------------------------------------------
def main():
    n = 1_000_000
    a = cp.random.randn(n, dtype=cp.float32)
    b = cp.random.randn(n, dtype=cp.float32)
    ref = a + b

    print("=" * 78)
    print("方案B 零拷贝实现示例（vector_add.cu）")
    print("=" * 78)
    print(f"GPU   : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"库    : {LIB_PATH}")
    print(f"n     : {n:,}\n")

    # ---- (1) 正确性：cupy 指针直接进 kernel ----
    c = cuda_add(a, b)
    cp.cuda.Stream.null.synchronize()
    print(f"[1] cupy.ndarray -> kernel     : max|diff| = {float(cp.max(cp.abs(c - ref))):.3e}")

    # ---- (2) 与 eneuro Tensor 对接 ----
    ta = EneTensor(a, device="cuda")
    tb = EneTensor(b, device="cuda")
    tc = EneTensor(cp.empty_like(a), device="cuda")
    cuda_add(ta, tb, tc)                       # 直接传 Tensor 也行
    cp.cuda.Stream.null.synchronize()
    print(f"[2] eneuro Tensor(device=cuda) : max|diff| = {float(cp.max(cp.abs(tc.data - ref))):.3e}"
          f"   (Tensor.device = {tc.device})")

    # ---- (3) 非默认流：stream 透传后与 cupy 有序 ----
    with cp.cuda.Stream() as s:
        c2 = cuda_add(a, b, stream=s)          # 在 s 上启动
        s.synchronize()
    print(f"[3] 非默认流 (stream 透传)      : max|diff| = {float(cp.max(cp.abs(c2 - ref))):.3e}")

    # ---- (4) 非法输入被拦住（而不是静默算错）----
    bad = cp.zeros((n, 2), dtype=cp.float32)[:, 0]        # 非连续视图
    for name, arr in [("非连续数组", bad),
                      ("float64 数组", cp.zeros(n, dtype=cp.float64)),
                      ("numpy 数组", np.zeros(n, dtype=np.float32))]:
        try:
            cuda_add(arr, arr)
            print(f"[4] {name:<12}: 未拦截 ??")
        except (ValueError, TypeError) as exc:
            print(f"[4] {name:<12}: 已拦截 -> {type(exc).__name__}: {str(exc)[:52]}")

    # ---- (5) 性能：交错测量，避免顺序漂移 ----
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = np.zeros(n, dtype=np.float32)
    pa, pb, pc = (ctypes.cast(x.ctypes.data, FP) for x in (a_np, b_np, c_np))
    out_fixed = cp.empty_like(a)
    stream_ptr = ctypes.c_void_p(cp.cuda.get_current_stream().ptr)

    def host_version():
        _lib.launch_add(pa, pb, pc, n)

    def dev_reuse_out():
        _lib.launch_add_dev(ctypes.c_void_p(a.data.ptr), ctypes.c_void_p(b.data.ptr),
                            ctypes.c_void_p(out_fixed.data.ptr), n, stream_ptr)

    def dev_alloc_out():
        cuda_add(a, b)

    def cupy_add():
        return a + b

    cases = [
        ("launch_add   主机指针(malloc+拷贝)", host_version, None),
        ("launch_add_dev 设备指针(out 复用)", dev_reuse_out, None),
        ("launch_add_dev 设备指针(每次分配)", dev_alloc_out, cp.cuda.Stream.null.synchronize),
        ("cupy a + b（自带算子）", cupy_add, cp.cuda.Stream.null.synchronize),
    ]

    for _, fn, _sync in cases:
        for _ in range(10):
            fn()
    cp.cuda.Stream.null.synchronize()

    iters, rounds = 50, 7
    samples = {name: [] for name, _, _ in cases}
    for _ in range(rounds):
        for name, fn, sync in cases:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            if sync:
                sync()
            samples[name].append((time.perf_counter() - t0) * 1000 / iters)

    print(f"\n[5] 性能（n={n:,}，交错测量 min-of-{rounds}）")
    base = min(samples[cases[0][0]])
    for name, _, _ in cases:
        ms = min(samples[name])
        print(f"    {name:<34} {ms:8.4f} ms   相对旧接口 {base / ms:6.1f}x")

    print("\n说明：")
    print("  · launch_add_dev 不分配、不拷贝，故耗时 ≈ 纯 kernel（内存带宽受限）")
    print("  · out 复用与每次 empty_like 差别很小，因为 cupy 显存池复用已分配块")
    print("  · 同步交给调用方，才能与 cupy 的异步流水线对齐（示例里只是为计时才 sync）")


if __name__ == "__main__":
    main()
