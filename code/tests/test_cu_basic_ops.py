"""
基础 CUDA 算子验证脚本 (对照 numpy 逐个校验)

使用前先编译:
    Windows :  code\eneuro\cuda\build.bat
    Linux   :  bash code/eneuro/cuda/build.sh

运行:
    python code/tests/test_cu_basic_ops.py
"""
import ctypes
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.path.join(ROOT, "eneuro", "cuda", "dll" if os.name == "nt" else "so")
EXT = "dll" if os.name == "nt" else "so"

FP = ctypes.POINTER(ctypes.c_float)
F = ctypes.c_float
I = ctypes.c_int


def load_lib(name, funcs):
    """按库名加载动态库并设置各导出函数的签名。"""
    path = os.path.join(LIB_DIR, f"{name}.{EXT}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}, 请先编译 (code/eneuro/cuda/build.bat)")
    lib = ctypes.CDLL(path)
    for fname, (argtypes, restype) in funcs.items():
        fn = getattr(lib, fname)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def ptr(arr):
    """numpy 数组 -> ctypes float* 指针"""
    return arr.ctypes.data_as(FP)


_results = []


def check(name, got, expect, rtol=1e-5, atol=1e-6, show=3):
    ok = np.allclose(got, expect, rtol=rtol, atol=atol, equal_nan=True)
    if np.ndim(got) == 0:
        diff = abs(float(got) - float(expect))
    else:
        diff = float(np.max(np.abs(got - expect)))
        if show and not ok:
            bad = np.argmax(np.abs(got - expect))
            print(f"    首个不匹配 idx={bad}: got={got[bad]!r} expect={expect[bad]!r}")
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<14} max|diff| = {diff:.3e}")
    return ok


N = 1_000_000
rng = np.random.default_rng(0)
a = rng.standard_normal(N).astype(np.float32)
b = rng.standard_normal(N).astype(np.float32)
b_pos = (np.abs(b) + 0.5).astype(np.float32)
out = np.zeros_like(a)

ALPHA = 1.7
BETA = -0.35
ZERO = np.zeros_like(a)


def test_binary(name, libname, fn_name, ref, extra_lib=False):
    """通用二元算子测试: c = fn(a, b)"""
    lib = load_lib(libname, {fn_name: ([FP, FP, FP, I], None)})
    out.fill(0)
    getattr(lib, fn_name)(ptr(a), ptr(b_pos if extra_lib else b), ptr(out), N)
    check(name, out, ref)


def main():
    print(f"设备库目录: {LIB_DIR}  (n = {N:,})")

    # ---------- 逐元素二元运算 ----------
    test_binary("add", "vector_add", "launch_add", a + b)
    test_binary("sub", "vector_sub", "launch_sub", a - b)
    test_binary("mul", "vector_mul", "launch_mul", a * b)
    test_binary("div", "vector_div", "launch_div", a / b_pos, extra_lib=True)
    test_binary("maximum", "vector_maximum", "launch_maximum", np.maximum(a, b))
    test_binary("minimum", "vector_minimum", "launch_minimum", np.minimum(a, b))

    # ---------- 逐元素一元运算 ----------
    lib = load_lib("vector_neg", {"launch_neg": ([FP, FP, I], None)})
    out.fill(0)
    lib.launch_neg(ptr(a), ptr(out), N)
    check("neg", out, -a)

    lib = load_lib("vector_relu", {"launch_relu": ([FP, FP, I], None)})
    out.fill(0)
    lib.launch_relu(ptr(a), ptr(out), N)
    check("relu", out, np.maximum(a, np.float32(0)))

    # ---------- 标量运算 ----------
    lib = load_lib("vector_scale", {"launch_scale": ([FP, FP, F, I], None)})
    out.fill(0)
    lib.launch_scale(ptr(a), ptr(out), F(ALPHA), N)
    check("scale", out, a * np.float32(ALPHA))

    lib = load_lib("vector_add_scalar", {"launch_add_scalar": ([FP, FP, F, I], None)})
    out.fill(0)
    lib.launch_add_scalar(ptr(a), ptr(out), F(BETA), N)
    check("add_scalar", out, a + np.float32(BETA))

    # ---------- 融合乘加 (原地) ----------
    lib = load_lib("axpy", {"launch_axpy": ([FP, FP, F, I], None)})
    y = b.copy()
    lib.launch_axpy(ptr(a), ptr(y), F(ALPHA), N)
    check("axpy", y, np.float32(ALPHA) * a + b, rtol=1e-4, atol=1e-4)

    # ---------- 归约运算 ----------
    lib = load_lib("vector_sum", {"launch_sum": ([FP, I], F)})
    got = lib.launch_sum(ptr(a), N)
    exp = np.sum(a, dtype=np.float64)
    check("sum", np.float32(got), np.float32(exp), rtol=1e-3, atol=1e-3)

    lib = load_lib("vector_dot", {"launch_dot": ([FP, FP, I], F)})
    got = lib.launch_dot(ptr(a), ptr(b), N)
    exp = np.dot(a.astype(np.float64), b.astype(np.float64))
    check("dot", np.float32(got), np.float32(exp), rtol=1e-3, atol=1e-3)

    # ---------- 汇总 ----------
    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} 通过")
    if failed:
        print("失败算子: " + ", ".join(failed))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
