import hashlib
import os

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

# RawModule 编译是昂贵操作；按源码、kernel 名称和 GPU 计算能力缓存，避免
# 每次前向传播都触发 NVRTC 编译。计算能力进入 key 是因为同一 CUDA 源码
# 可能需要针对不同 SM 架构重新生成代码。
_CACHE = {}
_AVAILABLE = None


def available():
    """返回当前进程是否能看到至少一个 CUDA 设备。"""
    global _AVAILABLE
    # ``dispatch._raw_enabled`` is called for every tensor operation.  Querying
    # the CUDA runtime on every call adds avoidable host overhead during a
    # multi-thousand-batch run, so cache the process-level result.
    if _AVAILABLE is not None:
        return _AVAILABLE
    if cp is None:
        _AVAILABLE = False
        return _AVAILABLE
    try:
        _AVAILABLE = int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def module(source, names=(), options=()):
    """用 CuPy/NVRTC 将内嵌 CUDA C 编译为可调用的 RawModule。"""
    if cp is None:
        raise RuntimeError("CuPy is not installed")
    try:
        device = cp.cuda.Device()
        cc = device.compute_capability
    except Exception:
        cc = "unknown"
    key = (hashlib.sha256(source.encode("utf8")).hexdigest(), tuple(names), cc, tuple(options))
    if key not in _CACHE:
        _CACHE[key] = cp.RawModule(
            code=source,
            options=tuple(options) + ("--std=c++14",),
            name_expressions=list(names),
        )
    return _CACHE[key]


def clear_cache():
    """清空进程内的 RawModule 缓存，便于修改 kernel 后重新编译。"""
    _CACHE.clear()
