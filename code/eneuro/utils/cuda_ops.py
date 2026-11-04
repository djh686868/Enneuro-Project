# -*- coding: utf-8 -*-
"""CUDA 算子桥接层：把 `code/eneuro/cuda/cu/*.cu` 编译出的算子统一接进 Python

每个算子都有两个入口：

    host 版  launch_xxx(主机指针...)                 每次调用 malloc + H2D + kernel + D2H + free
    dev  版  launch_xxx_dev(设备指针..., stream)     只借指针，零拷贝（方案B）

`dev` 版的前置条件（由 `device_ptr` 统一校验，不满足就抛错而不是静默算错）：

    · 必须是 DeviceBuffer（方案A）、cupy.ndarray / device='cuda' 的 Tensor（方案B），
      或**任何实现 `__cuda_array_interface__` 的对象**（torch / numba，阶段4）
    · dtype 必须是 float32        —— kernel 只按 float 解释内存
    · 必须 C 连续                 —— 非连续时 .ptr 只指向第一段
    · 必须在当前设备上            —— 与 nvcc 共用 primary context 的硬前提

典型用法：

    from eneuro.utils.cuda_ops import CudaOps, get_op
    import cupy as cp

    ops = CudaOps()
    spec = get_op("add")
    a, b = cp.random.randn(n, dtype=cp.float32), cp.random.randn(n, dtype=cp.float32)
    out = cp.empty_like(a)
    ops.dev(spec, a, b, out)                  # 零拷贝
    cp.cuda.Stream.null.synchronize()
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from eneuro.utils.dev_memory import DeviceBuffer, cai_ptr

try:
    import cupy as cp
except ImportError:                                     # pragma: no cover
    cp = None                                           # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 编译产物与本文件同属 eneuro 包：code/eneuro/cuda/{cu,dll}
ENEURO_DIR = Path(__file__).resolve().parents[1]        # .../code/eneuro
DEFAULT_LIB_DIR = Path(os.environ.get("ENE_CUDA_LIB_DIR")
                       or (ENEURO_DIR / "cuda" / ("dll" if os.name == "nt" else "so")))
EXT = "dll" if os.name == "nt" else "so"

FP = ctypes.POINTER(ctypes.c_float)
F = ctypes.c_float
I = ctypes.c_int
VP = ctypes.c_void_p


# ---------------------------------------------------------------------------
# 算子表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    """一个算子的两种入口 + 参考实现"""

    name: str
    lib: str                 # 动态库名（不含扩展名）
    host_fn: str             # 主机指针入口
    dev_fn: str              # 设备指针入口
    kind: str                # binary | unary | scalar | axpy | reduce1 | reduce2
    ref: Callable[[Any, Any, Any, float], Any]   # ref(xp, a, b, scalar) -> 期望值
    scalar: float = 0.0      # scalar 类算子的默认标量
    safe_b: bool = False     # 除法等需要非零分母


OPS: tuple[OpSpec, ...] = (
    OpSpec("add", "vector_add", "launch_add", "launch_add_dev", "binary",
           lambda xp, a, b, s: a + b),
    OpSpec("sub", "vector_sub", "launch_sub", "launch_sub_dev", "binary",
           lambda xp, a, b, s: a - b),
    OpSpec("mul", "vector_mul", "launch_mul", "launch_mul_dev", "binary",
           lambda xp, a, b, s: a * b),
    OpSpec("div", "vector_div", "launch_div", "launch_div_dev", "binary",
           lambda xp, a, b, s: a / b, safe_b=True),
    OpSpec("maximum", "vector_maximum", "launch_maximum", "launch_maximum_dev", "binary",
           lambda xp, a, b, s: xp.maximum(a, b)),
    OpSpec("minimum", "vector_minimum", "launch_minimum", "launch_minimum_dev", "binary",
           lambda xp, a, b, s: xp.minimum(a, b)),
    OpSpec("neg", "vector_neg", "launch_neg", "launch_neg_dev", "unary",
           lambda xp, a, b, s: -a),
    OpSpec("relu", "vector_relu", "launch_relu", "launch_relu_dev", "unary",
           lambda xp, a, b, s: xp.maximum(a, xp.float32(0))),
    OpSpec("scale", "vector_scale", "launch_scale", "launch_scale_dev", "scalar",
           lambda xp, a, b, s: a * s, scalar=1.7),
    OpSpec("add_scalar", "vector_add_scalar", "launch_add_scalar", "launch_add_scalar_dev", "scalar",
           lambda xp, a, b, s: a + s, scalar=-0.35),
    OpSpec("axpy", "axpy", "launch_axpy", "launch_axpy_dev", "axpy",
           lambda xp, a, b, s: s * a + b, scalar=1.7),
    OpSpec("sum", "vector_sum", "launch_sum", "launch_sum_dev", "reduce1",
           lambda xp, a, b, s: xp.sum(a)),
    OpSpec("dot", "vector_dot", "launch_dot", "launch_dot_dev", "reduce2",
           lambda xp, a, b, s: xp.sum(a * b)),
)

_BY_NAME = {spec.name: spec for spec in OPS}

ELEMENTWISE_KINDS = ("binary", "unary", "scalar", "axpy")
REDUCE_KINDS = ("reduce1", "reduce2")


def get_op(name: str) -> OpSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"未知算子 {name!r}，可选: {', '.join(_BY_NAME)}") from None


def signature(kind: str, host: bool):
    """返回 (argtypes, restype)。规则：指针在前，标量在后，dev 版最后追加 stream。"""
    P = FP if host else VP
    if kind == "binary":
        args = [P, P, P, I]
    elif kind == "unary":
        args = [P, P, I]
    elif kind in ("scalar", "axpy"):
        args = [P, P, F, I]
    elif kind == "reduce1":
        args = [P, I] if host else [P, I, P]
    elif kind == "reduce2":
        args = [P, P, I] if host else [P, P, I, P]
    else:
        raise ValueError(f"未知 kind: {kind}")

    if not host:
        args = args + [VP]                       # stream
    restype = F if (host and kind in REDUCE_KINDS) else None
    return args, restype


# ---------------------------------------------------------------------------
# 指针转换
# ---------------------------------------------------------------------------
def np_ptr(arr: Optional[np.ndarray]):
    """主机指针；None -> NULL"""
    if arr is None:
        return None
    return ctypes.cast(arr.ctypes.data, FP)


def device_ptr(x: Any):
    """取设备指针，并校验前置条件。

    接受四类输入：
      · `DeviceBuffer`（方案A：自管理显存，句柄已做所有权/设备校验）
      · `cupy.ndarray`（方案B：零拷贝复用 cupy 显存）
      · `eneuro.base.Tensor`（device='cuda'，内部还是 cupy.ndarray）
      · 任何实现 `__cuda_array_interface__` 的对象（torch / numba / 第三方，
        **这条路径不需要 cupy**，见阶段4 的 `cai_ptr`）
    """
    # 方案A：句柄的 ptr 属性自带「已释放」检测，取指针是 O(1)，不流出裸指针
    if isinstance(x, DeviceBuffer):
        return VP(x.ptr)

    # 方案B：cupy / Tensor 走专用分支，错误信息更具体（dtype/连续性/设备号）
    if cp is not None:
        from eneuro.base import Tensor as EneTensor

        arr: Any = x
        if isinstance(arr, EneTensor):
            if arr.device not in ("cuda", "gpu"):
                raise ValueError(f"Tensor 在 {arr.device} 上；dev 接口只接受显存数据")
            arr = arr.data

        if isinstance(arr, cp.ndarray):
            if arr.dtype != cp.float32:
                # kernel 按 float 解释内存，dtype 不对不会报错，只会算出垃圾值
                raise TypeError(f"kernel 按 float32 解释内存，实际 dtype = {arr.dtype}")
            if not arr.flags.c_contiguous:
                # 非连续时 .ptr 只指向第一段，size 会骗人
                raise ValueError("需要 C 连续数组；请先 cp.ascontiguousarray（注意会触发隐式拷贝）")
            if arr.device.id != cp.cuda.runtime.getDevice():
                raise ValueError(f"数组在第 {arr.device.id} 张卡，当前设备是 "
                                 f"{cp.cuda.runtime.getDevice()}")
            return VP(arr.data.ptr)

    # 通用兜底：任何实现 CUDA Array Interface 的对象（阶段4）；cupy 缺席也能走
    if getattr(x, "__cuda_array_interface__", None) is not None:
        ptr, _nbytes = cai_ptr(x, what=f"device_ptr({type(x).__name__})")
        return VP(ptr)

    if cp is None:
        raise RuntimeError("需要 cupy、DeviceBuffer 或实现 __cuda_array_interface__ 的对象"
                           "才能使用方案B 的设备指针接口")
    raise TypeError(f"需要 cupy.ndarray（或 device='cuda' 的 Tensor），得到 {type(x).__name__}")


def stream_ptr(stream: Any = None) -> int:
    """把 stream 规范化成裸指针

    · None            -> cupy 可用时取当前流，否则默认流 0
    · int             -> **原样透传**（方案A 的流就是裸指针，不能丢掉）
    · cupy Stream 等  -> 取 .ptr
    """
    if stream is None:
        if cp is None:
            return 0
        stream = cp.cuda.get_current_stream()
    if isinstance(stream, int):
        return stream
    return int(stream.ptr)


def _prepare_dev_args(spec_name: str, objs, sid: int) -> None:
    """下发 kernel 前的两项准备（阶段5）

    1. FR-8.1/8.2：**每个分配器只校验一次**当前设备（不是每个缓冲都查一次 CUDA）
    2. FR-7.1：把「本块在这个流上用过」告诉 `DeviceBuffer`；
       `cross_stream_reuse=False`（默认）时该方法立即返回，热路径可忽略
    """
    checked: set[int] = set()
    for obj in objs:
        if not isinstance(obj, DeviceBuffer):
            continue
        aid = id(obj.allocator)
        if aid not in checked:
            checked.add(aid)
            obj.assert_current_device(what=f"ops.dev({spec_name})")
        obj._ene_note_stream(sid)


# ---------------------------------------------------------------------------
class CudaOps:
    """13 个算子的 host / dev 两套入口的统一封装"""

    def __init__(self, lib_dir: Optional[Path | str] = None, ops: Sequence[OpSpec] = OPS):
        self.lib_dir = Path(lib_dir) if lib_dir else DEFAULT_LIB_DIR
        self.ops = list(ops)
        self._libs: dict[str, ctypes.CDLL] = {}
        self._fns: dict[tuple[str, bool], Any] = {}
        self._prepare()

    # ---- 加载 ----
    def _lib(self, name: str) -> ctypes.CDLL:
        if name not in self._libs:
            path = self.lib_dir / f"{name}.{EXT}"
            if not path.exists():
                raise FileNotFoundError(f"未找到 {path}，请先运行 code/eneuro/cuda/build.bat 编译")
            self._libs[name] = ctypes.CDLL(str(path))
        return self._libs[name]

    def _prepare(self) -> None:
        for spec in self.ops:
            for host in (True, False):
                fn = getattr(self._lib(spec.lib), spec.host_fn if host else spec.dev_fn)
                fn.argtypes, fn.restype = signature(spec.kind, host)
                self._fns[(spec.name, host)] = fn

    @property
    def missing(self) -> list[str]:
        """缺失的库（构造成功即无缺失）"""
        return [s.lib for s in self.ops if s.lib not in self._libs]

    # ---- host 版：主机指针 ----
    def host(self, spec: OpSpec | str, a: np.ndarray, b: Optional[np.ndarray] = None,
             out: Optional[np.ndarray] = None, scalar: Optional[float] = None):
        spec = get_op(spec) if isinstance(spec, str) else spec
        fn = self._fns[(spec.name, True)]
        n = int(a.size)
        s = spec.scalar if scalar is None else scalar
        pa, pb, pc = np_ptr(a), np_ptr(b), np_ptr(out)

        if spec.kind == "binary":
            fn(pa, pb, pc, n)
        elif spec.kind == "unary":
            fn(pa, pc, n)
        elif spec.kind == "scalar":
            fn(pa, pc, F(s), n)
        elif spec.kind == "axpy":
            fn(pa, pb, F(s), n)
        elif spec.kind == "reduce1":
            return float(fn(pa, n))
        elif spec.kind == "reduce2":
            return float(fn(pa, pb, n))
        return None

    # ---- dev 版：设备指针，零拷贝 ----
    def dev(self, spec: OpSpec | str, a: Any, b: Any = None, out: Any = None,
            scalar: Optional[float] = None, stream: Any = None, d_out: Any = None):
        """归约类算子请传 d_out（常驻的 1 元素 float32 显存），结果写在其中。"""
        spec = get_op(spec) if isinstance(spec, str) else spec
        fn = self._fns[(spec.name, False)]
        n = int(out.size) if out is not None else int(a.size)
        s = spec.scalar if scalar is None else scalar
        sid = stream_ptr(stream)
        sp = VP(sid)
        _prepare_dev_args(spec.name, (a, b, out, d_out), sid)

        pa = device_ptr(a)
        pb = device_ptr(b) if b is not None else None
        pc = device_ptr(out) if out is not None else None
        po = device_ptr(d_out) if d_out is not None else None

        if spec.kind == "binary":
            fn(pa, pb, pc, n, sp)
        elif spec.kind == "unary":
            fn(pa, pc, n, sp)
        elif spec.kind == "scalar":
            fn(pa, pc, F(s), n, sp)
        elif spec.kind == "axpy":
            fn(pa, pb, F(s), n, sp)
        elif spec.kind == "reduce1":
            fn(pa, n, po, sp)
        elif spec.kind == "reduce2":
            fn(pa, pb, n, po, sp)
        return None

    # ---- cupy 原生参考（用于对照） ----
    @staticmethod
    def cupy_ref(spec: OpSpec | str, a: Any, b: Any = None, scalar: Optional[float] = None):
        spec = get_op(spec) if isinstance(spec, str) else spec
        s = spec.scalar if scalar is None else scalar
        return spec.ref(cp, a, b, s)

    @staticmethod
    def numpy_ref(spec: OpSpec | str, a: np.ndarray, b: Optional[np.ndarray] = None,
                  scalar: Optional[float] = None):
        spec = get_op(spec) if isinstance(spec, str) else spec
        s = spec.scalar if scalar is None else scalar
        return spec.ref(np, a, b, s)
