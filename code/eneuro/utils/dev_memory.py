# -*- coding: utf-8 -*-
"""方案A 自管理设备显存

职责划分（见 `doc/CUDAC/显存管理.md` 第四节 D1/D2）：

    C 层（dev_memory.dll）   无状态薄封装：cudaMalloc/cudaFree/cudaMemcpy + 错误码 + 计数
    本模块                   句柄化 + 块表 + 双重释放检测 + 泄漏清单 + 上下文管理

**已实现**：
    阶段1  直通分配、同步传输、句柄与生命周期、错误上报、统计、无 CUDA 回退
    阶段2  size-class 内存池（命中即复用，稳态零 cudaMalloc）、trim、池驱逐、池统计
    阶段3  pinned staging 池、异步拷贝、流与事件、双缓冲流水线
    阶段4  `__cuda_array_interface__` / `as_cupy()` 零拷贝互操作、外部视图租约、
           混合模式（cupy 输入 + 池 workspace）、不依赖 cupy 的 CAI 反向读取

**不含**（后续阶段）：多设备池隔离、跨流复用防护、C 侧计数加锁（阶段5）。

不依赖 cupy（NFR-6）：`ctypes` + `numpy` 即可跑完整个方案A；
只有显式调用 `as_cupy()` / `cai_ptr()` 的互操作路径才需要 cupy 或对应的库。

典型用法：

    from eneuro.utils.dev_memory import DeviceAllocator
    from eneuro.utils.cuda_ops import CudaOps

    with DeviceAllocator() as mem:                       # 会话级
        da = mem.alloc_like(a_np, name="a")
        db = mem.alloc_like(b_np, name="b")
        dout = mem.alloc_like(a_np, name="out")
        da.upload(a_np); db.upload(b_np)                 # 一次进
        CudaOps().dev("add", da, db, dout)               # 多次算（复用现有算子）
        dout.download(out_np)                            # 一次出
    # 退出时自动归还；有残留会打印泄漏清单

互操作（阶段4）：

    with mem.alloc(n, name="ws") as d:                   # 池里的 workspace
        arr = d.as_cupy()                                # 零拷贝 + 登记租约
        arr[:] = 1.0                                     # cupy 直接写
        ops.dev("add", cp_a, cp_b, d)                     # cupy 输入 + 池输出 混用
"""
from __future__ import annotations

import atexit
import ctypes
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
ENEURO_DIR = Path(__file__).resolve().parents[1]        # .../code/eneuro
DEFAULT_LIB_DIR = Path(os.environ.get("ENE_CUDA_LIB_DIR")
                       or (ENEURO_DIR / "cuda" / ("dll" if os.name == "nt" else "so")))
EXT = "dll" if os.name == "nt" else "so"
LIB_NAME = "dev_memory"

ENE_OK = 0
ENE_ERR_OOM = 1
ENE_ERR_INVALID = 2
ENE_ERR_DEVICE = 3
ENE_ERR_CUDA = 4
ENE_ERR_NODEVICE = 5


# ---------------------------------------------------------------------------
# 内存池的 size-class 划分（见 doc/CUDAC/显存管理.md 5.1）
#
#   s = 256 B 步进（≤ 4 KB）  —— 小块最密集，按 256 B 对齐把内部碎片压到最小
#   p = 2 的幂（4 KB ~ 1 MB） —— 桶数有限（9 个），内部碎片 ≤ 2x
#   e = 精确大小（> 1 MB）    —— 不再取整，避免大块白白浪费显存
#
# 「best-fit」在**桶级别**完成：选最小的、装得下的桶；
# 桶内所有块容量相同，取任意一个即可（O(1) pop），无需遍历比较。
#
# 注：设计文档原本写「> 1 MB 直通不进池」，这里改为「按精确大小分桶进池」。
# 原因是 NFR-3（稳态零 cudaMalloc）在典型张量尺寸（往往 > 1 MB）上必须也成立；
# 而「大块长期占显存」的问题由 pool_max_bytes 上限 + trim(older_than=) 解决，
# 比一刀切直通更精细。需要旧行为时传 pool_max_block=1<<20 即可。
_SMALL_STEP = 256
_SMALL_MAX = 4096
_LARGE_MIN = 1 << 20

# debug 越界哨兵（FR-9.2）——头尾各留 256 B，保证用户指针仍是 256 B 对齐
_GUARD_BYTES = 256
_GUARD_MAGIC = b"\xee\x11\x22\x33" * (_GUARD_BYTES // 4)

BucketKey = tuple[str, int]                       # (kind, capacity)


def _size_class(nbytes: int) -> tuple[BucketKey, int]:
    """返回 (桶 key, 容量)。容量 ≥ nbytes，容量相同的块共用一个空闲链表。"""
    if nbytes <= _SMALL_STEP:
        return ("s", _SMALL_STEP), _SMALL_STEP
    if nbytes <= _SMALL_MAX:
        capacity = ((nbytes + _SMALL_STEP - 1) // _SMALL_STEP) * _SMALL_STEP
        return ("s", capacity), capacity
    if nbytes <= _LARGE_MIN:
        capacity = 1 << (nbytes - 1).bit_length()          # 向上取 2 的幂
        return ("p", capacity), capacity
    return ("e", nbytes), nbytes                           # 精确大小桶


# ---------------------------------------------------------------------------
# 异常体系（FR-1.2 / FR-4.4 / FR-8.2）
# ---------------------------------------------------------------------------
class DevMemError(RuntimeError):
    """显存管理相关错误基类"""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class OutOfMemoryError(DevMemError):
    """设备显存不足"""


class InvalidArgumentError(DevMemError):
    """参数非法（dtype / 连续性 / 长度不匹配等）"""


class DeviceError(DevMemError):
    """设备相关错误（设备号非法、跨设备访问）"""


class BufferReleasedError(DevMemError):
    """使用了已释放的句柄（含双重释放）"""


class BufferLeasedError(DevMemError):
    """块上还有未失效的外部视图（cupy/torch 零拷贝包装），此时释放会造成悬垂（FR-6.2）"""


class MemoryCorruptionError(DevMemError):
    """debug 越界哨兵被破坏：肯定发生了越界写（FR-9.2）"""


class AllocatorClosedError(DevMemError):
    """分配器已关闭"""


class LibraryNotFoundError(DevMemError):
    """找不到 dev_memory 动态库"""


_CODE_TO_EXC = {
    ENE_ERR_OOM: OutOfMemoryError,
    ENE_ERR_INVALID: InvalidArgumentError,
    ENE_ERR_DEVICE: DeviceError,
    ENE_ERR_CUDA: DevMemError,
    ENE_ERR_NODEVICE: DevMemError,
}


def _raise_for(code: int, message: str) -> None:
    exc = _CODE_TO_EXC.get(code, DevMemError)
    raise exc(f"{message} (code={code})", code=code)


def stream_ptr(stream: Optional[int]) -> int:
    """流句柄规范化：None / 0 -> 0（默认流）

    注意：这里的流是 CUDA 裸指针（int）。若要与 cupy 的流互操作，
    传 `cp.cuda.get_current_stream().ptr` 即可，两者是同一个东西。
    """
    return int(stream) if stream else 0


# ---------------------------------------------------------------------------
# ctypes 封装
# ---------------------------------------------------------------------------
class _Lib:
    """dev_memory.dll 的签名绑定"""

    def __init__(self, path: Path):
        self.path = path
        self.dll = ctypes.CDLL(str(path))
        self.dll.ene_mem_init.argtypes = [ctypes.c_int]
        self.dll.ene_mem_init.restype = ctypes.c_int
        self.dll.ene_mem_available.argtypes = []
        self.dll.ene_mem_available.restype = ctypes.c_int
        self.dll.ene_mem_device.argtypes = []
        self.dll.ene_mem_device.restype = ctypes.c_int
        # ---- 阶段5：线程级设备绑定 ----
        self.dll.ene_mem_set_device.argtypes = [ctypes.c_int]
        self.dll.ene_mem_set_device.restype = ctypes.c_int
        for name in ("ene_mem_current_device", "ene_mem_cached_device"):
            fn = getattr(self.dll, name)
            fn.argtypes = []
            fn.restype = ctypes.c_int
        self.dll.ene_mem_device_count.argtypes = []
        self.dll.ene_mem_device_count.restype = ctypes.c_int
        self.dll.ene_mem_last_code.argtypes = []
        self.dll.ene_mem_last_code.restype = ctypes.c_int
        self.dll.ene_mem_last_error.argtypes = []
        self.dll.ene_mem_last_error.restype = ctypes.c_char_p
        self.dll.ene_mem_alloc.argtypes = [ctypes.c_size_t]
        self.dll.ene_mem_alloc.restype = ctypes.c_void_p
        self.dll.ene_mem_free.argtypes = [ctypes.c_void_p]
        self.dll.ene_mem_free.restype = ctypes.c_int
        for name in ("ene_mem_upload", "ene_mem_download"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        self.dll.ene_mem_sync.argtypes = [ctypes.c_void_p]
        self.dll.ene_mem_sync.restype = ctypes.c_int
        for name in ("ene_mem_alloc_calls", "ene_mem_free_calls",
                     "ene_mem_upload_bytes", "ene_mem_download_bytes",
                     "ene_mem_host_alloc_calls"):
            fn = getattr(self.dll, name)
            fn.argtypes = []
            fn.restype = ctypes.c_ulonglong

        # ---- 阶段3：流 / 事件 / pinned 主机内存 / 异步传输 ----
        for name in ("ene_mem_stream_create", "ene_mem_event_create",
                     "ene_mem_event_create_timed", "ene_mem_host_alloc"):
            fn = getattr(self.dll, name)
            fn.argtypes = [] if name.startswith("ene_mem_stream") or "event_create" in name \
                else [ctypes.c_size_t]
            fn.restype = ctypes.c_void_p
        for name in ("ene_mem_stream_destroy", "ene_mem_event_destroy",
                     "ene_mem_event_sync", "ene_mem_event_done", "ene_mem_host_free"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
        for name in ("ene_mem_event_record", "ene_mem_event_wait"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            fn.restype = ctypes.c_int
        self.dll.ene_mem_event_elapsed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.dll.ene_mem_event_elapsed.restype = ctypes.c_float
        self.dll.ene_mem_is_pinned.argtypes = [ctypes.c_void_p]
        self.dll.ene_mem_is_pinned.restype = ctypes.c_int
        for name in ("ene_mem_upload_async", "ene_mem_download_async"):
            fn = getattr(self.dll, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            fn.restype = ctypes.c_int

    # 统一的「取错误信息并抛异常」helper
    def _check(self, code: int, what: str) -> None:
        if code != ENE_OK:
            detail = self.dll.ene_mem_last_error()
            detail = detail.decode("utf-8", "replace") if detail else ""
            _raise_for(code, f"{what} 失败: {detail}")

    def last_error(self) -> tuple[int, str]:
        code = int(self.dll.ene_mem_last_code())
        msg = self.dll.ene_mem_last_error()
        return code, (msg.decode("utf-8", "replace") if msg else "")


def load_library(lib_dir: Optional[Path | str] = None) -> Optional[_Lib]:
    """加载 dev_memory 库；文件不存在时返回 None（由调用方决定是否回退）"""
    path = Path(lib_dir or DEFAULT_LIB_DIR) / f"{LIB_NAME}.{EXT}"
    if not path.exists():
        return None
    return _Lib(path)


# ---------------------------------------------------------------------------
# 块表
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class _BlockInfo:
    """一块显存的元数据（主机侧，FR-1.4）；设备端不加块头，保证指针对齐

    `eq=False` 是故意的：让 `list.remove` 走**身份比较**。
    不同块的 ptr 虽然不同，但其他字段可能一致，用值比较会删错。
    """

    ptr: int
    nbytes: int                                   # 调用方请求的字节数
    capacity: int                                 # 实际分配到的字节数（桶容量 ≥ nbytes）
    dtype: np.dtype
    device: int                                   # -1 表示主机回退模式
    name: str
    released: bool = False
    host: Optional[bytearray] = None              # 主机回退模式的后备内存
    tag: int = 0                                  # 便于在泄漏清单里区分
    cls: Optional[BucketKey] = None               # size-class 桶 key
    free_seq: int = 0                             # 入池顺序（FIFO 驱逐用）
    free_time: float = 0.0                        # 入池时刻（trim(older_than=) 用）
    # 外部视图租约（阶段4）：cupy/torch 数组活着期间，本块不允许被池回收（FR-6.2）。
    # 存 weakref 而不是直接存标记对象 —— 直接存会形成「块→标记→块」的引用环，
    # 标记永远死不了，租约也就永远不释放。
    leases: set = field(default_factory=set)
    # 跨流复用保护（阶段5，FR-7.1）：最后使用本块的流；复用前要等它。
    last_stream: int = 0
    guard_event: int = 0                          # 已释放但目标流可能还在用时的守卫事件
    # debug 越界哨兵（阶段5，FR-9.2）：真实分配起点与占用字节数（0 = 等于 ptr/capacity）
    base_ptr: int = 0
    physical: int = 0

    @property
    def alloc_ptr(self) -> int:
        """真正要归还给驱动的指针（debug 哨兵下含头部保护区）"""
        return self.base_ptr or self.ptr

    @property
    def alloc_bytes(self) -> int:
        """真正从驱动/GC 拿到的字节数（debug 哨兵下含头尾保护区）"""
        return self.physical or self.capacity

    @property
    def size(self) -> int:
        """按 dtype 计的元素个数（13 个算子都按 float32 解释）"""
        return self.nbytes // self.dtype.itemsize

    @property
    def wasted(self) -> int:
        """桶取整带来的内部碎片"""
        return self.capacity - self.nbytes

    def describe(self) -> str:
        where = "host" if self.device < 0 else f"cuda:{self.device}"
        bucket = "直通" if self.cls is None else f"桶{self.cls[0]}/{self.cls[1]}"
        return (f"#{self.tag:<4} {self.name:<12} ptr=0x{self.ptr:012x} "
                f"请求 {self.nbytes:>9} B / 实占 {self.capacity:>9} B "
                f"({self.size} × {self.dtype}) {where} {bucket}")


# ---------------------------------------------------------------------------
# debug 越界哨兵（阶段5，FR-9.2）
#
# 仅在 DeviceAllocator(debug=True) 时启用：真实分配比用户请求的多 2×256 B，
# 用户指针居中。首尾各写一段魔术字：release 时有一侧对不上，就说明有人越界写了。
# 已损坏的块**绝不回池**，直接还给驱动 —— 脏数据不能流到下一个使用者手里。
#
# 读写必须走 `ene_mem_upload/download`：设备内存不能由主机直接 memmove。
# 具体实现见 `DeviceAllocator._write_guards / _check_guards`。
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 外部视图租约（阶段4，FR-6.2）
#
# 问题：把本池的块零拷贝交给 cupy 之后，cupy 数组只记得一个裸指针 ——
# 如果这时块被归还到池里并复用，那个数组就会静默地读到/写错别的张量。
#
# 解法：用 cupy 自己保证的引用链做「租约」
#
#     cp.ndarray ──► MemoryPointer ──► UnownedMemory ──► owner(=_Lease)
#
# `owner` 只是让 `_Lease` 活到数组被回收为止。我们同时在块上登记一个**弱引用 +
# 回调**：弱引用的目标（`_Lease`）一死，回调就把租约摘掉，最后一个租约消失时
# 才真正回收那块显存（见 `DeviceAllocator._finish_deferred`）。
#
# 用「弱引用 + 回调」而不是 WeakSet：回调是「最后一个视图消失」的唯一下手点。
# ---------------------------------------------------------------------------
class _Lease:
    """零负载的租约标记：唯一职责是「活到外部数组被回收为止」"""

    __slots__ = ("__weakref__",)


class _LeaseWatch:
    """挂在 `weakref.ref` 上的回调：租约死亡 → 摘牌子 → 可能触发真正的回收"""

    __slots__ = ("alloc", "info")

    def __init__(self, alloc: "DeviceAllocator", info: _BlockInfo):
        self.alloc = alloc
        self.info = info

    def __call__(self, ref: "weakref.ref") -> None:      # pragma: no cover - 由 GC 触发
        info = self.info
        info.leases.discard(ref)
        if not info.leases:
            self.alloc._finish_deferred(info)


# ---------------------------------------------------------------------------
# CUDA Array Interface（阶段4，FR-6.1 / FR-6.4）
# ---------------------------------------------------------------------------
def cupy_available() -> bool:
    """cupy 是否可导入（不缓存，便于测试里动态屏蔽/恢复）"""
    try:
        import cupy                                        # noqa: F401
    except Exception:
        return False
    return True


def require_cupy(what: str):
    """拿到 cupy 模块，拿不到就给一条能看懂的错误（NFR-6：核心路径不依赖 cupy）"""
    try:
        import cupy
    except Exception as exc:                               # pragma: no cover
        raise ImportError(
            f"{what} 需要 cupy；本模块的核心功能（分配/池/传输/算子）不依赖 cupy，"
            f"只有这条互操作路径需要") from exc
    return cupy


def cai_ptr(x: Any, *, what: str = "array", require_float32: bool = True) -> tuple[int, int]:
    """从任何实现了 CUDA Array Interface 的对象取 `(设备指针, 字节数)`

    **不依赖 cupy**：cupy / torch / numba / 第三方实现了该协议的对象都能走这条路
    （FR-6.1 / FR-6.4）。校验口径与 `cuda_ops.device_ptr` 一致，因为它们对应
    同一组前置条件：kernel 只按 float32 解释内存，且 `ptr` 必须指向连续段。
    """
    cai = getattr(x, "__cuda_array_interface__", None)
    if cai is None:
        raise TypeError(f"{what}: 需要实现了 __cuda_array_interface__ 的对象，"
                        f"得到 {type(x).__name__}")
    version = int(cai.get("version", 0))
    if version < 3:
        raise ValueError(f"{what}: CUDA Array Interface 版本 {version} 过低（需要 ≥ 3）")

    typestr = cai["typestr"]
    if require_float32 and typestr not in ("<f4", "=f4", "|f4"):
        raise TypeError(f"{what}: kernel 按 float32 解释内存，实际 typestr = {typestr!r}")

    strides = cai.get("strides")
    if strides is not None and tuple(strides) != (4,):
        raise ValueError(f"{what}: 需要 C 连续数组，strides = {strides!r}"
                         f"（非连续时 ptr 只指向第一段）")

    data = cai["data"]
    ptr = int(data[0] if isinstance(data, (tuple, list)) else data)
    if not ptr:
        raise ValueError(f"{what}: 指针为空")

    shape = cai["shape"]
    count = 1
    for dim in shape:
        count *= int(dim)
    return ptr, count * np.dtype(typestr).itemsize


# ---------------------------------------------------------------------------
# 锁页（pinned）主机内存（阶段3）
# ---------------------------------------------------------------------------
class PinnedBuffer:
    """一块锁页（pinned）主机内存

    `.array` 是它上面的 numpy 视图（**零拷贝**）：可以先把数据算到这里，
    再交给 `upload_async` 做真正的异步 DMA；也可以作为 `download_async` 的目标。
    """

    __slots__ = ("_alloc", "_ptr", "nbytes", "dtype", "name", "array", "released",
                 "_backing")

    def __init__(self, alloc: "DeviceAllocator", ptr: int, nbytes: int,
                 dtype: Any, name: str, backing: Any = None):
        self._alloc = alloc
        self._ptr = ptr
        self.nbytes = nbytes
        self.dtype = np.dtype(dtype)
        self.name = name
        self.released = False
        self._backing = backing                      # 主机回退时保活 ctypes 数组
        # 用 ctypes 数组拿缓冲协议，再 zero-copy 地 view 成目标 dtype
        raw = (ctypes.c_char * nbytes).from_address(ptr)
        self.array = np.frombuffer(raw, dtype=np.uint8, count=nbytes).view(self.dtype)

    @property
    def ptr(self) -> int:
        if self.released:
            raise BufferReleasedError(f"pinned 块 {self.name!r} 已归还，不能再使用")
        return self._ptr

    @property
    def __array_interface__(self) -> dict:
        """让 numpy 零拷贝消费（`np.asarray(pin)` / `np.frombuffer(pin)`）

        与 `.array` 一样**不保活**：numpy 数组活得比本句柄久就会指向已归还的锁页内存。
        """
        if self.released:
            raise BufferReleasedError(f"pinned 块 {self.name!r} 已归还，不能再导出视图")
        return {
            "data": (self._ptr, False),
            "shape": (self.array.size,),
            "strides": None,
            "typestr": self.dtype.str,
            "version": 3,
        }

    def as_numpy(self) -> np.ndarray:
        """`.array` 的同义方法（显式一点，便于阅读）"""
        if self.released:
            raise BufferReleasedError(f"pinned 块 {self.name!r} 已归还，不能再访问")
        return self.array

    def release(self) -> None:
        """归还到 pinned 池（重复调用会报错）"""
        self._alloc._pinned_release(self, quiet=False)

    def __enter__(self) -> "PinnedBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._alloc._pinned_release(self, quiet=True)
        return False

    def __del__(self) -> None:                           # pragma: no cover
        try:
            self._alloc._pinned_release(self, quiet=True)
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "released" if self.released else "live"
        return f"<PinnedBuffer {self.name!r} {self.nbytes}B {state}>"


# ---------------------------------------------------------------------------
# 句柄
# ---------------------------------------------------------------------------
class DeviceBuffer:
    """设备显存句柄（阶段1：直通分配，无池）

    裸指针只在 `ptr` 属性上暴露一次，且会拒绝已释放的块 —— 这是「句柄化」的落点（D1）。
    """

    __slots__ = ("_alloc", "_info")

    def __init__(self, alloc: "DeviceAllocator", info: _BlockInfo):
        self._alloc = alloc
        self._info = info

    # ---- 元数据 ----
    @property
    def nbytes(self) -> int:
        """请求的字节数（上传/下载按它校验）"""
        return self._info.nbytes

    @property
    def capacity(self) -> int:
        """实际占用显存的字节数（桶容量 ≥ nbytes，差额是内部碎片）"""
        return self._info.capacity

    @property
    def size(self) -> int:
        return self._info.size

    @property
    def dtype(self) -> np.dtype:
        return self._info.dtype

    @property
    def device(self) -> int:
        """-1 表示主机回退模式"""
        return self._info.device

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def released(self) -> bool:
        return self._info.released

    @property
    def leases(self) -> int:
        """当前活着的外部视图（cupy/torch 零拷贝数组）个数（阶段4，FR-6.2）"""
        return len(self._info.leases)

    @property
    def allocator(self) -> "DeviceAllocator":
        """本块所属的分配器"""
        return self._alloc

    def assert_current_device(self, *, what: str = "算子") -> None:
        """校验本块就在当前设备上（FR-8.1 / FR-8.2）

        跨设备访问**必须报错**，绝不隐式拷贝 —— 静默拷贝会让性能与显存都失控，
        而且掩盖了调用方的逻辑错误。`strict_device=False` 时不做检查。
        """
        info = self._info
        if info.released:
            raise BufferReleasedError(f"块 {info.name!r} 已被释放，不能再使用")
        if info.device < 0 or not self._alloc._strict_device:
            return
        cur = self._alloc.current_device()
        if cur >= 0 and cur != info.device:
            raise DeviceError(
                f"{what}: 块 {info.name!r} 在第 {info.device} 张卡，当前设备是 {cur}"
                f"（不会隐式拷贝，请显式把数据搬到对应设备）")

    def _ene_note_stream(self, stream: int) -> None:
        """记录本块最后一次被哪个流使用（FR-7.1）

        由 `CudaOps.dev()` 在取指针时顺带调用。`cross_stream_reuse=False`（默认）
        时第一行就返回，热路径开销可忽略。
        """
        if self._alloc._cross_stream and stream and not self._info.released:
            self._info.last_stream = int(stream)

    @property
    def ptr(self) -> int:
        """裸设备指针。仅供下发 kernel 使用（FR-5.2，O(1)）。"""
        if self._info.released:
            raise BufferReleasedError(f"块 {self._info.name!r} 已被释放，不能再取指针")
        return self._info.ptr

    def __int__(self) -> int:
        return self.ptr

    def __repr__(self) -> str:
        state = "released" if self._info.released else "live"
        cap = "" if self._info.capacity == self._info.nbytes else f" cap={self._info.capacity}B"
        lease = f" leases={len(self._info.leases)}" if self._info.leases else ""
        return f"<DeviceBuffer {self._info.name!r} {self.nbytes}B{cap}{lease} {state}>"

    # ---- 传输 ----
    def upload(self, src: np.ndarray, *, stream: Optional[int] = None) -> "DeviceBuffer":
        """把主机数组拷进本块（H2D）。要求 dtype 一致、C 连续、字节数相等。"""
        self._alloc._upload(self._info, src, stream)
        return self

    def download(self, dst: np.ndarray, *, stream: Optional[int] = None) -> np.ndarray:
        """把本块拷回主机数组（D2H）。"""
        self._alloc._download(self._info, dst, stream)
        return dst

    def to_numpy(self, *, stream: Optional[int] = None) -> np.ndarray:
        """拷回一个新的 numpy 数组（便捷方法）"""
        out = np.empty(self._info.size, dtype=self._info.dtype)
        self.download(out, stream=stream)
        return out

    # ---- 异步传输（阶段3，只入队不等待）----
    def upload_async(self, src: np.ndarray, *, stream: Optional[int] = None) -> "DeviceBuffer":
        """异步 H2D：只入队，**不等完成**

        `src` 不是锁页内存时，会先拷进内部 pinned staging（主机内 memcpy）再异步发出；
        staging 登记为「在飞」，要到 `allocator.sync()` 或池回收时才复用。
        使用数据前必须 sync（或依赖同一流上的后续操作）。
        """
        self._alloc._upload_async(self._info, src, stream)
        return self

    def download_async(self, dst: np.ndarray, *, stream: Optional[int] = None) -> np.ndarray:
        """异步 D2H：只入队，不等完成

        目标**必须是锁页内存**（用 `allocator.pinned(...)` 的 `.array`）。
        往 pageable 内存做异步 D2H 实际上并不会异步，这里直接报错而不是骗你。
        """
        self._alloc._download_async(self._info, dst, stream)
        return dst

    # ---- 生态互操作（阶段4，FR-6）----
    @property
    def __cuda_array_interface__(self) -> dict:
        """把本块暴露成 CUDA Array Interface v3（FR-6.1）

        消费方（`cp.asarray(buf)`、`torch.as_tensor(buf, device='cuda')`、
        `numba.cuda.as_cuda_array(buf)`）拿到这个字典后零拷贝建数组。

        **这是「瞬时借出」，不登记租约**：字典交给对方那一刻起，我们就不知道
        对方会把数组留多久。所以

        · 想写出 `cp.asarray(buf)` 这种一次性表达式 —— 用它，够用且零成本；
        · 想让数组活得比本句柄久 —— 请用 `as_cupy()`，它会登记租约阻止池回收。

        分配器上设 `strict_interop=True` 可以把这个「瞬时借出」直接禁掉，
        强制所有外部视图都走 `as_cupy()`（长生命周期流水线建议打开）。

        `shape` 取**请求的** `nbytes` 而非 `capacity`：桶取整的富余部分不外泄，
        外部视图写不进那块空隙。`stream=0` 声明「本块与 legacy 默认流有序」，
        用自定义流的调用方应改用 `as_cupy(stream=...)`。
        """
        info = self._info
        if info.released:
            raise BufferReleasedError(
                f"块 {info.name!r} 已被释放，不能再导出 CUDA Array Interface")
        if info.device < 0:
            raise DeviceError(
                f"块 {info.name!r} 在主机回退模式（无 CUDA），无法导出设备数组接口")
        if self._alloc._strict_interop:
            raise BufferLeasedError(
                f"分配器开启了 strict_interop，拒绝「瞬时借出」块 {info.name!r}；"
                f"请改用 as_cupy()（会登记租约，可安全长期持有）")
        return {
            "shape": (info.size,),
            "strides": None,
            "typestr": info.dtype.str,
            "data": (info.ptr, False),
            "version": 3,
            "stream": self._alloc._cai_stream,
        }

    def as_cupy(self, *, stream: Optional[int] = None) -> Any:
        """零拷贝包装成 `cupy.ndarray`，**并登记租约**（FR-6.2）

        返回的数组活着的时候本块不会被池回收 —— 哪怕 Python 侧已经把 `DeviceBuffer`
        删掉了（这时块进入「延迟回收」：句柄立即失效，但显存要等最后一个外部视图
        死亡才归还池）。数组被 GC 后租约自动解除，无需手动清理。

        `stream` 用来声明「本块的数据与哪个流有序」，会写进数组的流语义里；
        默认取分配器的 `cai_stream`（0 = legacy 默认流）。
        """
        cp = require_cupy("DeviceBuffer.as_cupy()")
        self._alloc._ensure_open()
        info = self._info
        if info.released:
            raise BufferReleasedError(f"块 {info.name!r} 已被释放，不能再包装")
        if info.device < 0:
            raise DeviceError(
                f"块 {info.name!r} 在主机回退模式（无 CUDA），无法包装成 cupy 数组")

        sid = self._alloc._cai_stream if stream is None else int(stream)
        lease = _Lease()
        info.leases.add(weakref.ref(lease, _LeaseWatch(self._alloc, info)))
        # UnownedMemory 是「别人的内存」：它不会去 free，只负责把 owner 保活 ——
        # 这正是我们需要的行为，单一所有者原则（D5）仍然由分配器履行。
        mem = cp.cuda.UnownedMemory(info.ptr, info.nbytes, lease, info.device)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray((info.size,), dtype=info.dtype, memptr=memptr)
        if sid:
            # 非默认流：交给 cupy 记录，后续跨流使用时它会插入事件等待
            try:
                arr.data.mem._stream = sid                 # type: ignore[attr-defined]
            except Exception:
                pass
        return arr

    # ---- 生命周期 ----
    def release(self, *, force: bool = False) -> None:
        """显式归还。重复调用会报错（FR-4.4 双重释放检测）。

        还有外部视图（`as_cupy()` 出来的 cupy 数组）活着时**默认报错而不是照做**：
        照做会让那些数组指向已被复用的显存，是最难查的一类静默错误（FR-6.2）。
        `force=True` 表示「我知道那些视图已经不用了」，此时照常归还。
        """
        self._alloc._release(self._info, quiet=False, force=force)

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # FR-4.3：异常路径也要回收
        self._alloc._release(self._info, quiet=True)
        return False

    def __del__(self) -> None:
        # 兜底自动回收；__del__ 不能抛异常
        try:
            self._alloc._release(self._info, quiet=True)
        except Exception:                            # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# 分配器
# ---------------------------------------------------------------------------
_LIVE_ALLOCATORS: "weakref.WeakSet[DeviceAllocator]" = weakref.WeakSet()
_ATEXIT_REGISTERED = False


def _atexit_check() -> None:                         # FR-4.5 泄漏自证
    for alloc in list(_LIVE_ALLOCATORS):
        if alloc.live_blocks:
            try:
                print(f"[dev_memory] 进程退出时仍有未释放显存:\n{alloc.leak_report()}")
            except Exception:                        # pragma: no cover
                pass


class DeviceAllocator:
    """设备显存分配器（阶段1：直通分配，无池）

    每个实例对应一张卡，并持有一张「块表」作为句柄的权威登记处。
    """

    def __init__(self, device: int = 0, *, lib_dir: Optional[Path | str] = None,
                 force_host: bool = False, name: str = "allocator",
                 pool: bool = True, pool_max_block: Optional[int] = None,
                 pool_max_bytes: Optional[int] = None,
                 chunk_bytes: int = 16 << 20,
                 strict_interop: bool = False,
                 cai_stream: int = 0,
                 strict_device: bool = True,
                 cross_stream_reuse: bool = False,
                 debug: bool = False):
        """
        Args:
            pool: 是否启用内存池；False 时退化为直通分配（每次 cudaMalloc）
            pool_max_block: 超过此容量的块不进池（None = 不限）
            pool_max_bytes: 池中空闲内存总量上限，超出按 FIFO 驱逐最旧的（None = 不限）
            chunk_bytes: 单次传输最大字节数，超过则自动分片
            strict_interop: True 时禁止 `__cuda_array_interface__` 的「瞬时借出」
            cai_stream: 写进 `__cuda_array_interface__['stream']` 的流（0 = 默认流）
            strict_device: True 时算子下发前校验「块所在设备 == 当前设备」（FR-8.1/8.2）
            cross_stream_reuse: True 时记录每块最后使用的流，跨流复用前用事件建立依赖（FR-7.1/7.3）
            debug: True 时启用头尾越界哨兵（每块额外 2×256 B，release 时校验，FR-9.2）
        """
        global _ATEXIT_REGISTERED

        self.name = name
        self._lock = threading.RLock()
        self._blocks: dict[int, _BlockInfo] = {}
        self._closed = False
        self._counter = 0
        self._peak_live_bytes = 0
        self._live_bytes = 0
        # 阶段4：已「逻辑释放」但还挂着外部视图的块，等租约清零后真正回收
        self._pending_release: dict[int, _BlockInfo] = {}
        self._strict_interop = bool(strict_interop)
        self._cai_stream = int(cai_stream)
        # 阶段5：设备绑定 / 跨流复用 / debug 哨兵
        self._strict_device = bool(strict_device)
        self._cross_stream = bool(cross_stream_reuse)
        self._debug = bool(debug)
        self._bound = -1                             # 本线程已绑定的设备（缓存）
        self._stream_guard: list[_BlockInfo] = []    # 已释放、但目标流可能还在用的块

        # ---- 内存池（阶段2）----
        self._pool = bool(pool)
        self._pool_max_block = None if pool_max_block is None else int(pool_max_block)
        self._pool_max_bytes = None if pool_max_bytes is None else int(pool_max_bytes)
        self._free: dict[BucketKey, list[_BlockInfo]] = {}
        self._free_bytes = 0
        self._free_seq = 0

        # ---- pinned staging 池（阶段3）----
        self._chunk_bytes = int(chunk_bytes)
        # nbytes -> [(ptr, nbytes, backing)]；存原始元组而不是 PinnedBuffer 对象，
        # 这样每次取出都是全新的句柄，旧句柄永远不会「复活」（与 DeviceBuffer 同理）
        self._pinned_free: dict[int, list[tuple[int, int, Any]]] = {}
        self._pinned_inflight: list[tuple[PinnedBuffer, int, int]] = []   # (buf, stream, event)
        self._pinned_bytes = 0

        # 本地统计（统一主机/设备两种模式；C 侧计数另有一份用于交叉验证）
        self._stat_allocs = 0
        self._stat_frees = 0
        self._stat_up_bytes = 0
        self._stat_down_bytes = 0
        self._stat_pool_hits = 0
        self._stat_pool_misses = 0
        self._stat_recycled = 0
        self._stat_evicted = 0
        self._stat_trimmed = 0
        self._stat_pinned_allocs = 0
        self._stat_pinned_hits = 0
        self._stat_async_up = 0
        self._stat_async_down = 0
        self._stat_deferred = 0                        # 阶段4：进入延迟回收的次数
        self._stat_lease_blocked = 0                   # 阶段4：因租约未清被拒绝释放的次数
        self._stat_stream_guards = 0                   # 阶段5：进入跨流守卫的块数
        self._stat_cross_stream_waits = 0              # 阶段5：因跨流复用插入的事件等待数
        self._stat_guard_failures = 0                  # 阶段5：越界哨兵被破坏的次数

        self._lib: Optional[_Lib] = None
        self._host_mode = True
        self._device = -1

        if not force_host:
            self._lib = load_library(lib_dir)
            if self._lib is not None:
                code = int(self._lib.dll.ene_mem_init(int(device)))
                if code == ENE_OK:
                    self._host_mode = False
                    # init 已把 device<0 归一为 0，并且已经 cudaSetDevice 到了那张卡
                    self._device = max(int(device), 0)
                    self._bound = self._device
                elif code == ENE_ERR_NODEVICE:
                    # FR-10.1：无 CUDA 环境时回退到主机内存，API 不崩
                    self._host_mode = True
                    self._device = -1
                else:
                    err_code, msg = self._lib.last_error()
                    _raise_for(err_code, f"ene_mem_init(device={device}) 失败: {msg}")

        if not _ATEXIT_REGISTERED:
            atexit.register(_atexit_check)
            _ATEXIT_REGISTERED = True
        _LIVE_ALLOCATORS.add(self)

    # ---- 能力查询 ----
    @property
    def available(self) -> bool:
        """True 表示在真设备显存上分配；False 表示主机回退模式（FR-10.1）"""
        return not self._host_mode

    @property
    def device(self) -> int:
        return self._device

    @property
    def mode(self) -> str:
        return "host-fallback" if self._host_mode else f"cuda:{self._device}"

    # ---- 设备绑定（阶段5，FR-8）----
    def _bind_device(self) -> None:
        """把**本线程**绑定到本分配器的设备

        CUDA 的 current device 是**线程局部**的，而同一线程可能交替使用两个分配器
        （比如先 device 0、再 device 1、又回到 device 0）。

        这里**必须每次向 C 层确认**，不能用 Python 侧的 `self._bound` 做短路：
        另一个 allocator 可能已经把这个线程绑到了别的卡，而 `self._bound` 仍然
        记着旧值 —— 那样就会跳过绑定，退回「静默分错卡」。

        真正的判重在 C 层：`g_current` 是线程局部的，命中时只是比较一个 int。
        """
        if self._host_mode or self._lib is None:
            return
        code = int(self._lib.dll.ene_mem_set_device(int(self._device)))
        self._lib._check(code, f"绑定设备 {self._device}")
        self._bound = self._device                   # 仅用于诊断/统计

    def current_device(self) -> int:
        """当前线程**真实**的设备号（向 CUDA 查询，主机回退模式返回 -1）

        用于 FR-8.1/8.2 的跨设备校验，不要放在热路径上。
        """
        if self._host_mode or self._lib is None:
            return -1
        return int(self._lib.dll.ene_mem_current_device())

    def cached_device(self) -> int:
        """本线程的**缓存**设备号（不调 CUDA；可能滞后于外部 `cudaSetDevice`）"""
        if self._host_mode or self._lib is None:
            return -1
        return int(self._lib.dll.ene_mem_cached_device())

    # ---- 分配 ----
    def _bucket_for(self, nbytes: int) -> tuple[Optional[BucketKey], int]:
        """决定这次分配走哪个桶（None = 不进池）"""
        if not self._pool:
            return None, nbytes
        key, capacity = _size_class(nbytes)
        if self._pool_max_block is not None and capacity > self._pool_max_block:
            return None, nbytes
        return key, capacity

    def alloc_bytes(self, nbytes: int, *, dtype: Any = np.uint8,
                    name: str = "buffer",
                    stream: Optional[int] = None) -> DeviceBuffer:
        """分配一块显存

        `stream` 只用于**跨流复用**：指定这块显存接下来将在这个流上使用，
        若复用的是被别的流用过的块，会插入一次 `cudaStreamWaitEvent`（主机不阻塞）。
        """
        self._ensure_open()
        if not isinstance(nbytes, (int, np.integer)) or nbytes <= 0:
            raise InvalidArgumentError(f"nbytes 必须是正整数，得到 {nbytes!r}")
        nbytes = int(nbytes)
        dt = np.dtype(dtype)
        key, capacity = self._bucket_for(nbytes)

        # ---- 阶段5：先把「流已跑完」的守卫块放回池（FR-7.1）----
        # 不放在这里的话，只要后续分配的都是别的尺寸（走不到 _take_guarded），
        # 已完成的守卫块就会一直挂在表里，等于那部分池容量永久不可用。
        # 默认配置（cross_stream_reuse=False）守卫表永远是空的，这里零成本。
        if self._stream_guard:
            self._sweep_stream_guard()

        # ---- 快路径：池命中（FR-2.1 / NFR-3，完全不碰 CUDA）----
        if key is not None:
            reused: Optional[_BlockInfo] = None
            with self._lock:
                bucket = self._free.get(key)
                if bucket:
                    old = bucket.pop()
                    if not bucket:
                        del self._free[key]
                    self._free_bytes -= old.capacity
                    self._stat_pool_hits += 1
                    reused = self._reuse_info(old, nbytes, dt, name, key)
            if reused is not None:
                if self._debug and reused.base_ptr:
                    self._write_guards(reused)       # 锁外写：不持锁调 CUDA（FR-11.2）
                return self._register(reused)

        # ---- 跨流复用（FR-7.1）：从守卫里取块，用事件建立依赖，主机不阻塞 ----
        if self._cross_stream and key is not None:
            taken = self._take_guarded(key)
            if taken is not None:
                old, ev = taken
                if ev:
                    self.wait_event(ev, stream_ptr(stream))
                    self.destroy_event(ev)
                    self._stat_cross_stream_waits += 1
                self._stat_pool_hits += 1
                info = self._reuse_info(old, nbytes, dt, name, key)
                if self._debug and info.base_ptr:
                    self._write_guards(info)
                return self._register(info)

        # ---- 慢路径：真正向驱动申请（cudaMalloc 约 0.25 ms/次）----
        self._stat_pool_misses += 1
        self._counter += 1
        info = _BlockInfo(ptr=0, nbytes=nbytes, capacity=capacity, dtype=dt,
                          device=-1, name=name, tag=self._counter, cls=key)
        guard = 2 * _GUARD_BYTES if self._debug else 0
        total = capacity + guard

        if self._host_mode:
            buf = bytearray(total)
            info.host = buf
            base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            info.device = -1
        else:
            assert self._lib is not None
            self._bind_device()
            raw = self._lib.dll.ene_mem_alloc(ctypes.c_size_t(total))
            if not raw:
                code, msg = self._lib.last_error()
                _raise_for(code, f"分配 {total} 字节失败: {msg}")
            if raw % 256 != 0:                       # FR-1.3 自检（cudaMalloc 本身保证 ≥256B）
                self._lib.dll.ene_mem_free(ctypes.c_void_p(raw))
                raise DevMemError(f"分配返回的指针未按 256B 对齐: 0x{raw:x}")
            base = int(raw)
            info.device = self._device

        if guard:
            # 用户指针居中，头尾各 256 B 哨兵；ptr 仍然是 256 B 对齐
            info.base_ptr = base
            info.physical = total
            info.ptr = base + _GUARD_BYTES
            self._write_guards(info)
        else:
            info.ptr = base

        return self._register(info)

    def _reuse_info(self, old: _BlockInfo, nbytes: int, dt: np.dtype,
                    name: str, key: BucketKey) -> _BlockInfo:
        """复用池中块：**新建** _BlockInfo，让旧句柄永远保持 released=True

        若就地把 old 的字段改回去，旧的 DeviceBuffer 会重新变成「可用」并指向新块 ——
        这是最典型的一类悬垂别名 bug。新建对象则旧句柄仍然会正确报错。
        """
        self._counter += 1
        info = _BlockInfo(ptr=old.ptr, nbytes=nbytes, capacity=old.capacity, dtype=dt,
                          device=old.device, name=name, host=old.host,
                          tag=self._counter, cls=key,
                          base_ptr=old.base_ptr, physical=old.physical)
        return info

    def _register(self, info: _BlockInfo) -> DeviceBuffer:
        with self._lock:
            self._blocks[info.ptr] = info
            self._stat_allocs += 1
            self._live_bytes += info.nbytes
            if self._live_bytes > self._peak_live_bytes:
                self._peak_live_bytes = self._live_bytes
        return DeviceBuffer(self, info)

    def alloc(self, size: int, *, dtype: Any = np.float32, name: str = "buffer",
              stream: Optional[int] = None) -> DeviceBuffer:
        """按元素个数分配。默认 float32 —— 与 13 个算子的内存解释一致。"""
        self._ensure_open()
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise InvalidArgumentError(f"size 必须是正整数，得到 {size!r}")
        dt = np.dtype(dtype)
        return self.alloc_bytes(int(size) * dt.itemsize, dtype=dt, name=name, stream=stream)

    def alloc_like(self, arr: np.ndarray, *, name: str = "buffer",
                   stream: Optional[int] = None) -> DeviceBuffer:
        """按 numpy 数组的形状/类型分配（不做任何拷贝）"""
        if not isinstance(arr, np.ndarray):
            raise InvalidArgumentError(f"需要 numpy 数组，得到 {type(arr).__name__}")
        return self.alloc(arr.size, dtype=arr.dtype, name=name, stream=stream)

    # ---- 传输 ----
    def _upload(self, info: _BlockInfo, src: np.ndarray, stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "upload")
        arr = _check_host_array(src, info, "upload")
        if self._host_mode:
            ctypes.memmove(info.ptr, arr.ctypes.data, arr.nbytes)
        else:
            assert self._lib is not None
            self._bind_device()
            code = int(self._lib.dll.ene_mem_upload(
                ctypes.c_void_p(info.ptr), ctypes.c_void_p(arr.ctypes.data),
                ctypes.c_size_t(arr.nbytes), ctypes.c_void_p(stream or 0)))
            self._lib._check(code, f"upload {arr.nbytes} 字节")
        self._stat_up_bytes += arr.nbytes

    def _download(self, info: _BlockInfo, dst: np.ndarray, stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "download")
        arr = _check_host_array(dst, info, "download")
        if self._host_mode:
            ctypes.memmove(arr.ctypes.data, info.ptr, arr.nbytes)
        else:
            assert self._lib is not None
            self._bind_device()
            code = int(self._lib.dll.ene_mem_download(
                ctypes.c_void_p(arr.ctypes.data), ctypes.c_void_p(info.ptr),
                ctypes.c_size_t(arr.nbytes), ctypes.c_void_p(stream or 0)))
            self._lib._check(code, f"download {arr.nbytes} 字节")
        self._stat_down_bytes += arr.nbytes

    def sync(self, stream: Optional[int] = None) -> None:
        """等待该流上所有工作完成，并回收已完成的 pinned staging（阶段3）

        阶段1/2 的 `upload`/`download` 自带同步，不需要调用；阶段3 的
        `upload_async`/`download_async` 之后**必须** sync 或依赖同流依赖。
        """
        if self._host_mode:
            return
        assert self._lib is not None
        self._bind_device()
        code = int(self._lib.dll.ene_mem_sync(ctypes.c_void_p(stream or 0)))
        self._lib._check(code, "sync")
        self._sweep_pinned(force_stream=stream_ptr(stream))
        self._sweep_stream_guard()

    # ---- 流与事件（阶段3）----
    def new_stream(self) -> int:
        """新建一条**非阻塞**流（返回裸指针）。用完请 destroy_stream。"""
        self._ensure_open()
        if self._host_mode:
            return 0
        assert self._lib is not None
        self._bind_device()
        ptr = self._lib.dll.ene_mem_stream_create()
        if not ptr:
            code, msg = self._lib.last_error()
            _raise_for(code, f"创建流失败: {msg}")
        return int(ptr)

    def destroy_stream(self, stream: int) -> None:
        if self._host_mode or not stream:
            return
        assert self._lib is not None
        self._bind_device()
        self._lib._check(int(self._lib.dll.ene_mem_stream_destroy(ctypes.c_void_p(stream))),
                         "destroy_stream")

    def create_event(self, *, timed: bool = False) -> int:
        """新建事件（默认关闭计时，更快）。用完请 destroy_event。"""
        if self._lib is None:
            raise DevMemError("主机回退模式下没有 CUDA 事件")
        self._bind_device()
        fn = self._lib.dll.ene_mem_event_create_timed if timed else self._lib.dll.ene_mem_event_create
        ev = fn()
        if not ev:
            code, msg = self._lib.last_error()
            _raise_for(code, f"创建事件失败: {msg}")
        return int(ev)

    def record_event_on(self, event: int, stream: Optional[int] = None) -> None:
        assert self._lib is not None
        self._bind_device()
        self._lib._check(int(self._lib.dll.ene_mem_event_record(
            ctypes.c_void_p(event), ctypes.c_void_p(stream or 0))), "record_event")

    def wait_event(self, event: int, stream: Optional[int] = None) -> None:
        """让 stream 上**后续**的工作都等该事件触发"""
        assert self._lib is not None
        self._bind_device()
        self._lib._check(int(self._lib.dll.ene_mem_event_wait(
            ctypes.c_void_p(event), ctypes.c_void_p(stream or 0))), "wait_event")

    def event_sync(self, event: int) -> None:
        """主机等该事件（会阻塞，不许在持锁时调）"""
        assert self._lib is not None
        self._bind_device()
        self._lib._check(int(self._lib.dll.ene_mem_event_sync(ctypes.c_void_p(event))),
                         "event_sync")

    def event_done(self, event: int) -> bool:
        """非阻塞查询事件是否已完成"""
        assert self._lib is not None
        self._bind_device()
        return bool(self._lib.dll.ene_mem_event_done(ctypes.c_void_p(event)))

    def destroy_event(self, event: int) -> None:
        if self._lib is None or not event:
            return
        self._bind_device()
        self._lib.dll.ene_mem_event_destroy(ctypes.c_void_p(event))

    def event_elapsed(self, start: int, end: int) -> float:
        """两个**计时事件**之间的 GPU 侧毫秒数（需用 create_event(timed=True) 创建）"""
        assert self._lib is not None
        self._bind_device()
        return float(self._lib.dll.ene_mem_event_elapsed(
            ctypes.c_void_p(start), ctypes.c_void_p(end)))

    # ---- pinned 主机内存（阶段3）----
    def pinned(self, size: int, *, dtype: Any = np.float32,
               name: str = "pinned") -> PinnedBuffer:
        """取一块锁页主机内存（按精确字节数池化复用，FR-3.2）

        `buf.array` 是它的 numpy 视图，可直接填数/读数，零拷贝。
        """
        self._ensure_open()
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise InvalidArgumentError(f"size 必须是正整数，得到 {size!r}")
        return self._pinned_acquire(int(size) * np.dtype(dtype).itemsize,
                                    np.dtype(dtype), name)

    def _pinned_acquire(self, nbytes: int, dtype: Any = np.uint8,
                        name: str = "pinned") -> PinnedBuffer:
        def take() -> Optional[PinnedBuffer]:
            bucket = self._pinned_free.get(nbytes)
            if not bucket:
                return None
            self._stat_pinned_hits += 1
            ptr, nb, backing = bucket.pop()
            if not bucket:
                del self._pinned_free[nbytes]
            return PinnedBuffer(self, ptr, nb, dtype, name, backing=backing)

        hit = take()
        if hit is not None:
            return hit

        # 池里没有 → 先扫一遍「在飞」的，可能已有完成的了（这样 pinned 用量自然有界）
        self._sweep_pinned()
        hit = take()
        if hit is not None:
            return hit

        if self._host_mode:
            raw = (ctypes.c_char * nbytes)()             # 主机模式：普通内存即可
            return PinnedBuffer(self, ctypes.addressof(raw), nbytes, dtype, name, backing=raw)

        assert self._lib is not None
        self._bind_device()
        ptr = self._lib.dll.ene_mem_host_alloc(ctypes.c_size_t(nbytes))
        if not ptr:
            code, msg = self._lib.last_error()
            _raise_for(code, f"分配 {nbytes} 字节锁页内存失败: {msg}")
        self._pinned_bytes += nbytes
        self._stat_pinned_allocs += 1
        return PinnedBuffer(self, int(ptr), nbytes, dtype, name)

    def _pinned_release(self, buf: PinnedBuffer, *, quiet: bool) -> None:
        if buf.released:
            if quiet:
                return
            raise BufferReleasedError(f"pinned 块 {buf.name!r} 已归还 —— 重复归还")
        buf.released = True
        entry = (buf._ptr, buf.nbytes, buf._backing)
        buf._backing = None
        with self._lock:
            if self._closed:
                drop = True
            else:
                self._pinned_free.setdefault(buf.nbytes, []).append(entry)
                drop = False
        if drop:
            self._free_pinned_native(*entry)

    def _free_pinned_native(self, ptr: int, nbytes: int, backing: Any) -> None:
        self._pinned_bytes -= nbytes
        if backing is not None:
            return                                       # 主机回退：交回 GC
        if self._lib is not None:
            self._bind_device()
            self._lib.dll.ene_mem_host_free(ctypes.c_void_p(ptr))

    def _sweep_pinned(self, force_stream: Optional[int] = None) -> int:
        """回收已完成的在飞 pinned 缓冲；force_stream 指定时无条件回收该流的"""
        if not self._pinned_inflight or self._lib is None:
            return 0
        remaining = []
        freed = 0
        for buf, sid, ev in self._pinned_inflight:
            if force_stream is not None:
                done = (sid == force_stream)
            else:
                done = self.event_done(ev)
            if done:
                self.destroy_event(ev)
                self._pinned_release(buf, quiet=True)
                freed += 1
            else:
                remaining.append((buf, sid, ev))
        self._pinned_inflight = remaining
        return freed

    # ---- 异步传输（阶段3）----
    def _issue_copy(self, dst_ptr: int, src_ptr: int, nbytes: int,
                    stream: Optional[int], *, up: bool) -> None:
        """分片下发拷贝（FR-3.3）：单次不超过 chunk_bytes"""
        assert self._lib is not None
        self._bind_device()
        s = ctypes.c_void_p(stream or 0)
        fn = self._lib.dll.ene_mem_upload_async if up else self._lib.dll.ene_mem_download_async
        what = "upload_async" if up else "download_async"
        off = 0
        while off < nbytes:
            n = min(self._chunk_bytes, nbytes - off)
            code = int(fn(ctypes.c_void_p(dst_ptr + off), ctypes.c_void_p(src_ptr + off),
                          ctypes.c_size_t(n), s))
            self._lib._check(code, f"{what} {n} 字节")
            off += n

    def _upload_async(self, info: _BlockInfo, src: np.ndarray,
                      stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "upload_async")
        arr = _check_host_array(src, info, "upload_async")
        nbytes = arr.nbytes

        if self._host_mode:
            ctypes.memmove(info.ptr, arr.ctypes.data, nbytes)
            self._stat_up_bytes += nbytes
            return

        assert self._lib is not None
        host_ptr = int(arr.ctypes.data)
        sid = stream_ptr(stream)
        if self._lib.dll.ene_mem_is_pinned(ctypes.c_void_p(host_ptr)):
            self._issue_copy(info.ptr, host_ptr, nbytes, stream, up=True)
        else:
            # pageable 源 -> 先落到 pinned staging（主机内 memcpy，约 10 GB/s），再异步 H2D
            pin = self._pinned_acquire(nbytes, np.uint8, "staging")
            ctypes.memmove(pin._ptr, host_ptr, nbytes)
            self._issue_copy(info.ptr, pin._ptr, nbytes, stream, up=True)
            ev = self.create_event()
            self.record_event_on(ev, stream)             # 记在 copy 流上
            self._pinned_inflight.append((pin, sid, ev))
        self._stat_up_bytes += nbytes
        self._stat_async_up += 1

    def _download_async(self, info: _BlockInfo, dst: np.ndarray,
                        stream: Optional[int]) -> None:
        self._ensure_open()
        self._ensure_live(info, "download_async")
        arr = _check_host_array(dst, info, "download_async")

        if self._host_mode:
            ctypes.memmove(arr.ctypes.data, info.ptr, arr.nbytes)
            self._stat_down_bytes += arr.nbytes
            return

        assert self._lib is not None
        dst_ptr = int(arr.ctypes.data)
        if not self._lib.dll.ene_mem_is_pinned(ctypes.c_void_p(dst_ptr)):
            raise InvalidArgumentError(
                "download_async 需要锁页（pinned）目标：请用 allocator.pinned(n) 拿 `.array`，"
                "或改用同步的 download()")
        self._issue_copy(dst_ptr, info.ptr, arr.nbytes, stream, up=False)
        self._stat_down_bytes += arr.nbytes
        self._stat_async_down += 1

    # ---- 双缓冲流水线（阶段3）----
    def double_buffer(self, size: int, *, dtype: Any = np.float32,
                      slots: int = 2, name: str = "pipe") -> "DoubleBuffer":
        """创建双缓冲流水线（FR-3.4）"""
        self._ensure_open()
        if self._host_mode:
            raise DevMemError("主机回退模式不支持双缓冲流水线（需要 CUDA 流与事件）")
        return DoubleBuffer(self, size, dtype=dtype, slots=slots, name=name)

    # ---- 生命周期 ----
    def _raw_copy(self, dev_ptr: int, data: bytes, *, up: bool) -> bytes:
        """小段主机↔设备拷贝（哨兵专用；同步语义）"""
        assert self._lib is not None
        self._bind_device()
        buf = ctypes.create_string_buffer(data, len(data))
        host = ctypes.c_void_p(ctypes.addressof(buf))
        dev = ctypes.c_void_p(dev_ptr)
        fn = self._lib.dll.ene_mem_upload if up else self._lib.dll.ene_mem_download
        args = (dev, host) if up else (host, dev)
        code = int(fn(args[0], args[1], ctypes.c_size_t(len(data)), ctypes.c_void_p(0)))
        self._lib._check(code, "越界哨兵拷贝")
        return buf.raw

    def _write_guards(self, info: _BlockInfo) -> None:
        """在用户区前后各写一段魔术字（必须先经 C 层拷贝：设备内存不能由 CPU 直接写）"""
        tail = info.base_ptr + _GUARD_BYTES + info.capacity
        if info.host is not None:                    # 主机回退：普通内存，直接写
            ctypes.memmove(info.base_ptr, _GUARD_MAGIC, _GUARD_BYTES)
            ctypes.memmove(tail, _GUARD_MAGIC, _GUARD_BYTES)
            return
        self._raw_copy(info.base_ptr, _GUARD_MAGIC, up=True)
        self._raw_copy(tail, _GUARD_MAGIC, up=True)

    def _check_guards(self, info: _BlockInfo) -> Optional[str]:
        """返回被破坏的位置描述（None = 完好）"""
        tail = info.base_ptr + _GUARD_BYTES + info.capacity
        if info.host is not None:
            head = ctypes.string_at(info.base_ptr, _GUARD_BYTES)
            back = ctypes.string_at(tail, _GUARD_BYTES)
        else:
            head = self._raw_copy(info.base_ptr, _GUARD_MAGIC, up=False)
            back = self._raw_copy(tail, _GUARD_MAGIC, up=False)
        bad = []
        if head != _GUARD_MAGIC:
            bad.append("头部")
        if back != _GUARD_MAGIC:
            bad.append("尾部")
        return "与".join(bad) if bad else None

    def _release(self, info: _BlockInfo, *, quiet: bool, force: bool = False) -> None:
        # 阶段4（FR-6.2）：还有外部视图活着时，绝不能把这块显存拿去复用
        if info.leases and not force:
            if quiet:
                # GC / with 退出路径：既然不能报错，也不能造成悬垂，就「延迟回收」
                self._defer_release(info)
                return
            with self._lock:
                self._stat_lease_blocked += 1
            raise BufferLeasedError(
                f"块 {info.name!r}(ptr=0x{info.ptr:x}) 还有 {len(info.leases)} 个"
                f"外部视图（as_cupy 出来的数组）未失效，现在释放会让它们指向已被复用的显存。"
                f"请先删除这些数组，或 release(force=True)")

        # 阶段5（FR-9.2）：debug 模式下先查越界哨兵
        corrupt = None
        if self._debug and info.base_ptr and not info.released:
            corrupt = self._check_guards(info)

        with self._lock:
            if info.released:
                if quiet:
                    return
                raise BufferReleasedError(
                    f"块 {info.name!r}(ptr=0x{info.ptr:x}) 已被释放 —— 双重释放")
            info.released = True
            self._blocks.pop(info.ptr, None)
            # 按身份删：force 释放后同一地址可能已被别的块占用，不能按 ptr 盲弹
            if self._pending_release.get(info.ptr) is info:
                del self._pending_release[info.ptr]
            self._stat_frees += 1
            self._live_bytes -= info.nbytes
            if corrupt:
                self._stat_guard_failures += 1

        if corrupt:
            # 脏内存绝不能流到下一个使用者：直接还给驱动，不进池
            self._free_native(info, quiet=True)
            if not quiet:
                raise MemoryCorruptionError(
                    f"块 {info.name!r}(ptr=0x{info.ptr:x}) 的{corrupt}哨兵被破坏，"
                    f"疑似越界写（请求 {info.nbytes} B / 桶容量 {info.capacity} B）。"
                    f"该块已归还驱动，不进入池")
            return

        # 阶段5（FR-7.1）：最后一个使用它的流可能还没跑完 → 挂事件，复用时再等
        if (self._cross_stream and info.last_stream and not self._closed
                and not self._host_mode and self._lib is not None):
            self._guard_stream(info)
            return

        self._pool_or_free(info)

    def _guard_stream(self, info: _BlockInfo) -> None:
        """块已逻辑释放，但最后使用它的流可能还在跑 → 挂个事件，不阻塞主机"""
        sid = info.last_stream
        info.last_stream = 0
        ev = self.create_event()
        self.record_event_on(ev, sid)
        with self._lock:
            info.guard_event = ev
            self._stream_guard.append(info)
            self._stat_stream_guards += 1

    def _take_guarded(self, key: BucketKey) -> Optional[tuple[_BlockInfo, int]]:
        """从跨流守卫里取一块同桶的块（调用方负责在外面插入事件等待）"""
        with self._lock:
            for i, blk in enumerate(self._stream_guard):
                if blk.cls == key:
                    self._stream_guard.pop(i)
                    ev = blk.guard_event
                    blk.guard_event = 0
                    return blk, ev
        return None

    def _sweep_stream_guard(self) -> int:
        """把已经跑完的守卫块放回池（查询事件，不阻塞；绝不持锁调 CUDA）"""
        if not self._stream_guard or self._lib is None:
            return 0
        with self._lock:
            pending = [(blk, blk.guard_event) for blk in self._stream_guard]
        done: list[_BlockInfo] = []
        for blk, ev in pending:
            if ev and self.event_done(ev):
                done.append(blk)
        cleared = 0
        for blk in done:
            with self._lock:
                try:
                    self._stream_guard.remove(blk)
                except ValueError:                   # 已被别的线程取走
                    continue
                ev, blk.guard_event = blk.guard_event, 0
            self.destroy_event(ev)
            self._pool_or_free(blk)
            cleared += 1
        return cleared

    def _pool_or_free(self, info: _BlockInfo) -> None:
        """把「已经逻辑释放」的块放回池里等复用，或直接归还驱动（FR-2.1）

        `_release` 与 `_finish_deferred` 共用；**不得在持锁时**调 `_free_native`（FR-11.2）。
        """
        victims: list[_BlockInfo] = []
        with self._lock:
            recycle = False
            if info.cls is not None and not self._closed:
                self._free_seq += 1
                info.free_seq = self._free_seq
                info.free_time = time.monotonic()
                self._free.setdefault(info.cls, []).append(info)
                self._free_bytes += info.capacity
                self._stat_recycled += 1
                recycle = True
                victims = self._evict_locked()        # FR-2.6：超上限则驱逐最旧的
        if not recycle:
            self._free_native(info, quiet=True)
        for victim in victims:
            self._free_native(victim, quiet=True)

    def _defer_release(self, info: _BlockInfo) -> None:
        """逻辑释放但暂不回收：等最后一个外部视图死亡（阶段4，FR-6.2）

        句柄立即失效（`released=True`），块从块表里摘出，但显存不还池也不还驱动。
        因为显存还没归还，同一地址不可能被再次分配出去，所以 `_pending_release`
        按 ptr 索引是安全的。
        """
        with self._lock:
            if info.ptr in self._pending_release:
                return
            if not info.released:
                info.released = True
                self._blocks.pop(info.ptr, None)
                self._stat_frees += 1
                self._stat_deferred += 1
                self._live_bytes -= info.nbytes
            self._pending_release[info.ptr] = info

    def _finish_deferred(self, info: _BlockInfo) -> None:
        """最后一个外部视图死亡时被 GC 回调触发（阶段4，FR-6.2）"""
        if info.leases:                                   # 还有别的视图，再等等
            return
        with self._lock:
            # 身份校验：本块可能早就被 force 释放、地址又被新块占用
            # （新块同样可能在延迟表中），按 ptr 盲弹会错杀
            if self._pending_release.get(info.ptr) is not info:
                return
            del self._pending_release[info.ptr]
        self._pool_or_free(info)

    def _free_native(self, info: _BlockInfo, *, quiet: bool) -> None:
        """真正归还给驱动/GC。**不得在持锁时调用**（FR-11.2：cudaFree 会阻塞）"""
        if info.host is not None:
            info.host = None                          # 交回 GC
            return
        if self._lib is None:
            return
        self._bind_device()
        code = int(self._lib.dll.ene_mem_free(ctypes.c_void_p(info.alloc_ptr)))
        if code != ENE_OK and not quiet:
            err_code, msg = self._lib.last_error()
            _raise_for(err_code, f"释放 0x{info.ptr:x} 失败: {msg}")

    def _evict_locked(self) -> list[_BlockInfo]:
        """池中空闲字节超上限时按 FIFO 挑出要归还的块（锁内只挑不 free）"""
        if self._pool_max_bytes is None or self._free_bytes <= self._pool_max_bytes:
            return []
        idle = sorted((b for bucket in self._free.values() for b in bucket),
                      key=lambda b: b.free_seq)
        victims: list[_BlockInfo] = []
        for block in idle:
            if self._free_bytes <= self._pool_max_bytes:
                break
            self._free[block.cls].remove(block)       # eq=False → 按身份比较
            self._free_bytes -= block.capacity
            self._stat_evicted += 1
            victims.append(block)
        for key in [k for k, v in self._free.items() if not v]:
            del self._free[key]
        return victims

    def trim(self, *, older_than: Optional[float] = None) -> dict[str, int]:
        """把池中的空闲块真正归还驱动（FR-2.5）

        `older_than=None` 全部归还；`older_than=30` 只归还已空闲 30 秒以上的块。
        """
        now = time.monotonic()
        victims: list[_BlockInfo] = []
        with self._lock:
            for key in list(self._free):
                keep = []
                for block in self._free[key]:
                    if older_than is None or (now - block.free_time) >= older_than:
                        self._free_bytes -= block.capacity
                        victims.append(block)
                    else:
                        keep.append(block)
                if keep:
                    self._free[key] = keep
                else:
                    del self._free[key]
            self._stat_trimmed += len(victims)
        for block in victims:
            self._free_native(block, quiet=True)
        return {"blocks": len(victims),
                "bytes": sum(b.capacity for b in victims)}

    @property
    def live_blocks(self) -> list[_BlockInfo]:
        with self._lock:
            return list(self._blocks.values())

    @property
    def free_blocks(self) -> list[_BlockInfo]:
        """池中空闲块（仅供检查/统计）"""
        with self._lock:
            return [b for bucket in self._free.values() for b in bucket]

    @property
    def deferred_blocks(self) -> list[_BlockInfo]:
        """已逻辑释放、但仍被外部视图（cupy/torch）引用而未回收的块（阶段4）"""
        with self._lock:
            return list(self._pending_release.values())

    def close(self) -> None:
        """释放本分配器名下所有设备显存与锁页内存，幂等

        会先把已知的流同步完，避免释放正在被 kernel 使用的显存。
        注意：**关闭时不再等待外部视图**（`close()` 就是「我彻底不用了」的声明），
        因此若还有 `as_cupy()` 出来的数组活着，关闭后它们即成为悬垂 —— 这种情况
        会打一行警告，而不是静默踩坑。
        """
        if not self._host_mode and self._lib is not None:
            for sid in {sid for _, sid, _ in self._pinned_inflight}:
                self._lib.dll.ene_mem_sync(ctypes.c_void_p(sid))
            self._lib.dll.ene_mem_sync(ctypes.c_void_p(0))

        with self._lock:
            if self._closed:
                return
            self._closed = True                       # 先置位：_release 便不再回池
            pending = list(self._blocks.values())
            pooled = [b for bucket in self._free.values() for b in bucket]
            deferred = list(self._pending_release.values())
            guarded = list(self._stream_guard)
            pinned = [t for bucket in self._pinned_free.values() for t in bucket]
            inflight = [(b._ptr, b.nbytes, b._backing) for b, _, _ in self._pinned_inflight]
            events = [ev for _, _, ev in self._pinned_inflight]
            self._free.clear()
            self._free_bytes = 0
            self._pending_release.clear()
            self._stream_guard = []
            self._pinned_free.clear()
            self._pinned_inflight = []
        stray = sum(len(b.leases) for b in deferred)
        if stray:
            print(f"[dev_memory / {self.name}] 警告: close() 时有 {stray} 个外部视图仍活着，"
                  f"它们引用的显存已归还驱动，继续使用会读写到别人的数据")
        for info in pending:
            self._release(info, quiet=True)           # 走正常回收路径（已关闭 → 直接 free）
        for info in pooled + deferred:
            self._free_native(info, quiet=True)
        for info in guarded:                          # 跨流守卫中的块（已逻辑释放）
            if info.guard_event:
                self.destroy_event(info.guard_event)
                info.guard_event = 0
            self._free_native(info, quiet=True)
        for ev in events:
            self.destroy_event(ev)
        for entry in pinned + inflight:
            self._free_pinned_native(*entry)
        _LIVE_ALLOCATORS.discard(self)

    def leak_report(self) -> str:
        live = self.live_blocks
        deferred = self.deferred_blocks
        with self._lock:
            free_blocks = sum(len(b) for b in self._free.values())
            free_bytes = self._free_bytes
        if live:
            total = sum(b.nbytes for b in live)
            lines = [f"[{self.name} / {self.mode}] 未释放块 {len(live)} 个，共 {total} 字节:"]
            lines += [f"  - {b.describe()}" for b in live]
            text = "\n".join(lines)
        else:
            text = f"[{self.name}] 无未释放块"
        if deferred:
            total = sum(b.nbytes for b in deferred)
            leases = sum(len(b.leases) for b in deferred)
            text += (f"\n  延迟回收中: {len(deferred)} 个块（{total} 字节）、"
                     f"外部视图 {leases} 个仍活着（已逻辑释放，显存未还池）")
        if free_blocks:
            text += (f"\n  提示: 池中另有 {free_blocks} 个空闲块（共 {free_bytes} 字节），"
                     f"可 trim() 归还驱动")
        return text

    # ---- 统计 ----
    def stats(self) -> dict[str, Any]:
        with self._lock:
            free_blocks = sum(len(b) for b in self._free.values())
            free_bytes = self._free_bytes
            idle = [b for bucket in self._free.values() for b in bucket]
            largest_free = max((b.capacity for b in idle), default=0)
            classes = {f"{k[0]}/{k[1]}": {"blocks": len(v), "bytes": sum(b.capacity for b in v)}
                       for k, v in sorted(self._free.items()) if v}
            hits, misses = self._stat_pool_hits, self._stat_pool_misses
            evicted, trimmed = self._stat_evicted, self._stat_trimmed
            pinned_free = sum(len(v) for v in self._pinned_free.values())
            pinned_inflight = len(self._pinned_inflight)
            deferred = list(self._pending_release.values())
            infos = list(self._blocks.values()) + deferred
            leased_blocks = sum(1 for b in infos if b.leases)
            live_leases = sum(len(b.leases) for b in infos)
            stream_guarded = len(self._stream_guard)
            strict_device = self._strict_device
            cross_stream = self._cross_stream
            debug_guards = self._debug

        requests = hits + misses
        out: dict[str, Any] = {
            "mode": self.mode,
            "device": self._device,
            "pool_enabled": self._pool,
            "alloc_calls": self._stat_allocs,
            "free_calls": self._stat_frees,
            "live_blocks": len(self._blocks),
            "live_bytes": self._live_bytes,
            "peak_live_bytes": self._peak_live_bytes,
            "upload_bytes": self._stat_up_bytes,
            "download_bytes": self._stat_down_bytes,
            # ---- 池（FR-2.4）----
            "pool_hits": hits,
            "pool_misses": misses,
            "hit_rate": (hits / requests) if requests else 0.0,
            "recycled_blocks": self._stat_recycled,
            "free_blocks": free_blocks,
            "free_bytes": free_bytes,
            "largest_free_block": largest_free,
            # 碎片率 = 1 - 最大可服务块 / 空闲总字节（见设计文档 5.1）
            "fragmentation": (1.0 - largest_free / free_bytes) if free_bytes else 0.0,
            "evicted_blocks": evicted,
            "trimmed_blocks": trimmed,
            "free_classes": classes,
            # ---- pinned staging / 异步（阶段3）----
            "pinned_allocs": self._stat_pinned_allocs,
            "pinned_hits": self._stat_pinned_hits,
            "pinned_free_blocks": pinned_free,
            "pinned_inflight": pinned_inflight,
            "pinned_bytes": self._pinned_bytes,
            "async_upload_calls": self._stat_async_up,
            "async_download_calls": self._stat_async_down,
            # ---- 生态互操作（阶段4，FR-6）----
            "strict_interop": self._strict_interop,
            "cai_stream": self._cai_stream,
            "leased_blocks": leased_blocks,
            "live_leases": live_leases,
            "deferred_blocks": len(deferred),
            "deferred_bytes": sum(b.nbytes for b in deferred),
            "defer_events": self._stat_deferred,
            "lease_blocked_releases": self._stat_lease_blocked,
            # ---- 多设备与并发（阶段5，FR-7 / FR-8 / FR-9.2 / FR-11）----
            "strict_device": strict_device,
            "cross_stream_reuse": cross_stream,
            "debug_guards": debug_guards,
            "stream_guarded_blocks": stream_guarded,
            "stream_guards": self._stat_stream_guards,
            "cross_stream_waits": self._stat_cross_stream_waits,
            "guard_failures": self._stat_guard_failures,
        }
        if self._lib is not None:
            # C 侧计数：用于交叉验证「热路径真的没再 cudaMalloc」（NFR-3）
            out["native_alloc_calls"] = int(self._lib.dll.ene_mem_alloc_calls())
            out["native_free_calls"] = int(self._lib.dll.ene_mem_free_calls())
            out["native_upload_bytes"] = int(self._lib.dll.ene_mem_upload_bytes())
            out["native_download_bytes"] = int(self._lib.dll.ene_mem_download_bytes())
            out["native_host_alloc_calls"] = int(self._lib.dll.ene_mem_host_alloc_calls())
        return out

    # ---- 内部校验 ----
    def _ensure_open(self) -> None:
        if self._closed:
            raise AllocatorClosedError("分配器已关闭，不能再使用")

    def _ensure_live(self, info: _BlockInfo, what: str) -> None:
        if info.released:
            raise BufferReleasedError(
                f"{what}: 块 {info.name!r} 已被释放，不能再使用")

    # ---- 上下文管理 ----
    def __enter__(self) -> "DeviceAllocator":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:                       # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 双缓冲流水线（阶段3）
# ---------------------------------------------------------------------------
class _PipelineSlot:
    """双缓冲流水线的一个槽"""

    __slots__ = ("_pipe", "index", "buffer", "copy_stream", "compute_stream",
                 "_ev_copy", "_ev_compute", "_used")

    def __init__(self, pipe: "DoubleBuffer", index: int, buffer: DeviceBuffer,
                 ev_copy: int, ev_compute: int):
        self._pipe = pipe
        self.index = index
        self.buffer = buffer
        self.copy_stream = pipe.copy_stream
        self.compute_stream = pipe.compute_stream
        self._ev_copy = ev_copy
        self._ev_compute = ev_compute
        self._used = False

    def upload_async(self, src: np.ndarray) -> "_PipelineSlot":
        """异步把 src 传进本槽，并保证本槽 compute_stream 上后续工作等它完成"""
        mem = self._pipe._alloc
        if self._used:
            # 复用本槽前，先等上一轮在本槽上的计算跑完，否则会覆写正在算的数据
            mem.wait_event(self._ev_compute, self.copy_stream)
        self.buffer.upload_async(src, stream=self.copy_stream)
        mem.record_event_on(self._ev_copy, self.copy_stream)
        mem.wait_event(self._ev_copy, self.compute_stream)
        self._used = True
        return self

    def done(self) -> None:
        """声明本槽本轮的计算已全部提交（记录事件，供下一轮复用前等待）"""
        self._pipe._alloc.record_event_on(self._ev_compute, self.compute_stream)

    def __repr__(self) -> str:
        return f"<Slot #{self.index} {self.buffer.name!r}>"


class DoubleBuffer:
    """双缓冲流水线：让「传第 k+1 块」与「算第 k 块」重叠（FR-3.4）

    两条**非阻塞**流 + 每槽一对事件：

        copy_stream    —— H2D 按序排队
        compute_stream —— kernel 按序排队

    同步规则（缺一不可）：

    * 在某槽上计算**之前**：让 `compute_stream` 等该槽本轮的 copy 事件；
    * 复用某槽**之前**：让 `copy_stream` 等该槽上一轮的 compute 事件。

    用法：

        with mem.double_buffer(count, dtype=np.float32, slots=2, name="pipe") as pipe:
            for chunk in chunks:
                slot = pipe.next()
                slot.upload_async(chunk)                    # 异步 H2D 到 slot.buffer
                ops.dev("add", slot.buffer, db, dout, stream=slot.compute_stream)
                slot.done()                                 # 提交完成事件
            pipe.sync()                                     # 等两条流跑完
    """

    def __init__(self, alloc: DeviceAllocator, size: int, *,
                 dtype: Any = np.float32, slots: int = 2, name: str = "pipe"):
        if slots < 2:
            raise InvalidArgumentError(f"双缓冲至少需要 2 个槽，得到 {slots}")
        self._alloc = alloc
        self.copy_stream = alloc.new_stream()
        self.compute_stream = alloc.new_stream()
        self._slots = [
            _PipelineSlot(self, i,
                          alloc.alloc(size, dtype=dtype, name=f"{name}[{i}]"),
                          alloc.create_event(), alloc.create_event())
            for i in range(slots)
        ]
        self._cursor = 0
        self._closed = False

    def next(self) -> _PipelineSlot:
        """轮转取下ー个槽（调用方需保证不会与仍在使用的槽冲突，槽数 ≥ 2 时自然成立）"""
        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._slots)
        return slot

    def sync(self) -> None:
        """等两条流都完成，并回收 staging"""
        for sid in (self.copy_stream, self.compute_stream):
            self._alloc.sync(sid)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sync()
        mem = self._alloc
        for slot in self._slots:
            mem.destroy_event(slot._ev_copy)
            mem.destroy_event(slot._ev_compute)
            slot.buffer.release()
        mem.destroy_stream(self.copy_stream)
        mem.destroy_stream(self.compute_stream)

    def __enter__(self) -> "DoubleBuffer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def _check_host_array(arr: Any, info: _BlockInfo, what: str) -> np.ndarray:
    """上传/下载前的三项校验 —— 和 dev 算子一样，宁可直接报错也不做隐式拷贝/错算"""
    if not isinstance(arr, np.ndarray):
        raise InvalidArgumentError(f"{what}: 需要 numpy 数组，得到 {type(arr).__name__}")
    if arr.dtype != info.dtype:
        raise InvalidArgumentError(
            f"{what}: dtype 不匹配（块为 {info.dtype}，数组为 {arr.dtype}）")
    if not arr.flags.c_contiguous:
        raise InvalidArgumentError(
            f"{what}: 需要 C 连续数组，请先 np.ascontiguousarray（那是显式拷贝）")
    if arr.nbytes != info.nbytes:
        raise InvalidArgumentError(
            f"{what}: 字节数不匹配（块为 {info.nbytes}，数组为 {arr.nbytes}）")
    return arr
