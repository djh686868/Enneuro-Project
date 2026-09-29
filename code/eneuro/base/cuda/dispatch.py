import os
from pathlib import Path
import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from .compiler import available, module

# cupy：完全使用 CuPy；rawmodule：严格要求自研 kernel；auto：kernel 失败时
# 回退到 CuPy，便于在 NVRTC 尚未配置好时仍能验证模型数值。
_BACKEND = os.environ.get("ENNEURO_CUDA_BACKEND", "cupy").lower()
_LAST = {"actual_backend": None, "fallback_reason": None, "last_error": None}
_LAUNCH_COUNTS = {}
_FUNCTIONS = {}
_EXTRA_FUNCTIONS = {}
# The CUDA-C Winograd kernel maps one thread to one output value and performs
# the channel reduction in that thread.  Keep that implementation for small
# correctness probes; large ResNet maps use the framework's vectorized CuPy
# paths, which have much better occupancy.  Raise this value explicitly when
# profiling the CUDA-C kernel itself.
_WINOGRAD_MAX_OUTPUT = int(os.environ.get("ENNEURO_WINOGRAD_MAX_OUTPUT", "4096"))
_RAW_REFERENCE_CONV_OUTPUT = int(os.environ.get("ENNEURO_RAW_REFERENCE_CONV_OUTPUT", "4096"))
_RAW_CUSTOM_ELEMENTWISE_MAX_SIZE = int(
    os.environ.get("ENNEURO_RAW_CUSTOM_ELEMENTWISE_MAX_SIZE", "4096")
)

_KERNEL_SOURCE_PATH = Path(__file__).with_name("sources") / "kernels.cu"
# RawModule 与 DLL 从同一份 kernels.cu 读取设备代码，避免两条路线的
# 数学公式、索引布局或边界行为随着维护逐渐分叉。
_SRC_FULL = _KERNEL_SOURCE_PATH.read_text(encoding="utf-8")
# Compile the larger Winograd/normalization source lazily, so basic operators
# do not pay its NVRTC register-allocation cost on every process startup.
_EXTRA_MARKER = "__device__ __forceinline__ float winograd_load"
if _EXTRA_MARKER in _SRC_FULL:
    _SRC, _EXTRA_SRC = _SRC_FULL.split(_EXTRA_MARKER, 1)
    _EXTRA_SRC = "#ifdef __CUDACC_RTC__\ntypedef long long int64_t;\n#else\n#include <stdint.h>\n#endif\n" + _EXTRA_MARKER + _EXTRA_SRC
else:
    _SRC, _EXTRA_SRC = _SRC_FULL, ""


def set_backend(mode):
    """切换当前进程的算子后端，不改变 Tensor 的公共 API。"""
    global _BACKEND
    mode = str(mode).lower()
    if mode not in {"cupy", "rawmodule", "auto", "extension"}:
        raise ValueError("backend must be cupy, rawmodule, extension, or auto")
    _BACKEND = mode


def get_backend():
    return _BACKEND


def diagnostics(reset=False):
    out = dict(_LAST)
    if reset:
        _LAST.update(actual_backend=None, fallback_reason=None, last_error=None)
    return out


def launch_counts(reset=False):
    """返回各自研 CUDA kernel 的发射次数，用于确认训练实际走了 RawModule。"""
    out = dict(_LAUNCH_COUNTS)
    if reset:
        _LAUNCH_COUNTS.clear()
    return out


def is_available(backend="rawmodule"):
    if cp is None or not available():
        return False
    if backend in ("rawmodule", "auto"):
        try:
            module(_SRC).get_function("relu_f32")
        except Exception:
            return False
    return True


def _raw_enabled():
    return cp is not None and available() and _BACKEND in ("rawmodule", "auto")


def _call(name, args, n):
    # 统一使用 256 threads/block；grid 采用 ceil(n/256)，最后一个 block
    # 通过 kernel 内的 i<n 判断处理尾部元素，避免越界写入。
    # 源码中均为 extern "C" 名称；整个 RawModule 仅编译一次，函数句柄也
    # 缓存在进程内，避免每种算子重复编译相同源码。
    kernel = _FUNCTIONS.get(name)
    if kernel is None:
        kernel = module(_SRC).get_function(name)
        _FUNCTIONS[name] = kernel
    kernel(((int(n)+255)//256,), (256,), args)
    _LAUNCH_COUNTS[name] = _LAUNCH_COUNTS.get(name, 0) + 1


def _call_extra(name, args, n):
    kernel = _EXTRA_FUNCTIONS.get(name)
    if kernel is None:
        if not _EXTRA_SRC:
            raise RuntimeError("extra CUDA source is unavailable")
        kernel = module(_EXTRA_SRC).get_function(name)
        _EXTRA_FUNCTIONS[name] = kernel
    kernel(((int(n)+255)//256,), (256,), args)
    _LAUNCH_COUNTS[name] = _LAUNCH_COUNTS.get(name, 0) + 1


def _unary(x, name, fallback):
    # 首版 kernel 只覆盖 contiguous float32。其他 dtype、空数组或不满足条件
    # 的输入交给 CuPy，保证框架原有广播和 dtype 语义不被破坏。
    if (not _raw_enabled() or x.dtype != cp.float32 or x.size == 0
            or int(x.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        _LAST.update(actual_backend="cupy", fallback_reason="backend_or_dtype")
        return fallback(x)
    try:
        x = cp.ascontiguousarray(x); y = cp.empty_like(x)
        _call(name, (x, y, np.int64(x.size)), x.size)
        _LAST.update(actual_backend="rawmodule", fallback_reason=None, last_error=None)
        return y
    except Exception as exc:
        _LAST.update(actual_backend="cupy", fallback_reason="kernel_error", last_error=repr(exc))
        if _BACKEND == "rawmodule":
            raise
        return fallback(x)


def _binary(a, b, name, fallback):
    # 二元 kernel 要求相同形状；广播场景保留 CuPy 实现，梯度仍由上层 sum_to 处理。
    if (not _raw_enabled() or a.dtype != cp.float32 or b.dtype != cp.float32
            or a.shape != b.shape or a.size == 0
            or int(a.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        _LAST.update(actual_backend="cupy", fallback_reason="backend_dtype_or_broadcast")
        return fallback(a, b)
    try:
        a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b); y = cp.empty_like(a)
        _call(name, (a, b, y, np.int64(a.size)), a.size)
        _LAST.update(actual_backend="rawmodule", fallback_reason=None, last_error=None)
        return y
    except Exception as exc:
        _LAST.update(actual_backend="cupy", fallback_reason="kernel_error", last_error=repr(exc))
        if _BACKEND == "rawmodule":
            raise
        return fallback(a, b)


def add_forward(a,b): return _binary(a,b,"add_f32",lambda x,y:x+y)
def sub_forward(a,b): return _binary(a,b,"sub_f32",lambda x,y:x-y)
def mul_forward(a,b): return _binary(a,b,"mul_f32",lambda x,y:x*y)
def div_forward(a,b): return _binary(a,b,"div_f32",lambda x,y:x/y)
def neg_forward(x): return _unary(x,"neg_f32",lambda a:-a)
def exp_forward(x): return _unary(x,"exp_f32",cp.exp)
def log_forward(x): return _unary(x,"log_f32",cp.log)
def sigmoid_forward(x): return _unary(x,"sigmoid_f32",lambda a:cp.tanh(a*.5)*.5+.5)

def bias_add_forward(x, bias):
    """Add a channel/feature bias to (N,C), (N,C,H,W), or matching arrays."""
    # NCHW 中 s=H*W，二维 Linear 中 s=1；kernel 用 (i/s)%C 找到当前通道。
    bias = cp.asarray(bias, dtype=x.dtype)
    if x.ndim == 2 and bias.ndim == 1 and x.shape[1] == bias.shape[0]:
        n, c, s = int(x.shape[0]), int(x.shape[1]), 1
    elif x.ndim == 4 and bias.ndim == 1 and x.shape[1] == bias.shape[0]:
        n, c, s = int(x.shape[0]), int(x.shape[1]), int(x.shape[2] * x.shape[3])
    else:
        # CuPy 的默认尾维广播不能把 (C,) 对齐到 NCHW 的第二维，显式
        # reshape 后再相加，确保回退路径与 CUDA kernel 语义一致。
        if x.ndim == 4 and bias.ndim == 1 and x.shape[1] == bias.shape[0]:
            return x + bias.reshape(1, -1, 1, 1)
        return x + bias
    if (not _raw_enabled() or x.dtype != cp.float32
            or int(x.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        return x + (bias.reshape((1, c) if x.ndim == 2 else (1, c, 1, 1)))
    try:
        x = cp.ascontiguousarray(x); bias = cp.ascontiguousarray(bias); y = cp.empty_like(x)
        _call("bias_f32", (x, bias, y, np.int64(x.size), np.int64(c), np.int64(s)), x.size)
        return y
    except Exception:
        if _BACKEND == "rawmodule": raise
        return x + bias.reshape((1, c) if x.ndim == 2 else (1, c, 1, 1))

def pow_forward(x, exponent):
    if (not _raw_enabled() or x.dtype != cp.float32
            or not np.isscalar(exponent)
            or int(x.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        return cp.power(x, exponent)
    try:
        x=cp.ascontiguousarray(x); y=cp.empty_like(x); _call("pow_f32",(x,y,np.float32(exponent),np.int64(x.size)),x.size); return y
    except Exception:
        if _BACKEND == "rawmodule": raise
        return cp.power(x, exponent)

def relu_forward(x): return _unary(x,"relu_f32",lambda a:cp.maximum(a,0))
def relu_backward(x,gy):
    if (not _raw_enabled() or x.dtype != cp.float32 or gy.dtype != cp.float32
            or int(x.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        return gy*(x>0)
    try:
        x,gy=cp.ascontiguousarray(x),cp.ascontiguousarray(gy); gx=cp.empty_like(x); _call("relu_bwd_f32",(x,gy,gx,np.int64(x.size)),x.size); return gx
    except Exception:
        if _BACKEND == "rawmodule": raise
        return gy*(x>0)

def im2col_forward(x,kernel,stride=1,pad=0,dilation=1,to_matrix=True):
    # 每个线程负责一个 (N,OH,OW,C,KH,KW) 位置；越界采样按当前框架约定填 0。
    kh,kw=(int(kernel),int(kernel)) if np.isscalar(kernel) else tuple(map(int,kernel)); sh,sw=(int(stride),int(stride)) if np.isscalar(stride) else tuple(map(int,stride)); ph,pw=(int(pad),int(pad)) if np.isscalar(pad) else tuple(map(int,pad)); dh,dw=(int(dilation),int(dilation)) if np.isscalar(dilation) else tuple(map(int,dilation))
    n,c,h,w=map(int,x.shape); oh=(h+2*ph-dh*(kh-1)-1)//sh+1; ow=(w+2*pw-dw*(kw-1)-1)//sw+1
    if not _raw_enabled() or x.dtype != cp.float32:
        from ..functions import im2col_array
        return im2col_array(x,(kh,kw),(sh,sw),(ph,pw),to_matrix=to_matrix,dilation=(dh,dw),xp=cp)
    col=cp.empty((n*oh*ow,c*kh*kw),dtype=x.dtype); _call("im2col_f32",(cp.ascontiguousarray(x),col,n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw),col.size)
    return col if to_matrix else col.reshape(n,oh,ow,c,kh,kw).transpose(0,3,4,5,1,2)


def _vectorized_conv2d_forward(x, w, b, stride, pad, dilation, standard_3x3):
    """Run the existing vectorized CuPy convolution for large tensors.

    Calling the public ``Conv2d`` function here would re-enter this dispatch
    module.  Calling its numerical path directly reuses the established
    Winograd, FFT, GEMM, and im2col implementations without recursion and
    without maintaining a second large-tensor convolution implementation.
    The path selector and cache are shared with ``Conv2d`` so RawModule and
    the CuPy baseline make the same algorithm choice for each tensor shape.
    """
    from ..functions import Conv2d as ConvFunction

    fn = ConvFunction(stride=stride, pad=pad, dilation=dilation)
    # Reuse the same cached/autotuned selector as the normal CuPy backend.
    # The former hand-written predicates always selected im2col for most
    # ResNet feature maps, while the CuPy route could select FFT/GEMM after
    # measuring the actual shape.  That silent path mismatch explains much of
    # the RawModule slowdown on full DonkeyCar batches.
    path = fn._get_forward_path(x, w, b)
    if path == "winograd":
        return fn.winograd_conv2d_forward(x, w, b)
    if path == "fft":
        return fn.fft_conv2d_forward(x, w, b)
    if path == "gemm":
        return fn.gemm_conv2d_forward(x, w, b)
    return fn.im2col_conv2d_forward(x, w, b)

def conv2d_forward(x,w,b=None,stride=1,pad=0,dilation=1):
    # 小张量的标准 3x3 卷积走 CUDA C Winograd；大张量复用 CuPy 的
    # 形状选择（FFT/GEMM/im2col/向量化 Winograd），避免低占用 kernel。
    sh,sw=(int(stride),)*2 if np.isscalar(stride) else tuple(map(int,stride))
    ph,pw=(int(pad),)*2 if np.isscalar(pad) else tuple(map(int,pad))
    dh,dw=(int(dilation),)*2 if np.isscalar(dilation) else tuple(map(int,dilation))
    n,_,h,wi=map(int,x.shape); kh,kw=map(int,w.shape[2:]); oh=(h+2*ph-dh*(kh-1)-1)//sh+1; ow=(wi+2*pw-dw*(kw-1)-1)//sw+1
    output_elements = n * int(w.shape[0]) * oh * ow
    standard_3x3 = tuple(map(int, w.shape[2:])) == (3, 3) and (sh,sw,ph,pw,dh,dw) == (1,1,1,1,1,1)
    if (_raw_enabled() and x.dtype == cp.float32 and w.dtype == cp.float32 and standard_3x3
            and output_elements <= _WINOGRAD_MAX_OUTPUT):
        return winograd_conv2d_forward(x, w, b)
    if (_raw_enabled() and x.dtype == cp.float32 and w.dtype == cp.float32
            and output_elements > _RAW_REFERENCE_CONV_OUTPUT):
        # Large feature maps dominate ResNet training.  The experimental
        # one-thread-per-output CUDA-C reduction is intentionally bypassed for
        # them in favor of the framework's vectorized CuPy implementation.
        _LAST.update(actual_backend="cupy", fallback_reason="large_conv_vectorized", last_error=None)
        return _vectorized_conv2d_forward(
            x, w, b, (sh, sw), (ph, pw), (dh, dw), standard_3x3,
        )
    col=im2col_forward(x,w.shape[2:],(sh,sw),(ph,pw),(dh,dw),True)
    y=col.dot(cp.ascontiguousarray(w).reshape(w.shape[0],-1).T); y=y.reshape(n,oh,ow,w.shape[0]).transpose(0,3,1,2); return bias_add_forward(y, b) if b is not None else y

def winograd_conv2d_forward(x, w, b=None):
    """CUDA C Winograd F(2x2,3x3) forward; inputs are contiguous float32 NCHW."""
    n,c,h,wi=map(int,x.shape); oc=int(w.shape[0]); oh,ow=h,wi
    y=cp.empty((n,oc,oh,ow),dtype=cp.float32)
    _call_extra("winograd_f32", (cp.ascontiguousarray(x), cp.ascontiguousarray(w), y,
                            np.int32(n), np.int32(c), np.int32(h), np.int32(wi),
                            np.int32(oc), np.int32(oh), np.int32(ow), np.int32((ow+1)//2)),
          int(y.size))
    _LAST.update(actual_backend="rawmodule", fallback_reason=None, last_error=None)
    return bias_add_forward(y, b) if b is not None else y

def batchnorm_forward(x, mean, var, gamma, beta, eps=1e-5):
    """Normalize NCHW float32 data with a CUDA C elementwise kernel."""
    eps_value = cp.asarray(eps, dtype=x.dtype)
    if (not _raw_enabled()
            or any(a.dtype != cp.float32 for a in (x, mean, var, gamma, beta))
            or int(x.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        return (gamma.reshape(1,-1,1,1) *
                (x-mean.reshape(1,-1,1,1)) /
                cp.sqrt(var.reshape(1,-1,1,1) + eps_value) +
                beta.reshape(1,-1,1,1))
    n,c,h,w=map(int,x.shape); y=cp.empty_like(x); spatial=h*w
    _call_extra("batchnorm_fwd_f32", (cp.ascontiguousarray(x), cp.ascontiguousarray(mean.reshape(-1)), cp.ascontiguousarray(var.reshape(-1)), cp.ascontiguousarray(gamma.reshape(-1)), cp.ascontiguousarray(beta.reshape(-1)), y, np.int64(n), np.int64(c), np.int64(spatial), np.float32(eps)), x.size)
    return y

def batchnorm_backward(x, gy, mean, var, gamma, eps=1e-5):
    """BatchNorm backward with CuPy channel reductions and CUDA C gx kernel."""
    n,c,h,w=map(int,x.shape); spatial=h*w
    dtype = x.dtype
    mean1=mean.reshape(1,c,1,1); var1=var.reshape(1,c,1,1)
    eps_value = cp.asarray(eps, dtype=dtype)
    inv=cp.asarray(1, dtype=dtype)/cp.sqrt(var1+eps_value)
    xhat=(x-mean1)*inv
    gbeta=gy.sum(axis=(0,2,3)); ggamma=(gy*xhat).sum(axis=(0,2,3)); gx=cp.empty_like(x)
    if (_raw_enabled()
            and all(a.dtype == cp.float32 for a in (x,gy,mean,gamma))
            and int(x.size) <= _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        _call_extra("batchnorm_bwd_f32", (cp.ascontiguousarray(x), cp.ascontiguousarray(gy), cp.ascontiguousarray(mean.reshape(-1)), cp.ascontiguousarray(inv.reshape(-1)), cp.ascontiguousarray(gamma.reshape(-1)), cp.ascontiguousarray(gbeta), cp.ascontiguousarray(ggamma), gx, np.int64(n), np.int64(c), np.int64(spatial)), x.size)
    else:
        m=cp.asarray(n*spatial, dtype=dtype)
        gx=(gamma.reshape(1,c,1,1)*inv*
            (m*gy-gbeta.reshape(1,c,1,1)-xhat*ggamma.reshape(1,c,1,1))/m)
    return gx, ggamma, gbeta

def global_average_pool_forward(x):
    if (not _raw_enabled() or x.dtype != cp.float32
            or int(x.size) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        return x.mean(axis=(2,3), keepdims=True)
    n,c,h,w=map(int,x.shape); y=cp.empty((n,c),dtype=x.dtype); _call_extra("global_avg_fwd_f32", (cp.ascontiguousarray(x), y, np.int64(n), np.int64(c), np.int64(h*w)), n*c); return y.reshape(n,c,1,1)

def global_average_pool_backward(gy, input_shape):
    n,c,h,w=map(int,input_shape); spatial=h*w
    if (not _raw_enabled() or gy.dtype != cp.float32
            or int(n*c*spatial) > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        # CuPy's scalar promotion can turn a float32 array into float64 when
        # dividing by a Python integer.  The resulting float64 gradient then
        # propagates through every preceding ResNet layer and doubles both
        # arithmetic cost and checkpoint size.  Keep the reduction factor on
        # the same device and dtype as ``gy`` explicitly.
        scale = cp.asarray(spatial, dtype=gy.dtype)
        return cp.broadcast_to(gy, input_shape) / scale
    gx=cp.empty(input_shape,dtype=gy.dtype); _call_extra("global_avg_bwd_f32", (cp.ascontiguousarray(gy).reshape(n,c), gx, np.int64(n), np.int64(c), np.int64(spatial)), n*c*spatial); return gx

def conv2d_backward(gy, x, w, b=None, stride=1, pad=0, dilation=1):
    """反向阶段复用 GPU im2col 和 CuPy/cuBLAS GEMM 计算 gW、gb。"""
    n,c,h,wi=map(int,x.shape); oc,_,kh,kw=map(int,w.shape); sh,sw=(int(stride),)*2 if np.isscalar(stride) else tuple(map(int,stride)); ph,pw=(int(pad),)*2 if np.isscalar(pad) else tuple(map(int,pad)); oh,ow=map(int,gy.shape[2:])
    dh,dw=(int(dilation),)*2 if np.isscalar(dilation) else tuple(map(int,dilation))
    use_custom_backward = (
        _raw_enabled() and x.dtype == cp.float32 and w.dtype == cp.float32
        and gy.dtype == cp.float32 and (dh, dw) == (1, 1)
        and int(x.size) <= _RAW_REFERENCE_CONV_OUTPUT
    )
    if use_custom_backward:
        gx = cp.zeros_like(x); _call("conv_bwd_x_f32", (cp.ascontiguousarray(gy), cp.ascontiguousarray(w), gx, n,c,h,wi,oc,kh,kw,oh,ow,sh,sw,ph,pw), x.size)
    else:
        from ..functions import (
            Conv2d as ConvFunction,
            col2im_array,
            conv2d_backward_input_array,
        )
        from ..core import as_Tensor
        # Keep the backward algorithm consistent with the cached forward path
        # when CuPy's Winograd implementation was selected.  FFT/GEMM use the
        # same vectorized deconvolution/im2col gradients below as the regular
        # framework path.
        fn = ConvFunction(stride=(sh, sw), pad=(ph, pw), dilation=(dh, dw))
        selected_path = fn._get_forward_path(x, w, b)
        if selected_path == "winograd" and (kh, kw) == (3, 3):
            gx_t, gw_t, gb_t = fn.winograd_conv2d_backward(
                gy, x, w, as_Tensor(b) if b is not None else None,
            )
            return (
                gx_t.data,
                gw_t.data,
                gb_t.data if gb_t is not None else None,
            )
        if (dh, dw) == (1, 1):
            # Match Deconv2d.forward used by the CuPy baseline: one
            # (OC,C,KH,KW) x (N,OC,OH,OW) contraction, then vectorized col2im.
            # The older conv2d_backward_input_array path issued one tensordot
            # for every kernel position and became the dominant cost on the
            # full DonkeyCar run.
            gcol = cp.tensordot(w, gy, axes=(0, 1))
            gcol = cp.rollaxis(gcol, 3)
            gx = col2im_array(
                gcol, (n, c, h, wi), (kh, kw), (sh, sw), (ph, pw),
                to_matrix=False,
            )
        else:
            gx = conv2d_backward_input_array(
                gy, w, stride=stride, pad=pad, dilation=dilation,
                out_h=x.shape[2], out_w=x.shape[3],
            )
    if use_custom_backward:
        col = im2col_forward(x, w.shape[2:], stride, pad, dilation, True)
    else:
        # Match the vectorized CuPy baseline for large weight gradients too;
        # the custom im2col kernel is retained for small operator probes.
        from ..functions import im2col_array
        col = im2col_array(
            x, (kh, kw), (sh, sw), (ph, pw), False,
            dilation=(dh, dw), xp=cp,
        )
    if use_custom_backward:
        gmat = gy.transpose(0, 2, 3, 1).reshape(-1, w.shape[0])
        gw = gmat.T.dot(col).reshape(w.shape)
    else:
        # Same contraction as Conv2DGradW; avoid an extra reshape/copy of the
        # large im2col matrix on every training batch.
        gw = cp.tensordot(gy, col, axes=((0, 2, 3), (0, 4, 5)))
    gb = gy.sum(axis=(0, 2, 3)) if b is not None else None
    return gx, gw, gb

def maxpool_forward(x,kernel,stride=1,pad=0):
    # pool kernel 保存窗口内首次出现的最大值索引，保证与 NumPy/CuPy argmax 的
    # tie-break 一致；padding 使用 0，与现有 im2col_array 基线保持一致。
    kh,kw=(int(kernel),)*2 if np.isscalar(kernel) else tuple(map(int,kernel)); sh,sw=(int(stride),)*2 if np.isscalar(stride) else tuple(map(int,stride)); ph,pw=(int(pad),)*2 if np.isscalar(pad) else tuple(map(int,pad)); n,c,h,w=map(int,x.shape); oh=(h+2*ph-kh)//sh+1; ow=(w+2*pw-kw)//sw+1
    output_size = int(n * c * oh * ow)
    if (not _raw_enabled() or x.dtype != cp.float32
            or output_size > _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        from ..functions import im2col_array
        col=im2col_array(x,(kh,kw),(sh,sw),(ph,pw),False,xp=cp).reshape(n,c,kh*kw,oh,ow); return col.max(2),col.argmax(2).astype(cp.int64)
    y=cp.empty((n,c,oh,ow),dtype=x.dtype); idx=cp.empty((n,c,oh,ow),dtype=cp.int64); _call("pool_f32",(cp.ascontiguousarray(x),y,idx,n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw),y.size); return y,idx

def maxpool_backward(gy,indexes,input_shape,kernel,stride=1,pad=0):
    # float32/int64 路径由 pool_bwd_f32 按 forward 保存的 kh*KW+kw 索引
    # scatter-add；重叠窗口用 atomicAdd 累加。其他 dtype 保留 CuPy 回退。
    n,c,h,w=map(int,input_shape); kh,kw=(int(kernel),)*2 if np.isscalar(kernel) else tuple(map(int,kernel)); sh,sw=(int(stride),)*2 if np.isscalar(stride) else tuple(map(int,stride)); ph,pw=(int(pad),)*2 if np.isscalar(pad) else tuple(map(int,pad)); oh,ow=gy.shape[2:]; gx=cp.zeros(input_shape,dtype=gy.dtype)
    if (_raw_enabled() and gy.dtype == cp.float32 and indexes.dtype == cp.int64
            and int(gy.size) <= _RAW_CUSTOM_ELEMENTWISE_MAX_SIZE):
        _call("pool_bwd_f32", (cp.ascontiguousarray(gy), cp.ascontiguousarray(indexes), gx, n,c,h,w,int(oh),int(ow),kh,kw,sh,sw,ph,pw), int(n*c*oh*ow)); return gx
    # Vectorized CuPy fallback for large maps.  The previous Python loop over
    # every channel and window made the performance guard slower than the
    # kernel it was meant to replace.
    from ..functions import col2im_array
    flat = cp.zeros((n * c * oh * ow * kh * kw,), dtype=gy.dtype)
    positions = indexes.reshape(-1).astype(cp.int64)
    positions = positions + cp.arange(positions.size, dtype=cp.int64) * (kh * kw)
    flat[positions] = gy.reshape(-1)
    gcol = flat.reshape(n, c, oh, ow, kh, kw)
    gcol = gcol.swapaxes(2, 4).swapaxes(3, 5)
    return col2im_array(gcol, (n, c, h, w), (kh, kw), (sh, sw), (ph, pw), to_matrix=False)
