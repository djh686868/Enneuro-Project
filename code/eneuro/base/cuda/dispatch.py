import os
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

_SRC = r'''
// 所有 kernel 均采用“一线程处理一个输出元素”的简单映射。
// 这样可以先验证索引、边界和梯度语义，再在最终 DLL 路线中优化线程块、共享内存。
extern "C" __global__ void add_f32(const float*a,const float*b,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]+b[i];}
extern "C" __global__ void sub_f32(const float*a,const float*b,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]-b[i];}
extern "C" __global__ void mul_f32(const float*a,const float*b,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]*b[i];}
extern "C" __global__ void div_f32(const float*a,const float*b,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]/b[i];}
extern "C" __global__ void neg_f32(const float*x,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=-x[i];}
extern "C" __global__ void exp_f32(const float*x,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=expf(x[i]);}
extern "C" __global__ void log_f32(const float*x,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=logf(x[i]);}
extern "C" __global__ void pow_f32(const float*x,float*y,float c,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=powf(x[i],c);}
extern "C" __global__ void relu_f32(const float*x,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]>0.0f?x[i]:0.0f;}
extern "C" __global__ void relu_bwd_f32(const float*x,const float*gy,float*gx,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)gx[i]=x[i]>0.0f?gy[i]:0.0f;}
extern "C" __global__ void sigmoid_f32(const float*x,float*y,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=0.5f*tanhf(0.5f*x[i])+0.5f;}
extern "C" __global__ void bias_f32(const float*x,const float*b,float*y,long n,long c,long s){long i=(long)blockIdx.x*blockDim.x+threadIdx.x;long total=n*c*s;if(i<total){long ch=(i/s)%c;y[i]=x[i]+b[ch];}}
extern "C" __global__ void im2col_f32(const float*x,float*col,int N,int C,int H,int W,int KH,int KW,int OH,int OW,int SH,int SW,int PH,int PW,int DH,int DW){long t=(long)blockIdx.x*blockDim.x+threadIdx.x;long total=(long)N*OH*OW*C*KH*KW;if(t>=total)return;int kw=t%KW;t/=KW;int kh=t%KH;t/=KH;int c=t%C;t/=C;int ow=t%OW;t/=OW;int oh=t%OH;t/=OH;int n=t;int ih=oh*SH+kh*DH-PH,iw=ow*SW+kw*DW-PW;long o=((((long)n*OH+oh)*OW+ow)*C*KH*KW)+(c*KH*KW+kh*KW+kw);col[o]=(ih>=0&&ih<H&&iw>=0&&iw<W)?x[(((long)n*C+c)*H+ih)*W+iw]:0.0f;}
extern "C" __global__ void pool_f32(const float*x,float*y,long long*idx,int N,int C,int H,int W,int KH,int KW,int OH,int OW,int SH,int SW,int PH,int PW){long t=(long)blockIdx.x*blockDim.x+threadIdx.x;long total=(long)N*C*OH*OW;if(t>=total)return;int ow=t%OW;t/=OW;int oh=t%OH;t/=OH;int c=t%C;int n=t/C;float best=0.0f;long long bi=0;bool found=false;for(int kh=0;kh<KH;++kh)for(int kw=0;kw<KW;++kw){int ih=oh*SH+kh-PH,iw=ow*SW+kw-PW;float v=(ih>=0&&ih<H&&iw>=0&&iw<W)?x[(((long)n*C+c)*H+ih)*W+iw]:0.0f;long long k=(long long)kh*KW+kw;if(!found||v>best){best=v;bi=k;found=true;}}long o=((long)n*C+c)*OH*OW+(long)oh*OW+ow;y[o]=best;idx[o]=bi;}
'''


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


def is_available(backend="rawmodule"):
    if cp is None or not available():
        return False
    if backend in ("rawmodule", "auto"):
        try:
            module(_SRC, ("relu_f32",)).get_function("relu_f32")
        except Exception:
            return False
    return True


def _raw_enabled():
    return cp is not None and available() and _BACKEND in ("rawmodule", "auto")


def _call(name, args, n):
    # 统一使用 256 threads/block；grid 采用 ceil(n/256)，最后一个 block
    # 通过 kernel 内的 i<n 判断处理尾部元素，避免越界写入。
    mod = module(_SRC, (name,))
    mod.get_function(name)(((int(n)+255)//256,), (256,), args)


def _unary(x, name, fallback):
    # 首版 kernel 只覆盖 contiguous float32。其他 dtype、空数组或不满足条件
    # 的输入交给 CuPy，保证框架原有广播和 dtype 语义不被破坏。
    if not _raw_enabled() or x.dtype != cp.float32 or x.size == 0:
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
    if not _raw_enabled() or a.dtype != cp.float32 or b.dtype != cp.float32 or a.shape != b.shape or a.size == 0:
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
    if not _raw_enabled() or x.dtype != cp.float32:
        return x + (bias.reshape((1, c) if x.ndim == 2 else (1, c, 1, 1)))
    try:
        x = cp.ascontiguousarray(x); bias = cp.ascontiguousarray(bias); y = cp.empty_like(x)
        _call("bias_f32", (x, bias, y, np.int64(x.size), np.int64(c), np.int64(s)), x.size)
        return y
    except Exception:
        if _BACKEND == "rawmodule": raise
        return x + bias.reshape((1, c) if x.ndim == 2 else (1, c, 1, 1))

def pow_forward(x, exponent):
    if not _raw_enabled() or x.dtype != cp.float32 or not np.isscalar(exponent):
        return cp.power(x, exponent)
    try:
        x=cp.ascontiguousarray(x); y=cp.empty_like(x); _call("pow_f32",(x,y,np.float32(exponent),np.int64(x.size)),x.size); return y
    except Exception:
        if _BACKEND == "rawmodule": raise
        return cp.power(x, exponent)

def relu_forward(x): return _unary(x,"relu_f32",lambda a:cp.maximum(a,0))
def relu_backward(x,gy):
    if not _raw_enabled() or x.dtype != cp.float32 or gy.dtype != cp.float32:
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

def conv2d_forward(x,w,b=None,stride=1,pad=0,dilation=1):
    col=im2col_forward(x,w.shape[2:],stride,pad,dilation,True); y=col.dot(cp.ascontiguousarray(w).reshape(w.shape[0],-1).T); n,_,h,wi=x.shape; kh,kw=w.shape[2:]; sh,sw=(int(stride),)*2 if np.isscalar(stride) else stride; ph,pw=(int(pad),)*2 if np.isscalar(pad) else pad; dh,dw=(int(dilation),)*2 if np.isscalar(dilation) else dilation; oh=(h+2*ph-dh*(kh-1)-1)//sh+1; ow=(wi+2*pw-dw*(kw-1)-1)//sw+1; y=y.reshape(n,oh,ow,w.shape[0]).transpose(0,3,1,2); return y+cp.asarray(b).reshape(1,-1,1,1) if b is not None else y

def conv2d_backward(gy, x, w, b=None, stride=1, pad=0, dilation=1):
    """反向阶段复用 GPU im2col 和 CuPy/cuBLAS GEMM 计算 gW、gb。"""
    from ..functions import conv2d_backward_input_array
    gx = conv2d_backward_input_array(gy, w, stride=stride, pad=pad,
                                     dilation=dilation, out_h=x.shape[2], out_w=x.shape[3])
    col = im2col_forward(x, w.shape[2:], stride, pad, dilation, True)
    gmat = gy.transpose(0, 2, 3, 1).reshape(-1, w.shape[0])
    gw = gmat.T.dot(col).reshape(w.shape)
    gb = gy.sum(axis=(0, 2, 3)) if b is not None else None
    return gx, gw, gb

def maxpool_forward(x,kernel,stride=1,pad=0):
    # pool kernel 保存窗口内首次出现的最大值索引，保证与 NumPy/CuPy argmax 的
    # tie-break 一致；padding 使用 0，与现有 im2col_array 基线保持一致。
    kh,kw=(int(kernel),)*2 if np.isscalar(kernel) else tuple(map(int,kernel)); sh,sw=(int(stride),)*2 if np.isscalar(stride) else tuple(map(int,stride)); ph,pw=(int(pad),)*2 if np.isscalar(pad) else tuple(map(int,pad)); n,c,h,w=map(int,x.shape); oh=(h+2*ph-kh)//sh+1; ow=(w+2*pw-kw)//sw+1
    if not _raw_enabled() or x.dtype != cp.float32:
        from ..functions import im2col_array
        col=im2col_array(x,(kh,kw),(sh,sw),(ph,pw),False,xp=cp).reshape(n,c,kh*kw,oh,ow); return col.max(2),col.argmax(2).astype(cp.int64)
    y=cp.empty((n,c,oh,ow),dtype=x.dtype); idx=cp.empty((n,c,oh,ow),dtype=cp.int64); _call("pool_f32",(cp.ascontiguousarray(x),y,idx,n,c,h,w,kh,kw,oh,ow,sh,sw,ph,pw),y.size); return y,idx

def maxpool_backward(gy,indexes,input_shape,kernel,stride=1,pad=0):
    # 当前阶段采用确定性的 CuPy 索引累加实现反向；它与 forward 保存的
    # kh*KW+kw 索引严格配套，后续 DLL 路线再替换为专用 gather kernel。
    n,c,h,w=map(int,input_shape); kh,kw=(int(kernel),)*2 if np.isscalar(kernel) else tuple(map(int,kernel)); sh,sw=(int(stride),)*2 if np.isscalar(stride) else tuple(map(int,stride)); ph,pw=(int(pad),)*2 if np.isscalar(pad) else tuple(map(int,pad)); oh,ow=gy.shape[2:]; gx=cp.zeros(input_shape,dtype=gy.dtype)
    for a in range(kh):
        for b in range(kw):
            mask=(indexes==a*kw+b); hs=cp.arange(oh)[:,None]*sh+a-ph; ws=cp.arange(ow)[None,:]*sw+b-pw; valid=(hs>=0)&(hs<h)&(ws>=0)&(ws<w); hh=cp.broadcast_to(hs,(oh,ow))[valid]; ww=cp.broadcast_to(ws,(oh,ow))[valid]
            for nn in range(n):
                for cc in range(c): cp.add.at(gx,(nn,cc,hh,ww),(gy[nn,cc]*mask[nn,cc])[valid])
    return gx
