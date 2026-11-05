import weakref
import time
import os
import math
from .core import Tensor 
from .core import as_Tensor, as_array
from .core import Function
import numpy as np
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

from .core import Config
import builtins


_CPU_TILED_ENABLED = os.environ.get("ENE_CONV_CPU_TILED", "1") != "0"
_CPU_TILED_MIN_OH_OW = int(os.environ.get("ENE_CONV_CPU_TILED_MIN_OH_OW", "4096"))
_CPU_TILE_TARGET_BYTES = int(os.environ.get("ENE_CONV_CPU_TILE_TARGET_BYTES", "262144"))
_CPU_IM2COL_TILE = (
    int(os.environ.get("ENE_CONV_CPU_IM2COL_TILE_H", "32")),
    int(os.environ.get("ENE_CONV_CPU_IM2COL_TILE_W", "32")),
)
_CPU_COL2IM_TILE = (
    int(os.environ.get("ENE_CONV_CPU_COL2IM_TILE_H", "32")),
    int(os.environ.get("ENE_CONV_CPU_COL2IM_TILE_W", "32")),
)


def _use_cpu_tiled_kernel(xp, oh, ow):
    if not _CPU_TILED_ENABLED or xp is not np:
        return False
    return (oh * ow) >= _CPU_TILED_MIN_OH_OW


def _select_cpu_tile_shape(n, c, kh, kw, oh, ow, itemsize, base_tile):
    base_h = builtins.max(1, base_tile[0])
    base_w = builtins.max(1, base_tile[1])
    bytes_per_spatial_cell = builtins.max(1, n * c * kh * kw * itemsize)
    budget_cells = builtins.max(1, _CPU_TILE_TARGET_BYTES // bytes_per_spatial_cell)
    max_cells = builtins.max(1, builtins.min(base_h * base_w, budget_cells))

    tile_h = builtins.min(base_h, oh, builtins.max(1, int(math.sqrt(max_cells))))
    tile_w = builtins.min(base_w, ow, builtins.max(1, max_cells // tile_h))

    while tile_h * tile_w > max_cells and tile_w > 1:
        tile_w -= 1
    while tile_h * tile_w > max_cells and tile_h > 1:
        tile_h -= 1
        tile_w = builtins.min(base_w, ow, builtins.max(1, max_cells // tile_h))
        while tile_h * tile_w > max_cells and tile_w > 1:
            tile_w -= 1

    return tile_h, tile_w


def get_array_module(arr):
    """获取数组对应的计算模块 (numpy 或 cupy)"""
    if isinstance(arr, Tensor):
        return get_array_module(arr.data)
    if isinstance(arr, np.ndarray):
        return np
    elif has_cupy and isinstance(arr, cp.ndarray):
        return cp
    else:
        return np


def to_xp(arr, xp):
    """
    将数组arr转换为xp对应的类型。

    Args:
        arr: 输入数组（可以是numpy数组、cupy数组或Tensor）
        xp: 目标数组模块（numpy或cupy）

    Returns:
        转换后的数组

    Notes:
        - 当目标是numpy但arr是cupy数组时，使用cp.asnumpy()转换
        - 当目标是cupy但arr是numpy数组时，使用cp.asarray()转换
        - 如果arr已经是目标类型，则不进行转换
    """
    if isinstance(arr, Tensor):
        arr = arr.data
    if xp is np:
        if has_cupy and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    else:  # xp is cp
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr

'''
other functions
'''

class Sin(Function):
    def forward(self, *xs):
        xs = xs[0]
        xp = get_array_module(xs)
        return xp.sin(xs)

    def backward(self, gys):
        x = self.inputs[0].data
        xp = get_array_module(x)
        gx = gys * xp.cos(x)
        return gx
    
class Cos(Function):
    def forward(self,*xs):
        xs = xs[0]
        xp = get_array_module(xs)
        return xp.cos(xs)
    def backward(self, gys):
        x = self.inputs[0].data
        xp = get_array_module(x)
        gx = gys * -xp.sin(x)
        return gx

class Exp(Function):
    def forward(self,*xs):
        xs = xs[0]
        xp = get_array_module(xs)
        return xp.exp(xs)
    def backward(self, gys):
        x = self.inputs[0].data
        xp = get_array_module(x)
        gx = gys * xp.exp(x)
        return gx

class Tanh(Function):
    def forward(self,*xs):
        xs=xs[0]
        xp = get_array_module(xs)
        return xp.tanh(xs)
    def backward(self, gys):
        y = self.outputs[0].data
        gx = gys * (1 - y**2)
        return gx
    
class Log(Function):
    def forward(self,*xs):
        xs=xs[0]
        xp = get_array_module(xs)
        return xp.log(xs)
    def backward(self, gys):
        x = self.inputs[0].data
        gx = gys / x
        return gx


class Reshape(Function):
    def __init__(self,shape,visualize=False):
        super().__init__()
        self.shape = shape
        self.visualize = visualize

    def forward(self,*xs):
        xs=xs[0]
        self.x_shape = xs.shape
        return xs.reshape(self.shape)
    def backward(self,gys):
        return  reshape(gys, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return x
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self,axes = None):
        super().__init__()
        self.axes = axes
    def forward(self,*xs):
        xs=xs[0]
        return xs.transpose(self.axes)
    def backward(self,gys):
        if self.axes is None:
            return gys.transpose()
        axes_len = len(self.axes)
        #计算逆转置的轴
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gys, inv_axes)
#转置操作的便捷函数,Tensor
#的transpose方法会调用它    
def transpose(x, axes = None):
    return Transpose(axes)(x)
#切片操作
class GetItem(Function):
    def __init__ (self,slices,visualize=False):
        self.slices = slices
        self.visualize = visualize

    def forward (self,*xs):
        xs=xs[0]
        return xs[self.slices]
    def backward (self,gys):
        x=self.inputs[0]
        gx=GetItemGrad(self.slices,x.shape)(gys)
        return gx   
    
#切片操作的反向传播
class GetItemGrad(Function):
    def __init__ (self,slices,x_shape,visualize=False):
        self.slices = slices
        self.x_shape = x_shape
        self.visualize = visualize
    def forward (self,*xs):
        xs=xs[0]
        xp = get_array_module(xs)
        gx = xp.zeros(self.x_shape)
        gx[self.slices] = xs
        return gx
    #切片操作的反向传播
    def backward (self,gys):
        return GetItem(self.slices)(gys)
    

#切片操作的便捷函数,Tensor
#的getitem方法会调用它
def get_item(x, slices):
    return GetItem(slices)(x)

#增加维度
def expand_dims(x, axis):
    x = as_Tensor(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

#展平操作
def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))

#沿指定轴求和操作
class Sum(Function):
    def __init__ (self,axis,keepdims,visualize=False):
        self.axis = axis
        self.keepdims = keepdims
        self.visualize = visualize
    def forward (self,*xs):
        xs=xs[0]
        self.x_shape = xs.shape
        return xs.sum(self.axis,self.keepdims)
    def backward (self,gys):
        ndim = len(self.x_shape)
        tupled_axis = self.axis
        if self.axis is None:
            tupled_axis = None
        elif not isinstance(self.axis, tuple):
            tupled_axis = (self.axis,)

        if not (ndim == 0 or tupled_axis is None or self.keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
            shape = list(gys.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gys.shape
        gys = gys.reshape(shape)  # reshape
        gx = broadcast_to(gys, self.x_shape)
        return gx

def sum (x,axis = None,keepdims = False):
    return Sum(axis,keepdims)(x)

class Mean(Function):
    def __init__ (self,axis,keepdims,visualize=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.visualize = visualize

    def forward (self,*xs):
        xs=xs[0]
        self.x_shape = xs.shape
        return xs.mean(self.axis,self.keepdims)
    def backward (self,gys):
        ndim = len(self.x_shape)
        tupled_axis = self.axis
        if self.axis is None:
            tupled_axis = None
        elif not isinstance(self.axis, tuple):
            tupled_axis = (self.axis,)

        # 计算元素个数
        count = 1
        if tupled_axis is not None:
            for axis in tupled_axis:
                count *= self.x_shape[axis]
        else:
            count = np.prod(self.x_shape)

        # 与Sum类似，但需要除以元素个数
        if not (ndim == 0 or tupled_axis is None or self.keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
            shape = list(gys.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gys.shape
        gys = gys.reshape(shape)  # reshape
        gx = broadcast_to(gys, self.x_shape) / count
        return gx

def mean (x,axis = None,keepdims = False):
    return Mean(axis,keepdims)(x)


#求和到目标形状
class SumTo(Function):
    def __init__ (self,shape,visualize=False):
        self.shape = shape
        self.visualize = visualize
    def forward (self,*xs):
        xs=xs[0]
        self.x_shape = xs.shape
        ndim = len(self.shape)
        lead = xs.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = xs.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
             y = y.squeeze(lead_axis)
        return y

    def backward (self,gys):
        return reshape(gys,self.x_shape)
 
def sum_to(x, shape):
    if x.shape == shape:
        return as_Tensor(x)
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__ (self,shape,visualize=False):
        super().__init__()
        self.visualize = visualize
        self.shape = shape
    def forward (self,*xs):
        xs=xs[0]
        self.x_shape = xs.shape
        xp = get_array_module(xs)
        return xp.broadcast_to(xs,self.shape)
    def backward (self,gys):
        return sum_to(gys,self.x_shape)
  

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_Tensor(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_Tensor(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)



class MatMul(Function):
    def __init__(self):
        super().__init__()

    def forward(self, *xs):
        """
        矩阵乘法的前向传播。

        Args:
            x: 输入数据，形状 (N, M)
            W: 权重，形状 (M, K)

        Returns:
            输出数据，形状 (N, K)

        GPU适配说明:
            - 优先从权重W获取数组模块类型
            - 当W是cupy数组时，需要先将x转换为cupy数组
        """
        x = xs[0]
        W = xs[1]

        # 关键修复：优先从权重W获取数组模块，而非输入x
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 如果x不是xp类型的数组，需要先转换
        # 使用to_xp辅助函数正确处理numpy/cupy之间的转换
        x_data = to_xp(x, xp)

        y = x_data.dot(W_data)
        return y

    def backward(self, gys):
        x, W = self.inputs
        #调用的是自定义的matmul函数
        gx = matmul(gys, W.T)
        gW = matmul(x.T, gys)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)

##线性变换函数###

class Linear(Function):
    def forward(self, *xs):
        """
        线性变换的前向传播（矩阵乘法）。

        Args:
            x: 输入数据，形状 (N, in_features)
            w: 权重，形状 (in_features, out_features)
            b: 偏置，形状 (out_features,)，可为None

        Returns:
            输出数据，形状 (N, out_features)

        GPU适配说明:
            - 优先从权重w获取数组模块类型，因为输出格式由权重决定
            - 当模型在GPU上但输入数据还在CPU上时，w会是cupy数组
            - 需要先将x转换为cupy数组，然后才能执行dot操作
        """
        x = xs[0]
        #print(f"x.dtype = {x.dtype}")
        w = xs[1]
        b = xs[2]

        # 关键修复：优先从权重w获取数组模块，而非输入x
        # 这样可以确保当x是numpy但w是cupy时，使用cupy进行计算
        w_data = w.data if isinstance(w, Tensor) else w
        xp = get_array_module(w_data)

        # 如果x不是xp类型的数组，需要先转换
        # 使用to_xp辅助函数正确处理numpy/cupy之间的转换
        x_data = to_xp(x, xp)

        y = x_data.dot(w_data)
        if b is None:
            return y
        else:
            b_data = to_xp(b, xp)
            return y + b_data 
    def backward (self,gys):
        x,w,b = self.inputs
        gx = matmul(gys,w.T)
        gw = matmul(x.T,gys)
        if b is None:
            gb = None
        else:
            gb = gys.sum(axis=0)
        return gx,gw,gb

#封装成线性变换的便捷函数
def linear(x, W, b=None):
    return Linear()(x, W, b)

class Sigmoid(Function):
    def forward(self, *xs):
        x = xs[0]
        xp = get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  #使用numpy/cupy函数而非自定义函数
        return y

    def backward(self, gys):
        y = self.outputs[0]()
        gx = gys * y * (1 - y)
        return gx

# Sigmoid函数的便捷函数
def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Function):
    def forward(self, *xs):
        x = xs[0]
        xp = get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gys):
        x, = self.inputs
        mask = x.data > 0
        gx = gys * mask
        return gx


def relu(x):
    return ReLU()(x)

class Softmax(Function):
    def __init__(self, axis=1,visualize=False):
        super().__init__()
        self.axis = axis
        self.visualize = visualize

    def forward(self, *xs):
        x = xs[0]
        xp = get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gys):
        y = self.outputs[0]()
        gx = y * gys
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)



class Max(Function):
    def __init__(self, axis=None, keepdims=False, visualize=False):
        self.axis = axis
        self.keepdims = keepdims
        self.visualize = visualize

    def forward(self, *xs):
        x = xs[0]
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def _get_backward_shape(self, x):
        """获取反向传播所需的形状"""
        if self.axis is None:
            axis = range(x.ndim)
        elif isinstance(self.axis, int):
            axis = (self.axis,)
        else:
            axis = self.axis

        shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
        return shape

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref
        shape = self._get_backward_shape(x)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, *xs):
        x = xs[0]        
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)





##卷积部分函数###

#准备函数1#
#反卷积输出尺寸，即前向传播输入的未填充的尺寸
def get_deconv_outsize(size, k, s, p, dilation=1):
    return s * (size - 1) + (k - 1) * dilation + 1 - 2 * p

#卷积输出尺寸，即前向传播输出的尺寸
def get_conv_outsize(input_size, kernel_size, stride, pad, dilation=1):
    return (input_size + pad * 2 - (kernel_size - 1) * dilation - 1) // stride + 1

#确保输入为二元组
def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

#准备函数2#
#col2im_array纯计算不微分，就单独成函数，不储存任何信息
#col2im和im2col函数的Function类封装互为逆
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix,visualize=False):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix
        self.visualize = visualize

    def forward(self, *xs):
        x = xs[0]
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gys):
        gx = col2im(gys, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx

#依旧是im2col的便捷函数
def im2col(x, kernel_size, stride=(1,1), pad=(0,0), to_matrix=True):
    #     参数说明
    # x (dezero.Variable 或 ndarray): 输入变量，形状为 (N, C, H, W)
    # kernel_size (int 或 (int, int)): 卷积核大小
    # stride (int 或 (int, int)): 卷积核步长
    # pad (int 或 (int, int)): 输入数组的空间填充宽度
    # to_matrix (bool): 如果为True，col将被重塑为2D数组，形状为 (N*OH*OW, C*KH*KW)
    # 返回值
    # dezero.Variable: 输出变量。如果to_matrix为False，输出形状为 (N, C, KH, KW, OH, OW)；否则为 (N*OH*OW, C*KH*KW)
    # 符号说明
    # N: 批次大小
    # C: 输入通道数
    # H 和 W: 输入图像的高度和宽度
    # KH 和 KW: 滤波器的高度和宽度
    # SH 和 SW: 滤波器的步长
    # PH 和 PW: 空间填充大小
    # OH 和 OW: 输出的高度和宽度
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix,visualize=False):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix
        self.visualize = visualize


    def forward(self, *xs):
        x = xs[0]
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gys):
        gx = im2col(gys, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=(1,1), pad=(0,0), to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def im2col_array(img, kernel_size, stride, pad, to_matrix=True, dilation=1, xp=None):
    """
    将图像转换为列矩阵（im2col操作），用于卷积操作。

    Args:
        img: 输入图像，形状为 (N, C, H, W) 的4D张量
        kernel_size: 卷积核大小
        stride: 步长
        pad: 填充大小
        to_matrix: 是否将输出转换为矩阵形式
        dilation: 扩张率（空洞卷积）
        xp: 指定数组模块（numpy或cupy），如果为None则自动从img推断

    Returns:
        转换后的列矩阵

    Notes:
        - 支持CPU（numpy）和GPU（cupy）两种后端
        - 当指定xp参数时，优先使用指定的数组模块进行计算
        - 这对于GPU训练至关重要，因为输入img可能是numpy数组而权重W是cupy数组
        - 如果xp与img的实际类型不匹配，需要先将img转换为xp对应的类型
    """
    # 如果传入的是 Tensor，提取底层数据
    if isinstance(img, Tensor):
        img = img.data

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    DH, DW = pair(dilation)
    OH = get_conv_outsize(H, KH, SH, PH, DH)
    OW = get_conv_outsize(W, KW, SW, PW, DW)

    # 关键修复：优先使用传入的xp参数，而非从img推断
    # 这样可以确保当img是numpy但权重是cupy时，使用cupy进行计算
    if xp is None:
        xp = get_array_module(img)
    else:
        # 当指定了xp但img类型不匹配时，需要先将img转换为xp对应的类型
        # 注意：img可能是Tensor对象，需要先提取其data
        img_data = img.data if isinstance(img, Tensor) else img
        img_xp = get_array_module(img_data)
        if img_xp != xp:
            # 根据目标xp模块正确转换数组类型
            # 当xp是numpy但img是cupy时，需要使用cupy.asnumpy()
            # 当xp是cupy但img是numpy时，需要使用cupy.asarray()
            if xp is np:
                # 目标xp是numpy，源是cupy
                img_data = cp.asnumpy(img_data)
            else:
                # 目标xp是cupy，源是numpy
                img_data = xp.asarray(img_data)
        img = img_data

    # CPU/GPU实现：使用NumPy/CuPy进行填充和patch提取
    img = xp.pad(img,
                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    col = xp.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    use_tiled = _use_cpu_tiled_kernel(xp, OH, OW)
    if use_tiled:
        tile_h, tile_w = _select_cpu_tile_shape(
            N, C, KH, KW, OH, OW, img.dtype.itemsize, _CPU_IM2COL_TILE
        )
        for oh0 in range(0, OH, tile_h):
            oh1 = OH if (oh0 + tile_h) > OH else (oh0 + tile_h)
            h_base = oh0 * SH
            oh_span = oh1 - oh0
            for ow0 in range(0, OW, tile_w):
                ow1 = OW if (ow0 + tile_w) > OW else (ow0 + tile_w)
                w_base = ow0 * SW
                ow_span = ow1 - ow0
                for j in range(KH):
                    j_base = j * DH + h_base
                    j_lim = j_base + SH * oh_span
                    for i in range(KW):
                        i_base = i * DW + w_base
                        i_lim = i_base + SW * ow_span
                        col[:, :, j, i, oh0:oh1, ow0:ow1] = img[:, :, j_base:j_lim:SH, i_base:i_lim:SW]
    else:
        # 计算每个patch在输入图像中的位置，支持扩张卷积
        for j in range(KH):
            j_lim = j * DH + SH * OH
            for i in range(KW):
                i_lim = i * DW + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j*DH:j_lim:SH, i*DW:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True, dilation=(1,1)):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    DH, DW = pair(dilation)
    OH = get_conv_outsize(H, KH, SH, PH, DH)
    OW = get_conv_outsize(W, KW, SW, PW, DW)

    # Ensure `col` has shape (N, C, KH, KW, OH, OW)
    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        
    xp = get_array_module(col)
    
    img = xp.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                   dtype=col.dtype)
    use_tiled = _use_cpu_tiled_kernel(xp, OH, OW)
    if use_tiled:
        tile_h, tile_w = _select_cpu_tile_shape(
            N, C, KH, KW, OH, OW, col.dtype.itemsize, _CPU_COL2IM_TILE
        )
        for oh0 in range(0, OH, tile_h):
            oh1 = OH if (oh0 + tile_h) > OH else (oh0 + tile_h)
            h_base = oh0 * SH
            oh_span = oh1 - oh0
            for ow0 in range(0, OW, tile_w):
                ow1 = OW if (ow0 + tile_w) > OW else (ow0 + tile_w)
                w_base = ow0 * SW
                ow_span = ow1 - ow0
                for j in range(KH):
                    j_base = j * DH + h_base
                    j_lim = j_base + SH * oh_span
                    for i in range(KW):
                        i_base = i * DW + w_base
                        i_lim = i_base + SW * ow_span
                        img[:, :, j_base:j_lim:SH, i_base:i_lim:SW] += col[:, :, j, i, oh0:oh1, ow0:ow1]
    else:
        #适合小图像处理
        # for oh in range(OH):
        #     for ow in range(OW):
        #         # 考虑填充偏移
        #         j_start = oh * SH
        #         i_start = ow * SW
        #         img[:, :, j_start:j_start+KH, i_start:i_start+KW] += col[:, :, :, :, oh, ow]
        for j in range(KH):
            j_lim = j * DH + SH * OH
            for i in range(KW):
                i_lim = i * DW + SW * OW
                img[:, :, j*DH:j_lim:SH, i*DW:i_lim:SW] += col[:, :, j, i, :, :]
    return img[:, :, PH:H + PH, PW:W + PW]



#正式函数#
#卷积函数和反卷积函数对称性强，大部分代码互为镜像
class Conv2d(Function):
    WINOGRAD_MIN_INPUT_SHAPE = (12, 12, 256, 256)
    FFT_MIN_KERNEL_SIZE = 5
    FFT_MIN_SPATIAL_SIZE = 32
    AUTOTUNE_ENABLED = True
    AUTOTUNE_TRAIN_ONLY = True
    _path_cache = {}
    _path_cache_max = 256

    def __init__(self, stride=(1,1), pad=(0,0), dilation=1, visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.dilation = pair(dilation)
        self.visualize = visualize

    def _gemm_output_shape(self, x, W):
        KH, KW = W.shape[2:]
        SH, SW = self.stride
        PH, PW = self.pad
        DH, DW = self.dilation
        N, _, H, W_in = x.shape
        OC = W.shape[0]
        OH = get_conv_outsize(H, KH, SH, PH, DH)
        OW = get_conv_outsize(W_in, KW, SW, PW, DW)
        return N, OC, KH, KW, OH, OW



    def _select_forward_path(self, x, W):
        """选择前向路径: winograd | fft | gemm | im2col。"""
        if len(x.shape) != 4 or len(W.shape) != 4:
            return 'im2col'

        N, C, H, W_in = x.shape
        _, _, KH, KW = W.shape
        SH, SW = self.stride
        DH, DW = self.dilation
        min_n, min_c, min_h, min_w = self.WINOGRAD_MIN_INPUT_SHAPE
        is_large_input = N >= min_n and C >= min_c and H >= min_h and W_in >= min_w

        # 3x3 stride=1 dilation=1 且大图优先 Winograd
        if KH == 3 and KW == 3 and SH == 1 and SW == 1 and DH == 1 and DW == 1 and is_large_input:
            return 'winograd'

        # 大核卷积在较大空间尺寸上优先 FFT
        # 保持保守：只在非空洞卷积时启用，避免路径选择过早偏向 FFT
        if DH == 1 and DW == 1:
            is_fft_kernel = (KH if KH >= KW else KW) >= self.FFT_MIN_KERNEL_SIZE
            is_fft_input = H >= self.FFT_MIN_SPATIAL_SIZE and W_in >= self.FFT_MIN_SPATIAL_SIZE
            if is_fft_kernel and is_fft_input:
                return 'fft'

        # 大图但不满足 Winograd 条件时：
        # 细长核(如 1x1 / 1x3 / 3x1)实测 im2col 更快，其余优先 GEMM
        if is_large_input:
            if KH == 1 or KW == 1:
                return 'im2col'
            return 'gemm'

        # 其余场景回退到原始 im2col 路径
        return 'im2col'

    def _path_cache_key(self, x, W):
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)
        return (
            x.shape,
            W_data.shape,
            self.stride,
            self.pad,
            self.dilation,
            xp.__name__,
        )

    def _autotune_select_path(self, x, W, b):
        default_path = self._select_forward_path(x, W)
        if not self.AUTOTUNE_ENABLED:
            return default_path
        if self.AUTOTUNE_TRAIN_ONLY and not Config.train:
            return default_path

        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)
        if xp is np:
            return default_path

        candidates = [default_path]

        N, C, H, W_in = x.shape
        _, _, KH, KW = W_data.shape
        SH, SW = self.stride
        DH, DW = self.dilation
        min_n, min_c, min_h, min_w = self.WINOGRAD_MIN_INPUT_SHAPE
        is_large_input = N >= min_n and C >= min_c and H >= min_h and W_in >= min_w

        if KH == 3 and KW == 3 and SH == 1 and SW == 1 and DH == 1 and DW == 1 and is_large_input:
            candidates.append('winograd')
        if DH == 1 and DW == 1:
            is_fft_kernel = (KH if KH >= KW else KW) >= self.FFT_MIN_KERNEL_SIZE
            is_fft_input = H >= self.FFT_MIN_SPATIAL_SIZE and W_in >= self.FFT_MIN_SPATIAL_SIZE
            if is_fft_kernel and is_fft_input and KH >= 7 and KW >= 7 and H >= 64 and W_in >= 64:
                candidates.append('fft')
        if is_large_input and not (KH == 1 or KW == 1):
            candidates.append('gemm')

        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            return default_path

        def run_path(path):
            if path == 'winograd':
                return self.winograd_conv2d_forward(x, W, b)
            if path == 'fft':
                return self.fft_conv2d_forward(x, W, b)
            if path == 'gemm':
                return self.gemm_conv2d_forward(x, W, b)
            return self.im2col_conv2d_forward(x, W, b)

        def time_path(path):
            if has_cupy:
                cp.cuda.Stream.null.synchronize()
            run_path(path)
            if has_cupy:
                cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            run_path(path)
            if has_cupy:
                cp.cuda.Stream.null.synchronize()
            end = time.perf_counter()
            return end - start

        best_path = default_path
        best_time = None
        for path in candidates:
            try:
                t = time_path(path)
            except Exception:
                continue
            if best_time is None or t < best_time:
                best_time = t
                best_path = path

        return best_path

    def _get_forward_path(self, x, W, b):
        cls = self.__class__
        key = self._path_cache_key(x, W)
        cached = cls._path_cache.get(key)
        if cached is not None:
            return cached

        path = self._autotune_select_path(x, W, b)
        cls._path_cache[key] = path
        if len(cls._path_cache) > cls._path_cache_max:
            cls._path_cache.pop(next(iter(cls._path_cache)))
        return path

    def forward(self, *xs):
        x = xs[0]
        #print(f"x.dtype = {x.dtype}")
        W = xs[1]
        b = xs[2]
        self._fw_workspace = None
        self._fw_workspace_version = None
        self._used_fft = False
        path = self._get_forward_path(x, W, b)
        self._used_winograd = (path == 'winograd')
        if path == 'winograd':
            return self.winograd_conv2d_forward(x, W, b)
        elif path == 'fft':
            self._used_winograd = False
            self._used_fft = True
            return self.fft_conv2d_forward(x, W, b)
        elif path == 'gemm':
            self._used_winograd = False
            self._used_fft = False
            return self.gemm_conv2d_forward(x, W, b)
        else:
            self._used_winograd = False
            self._used_fft = False
            # 回退到原本基于 im2col + tensordot 的标准卷积
            return self.im2col_conv2d_forward(x, W, b)



    def backward(self, gys):
        x, W, b = self.inputs
        if getattr(self, '_used_winograd', False):
            self._used_winograd_backward = True
            return self.winograd_conv2d_backward(gys, x, W, b)

        self._used_winograd_backward = False

        # ==== gx ====
        if self.dilation != (1, 1):
            gy_data = gys.data if isinstance(gys, Tensor) else gys
            W_data = W.data if isinstance(W, Tensor) else W
            gx_np = conv2d_backward_input_array(
                gy_data,
                W_data,
                stride=self.stride,
                pad=self.pad,
                dilation=self.dilation,
                out_h=x.shape[2],
                out_w=x.shape[3],
            )
            gx = as_Tensor(gx_np)
        else:
            gx = deconv2d(gys, W, b=None, stride=self.stride, pad=self.pad,
                          outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gys)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gys.sum(axis=(0, 2, 3))
        return gx, gW, gb

    def winograd_conv2d_backward(self, gys, x, W, b):
        x_np = x.data if isinstance(x, Tensor) else x
        W_np = W.data if isinstance(W, Tensor) else W
        gy_np = gys.data if isinstance(gys, Tensor) else gys
        
        xp = get_array_module(x_np)

        if not isinstance(x_np, xp.ndarray):
            x_np = xp.array(x_np)
        if not isinstance(W_np, xp.ndarray):
            W_np = xp.array(W_np)
        if not isinstance(gy_np, xp.ndarray):
            gy_np = xp.array(gy_np)

        N, C, H_in, W_in = x_np.shape
        _, OC, out_h, out_w = gy_np.shape
        ph, pw = self.pad

        calc_dtype = np.result_type(x_np.dtype, W_np.dtype, gy_np.dtype)
        x_cast = x_np.astype(calc_dtype, copy=False)
        gy_cast = gy_np.astype(calc_dtype, copy=False)
        dtype_key = np.dtype(calc_dtype).str

        tile_h = (out_h + 1) // 2
        tile_w = (out_w + 1) // 2
        req_h = (tile_h - 1) * 2 + 4
        req_w = (tile_w - 1) * 2 + 4

        pad_bottom = req_h - H_in - ph
        if pad_bottom < 0:
            pad_bottom = 0
        pad_right = req_w - W_in - pw
        if pad_right < 0:
            pad_right = 0

        xh = H_in + ph + pad_bottom
        xw = W_in + pw + pad_right

        cls = self.__class__
        if not hasattr(cls, '_winograd_backward_workspace_cache'):
            cls._winograd_backward_workspace_cache = {}

        workspace_key = (N, C, OC, tile_h, tile_w, xh, xw, dtype_key)
        workspace = cls._winograd_backward_workspace_cache.get(workspace_key)
        if workspace is None:
            workspace = {
                'x_work': xp.empty((N, C, xh, xw), dtype=calc_dtype),
                'L': xp.empty((N, C, tile_h, tile_w, 4, 4), dtype=calc_dtype),
                'V': xp.empty((N, C, tile_h, tile_w, 4, 4), dtype=calc_dtype),
                'gy_buffer': xp.empty((N, OC, tile_h * 2, tile_w * 2), dtype=calc_dtype),
                'dM': xp.empty((N, OC, tile_h, tile_w, 4, 4), dtype=calc_dtype),
                'gU': xp.empty((OC, C, 4, 4), dtype=calc_dtype),
            }
            cls._winograd_backward_workspace_cache[workspace_key] = workspace
            if len(cls._winograd_backward_workspace_cache) > 2:
                cls._winograd_backward_workspace_cache.pop(next(iter(cls._winograd_backward_workspace_cache)))

        # 1) 优先复用前向 V（零拷贝）；校验失效则按需重算 V = B^T d B。
        V = None
        fw_workspace = getattr(self, '_fw_workspace', None)
        fw_version = getattr(self, '_fw_workspace_version', None)
        if fw_workspace is not None and fw_version is not None:
            current_version = fw_workspace.get('version', None)
            fw_v = fw_workspace.get('V', None)
            if (
                current_version == fw_version and
                fw_v is not None and
                fw_v.shape == (N, C, tile_h, tile_w, 4, 4) and
                fw_v.dtype == np.dtype(calc_dtype)
            ):
                V = fw_v

        if V is None:
            x_work = workspace['x_work']
            h0, h1 = ph, ph + H_in
            w0, w1 = pw, pw + W_in
            x_work[:, :, h0:h1, w0:w1] = x_cast
            if h0 > 0:
                x_work[:, :, :h0, :] = 0
            if w0 > 0:
                x_work[:, :, :, :w0] = 0
            if h1 < xh:
                x_work[:, :, h1:, :] = 0
            if w1 < xw:
                x_work[:, :, :, w1:] = 0

            tiles = xp.lib.stride_tricks.sliding_window_view(x_work, (4, 4), axis=(2, 3))
            d = tiles[:, :, 0:2 * tile_h:2, 0:2 * tile_w:2, :, :]

            d0 = d[..., 0, :]
            d1 = d[..., 1, :]
            d2 = d[..., 2, :]
            d3 = d[..., 3, :]

            L = workspace['L']
            L[..., 0, :] = d0 - d2
            L[..., 1, :] = d1 + d2
            L[..., 2, :] = d2 - d1
            L[..., 3, :] = d1 - d3

            V = workspace['V']
            V[..., :, 0] = L[..., :, 0] - L[..., :, 2]
            V[..., :, 1] = L[..., :, 1] + L[..., :, 2]
            V[..., :, 2] = L[..., :, 2] - L[..., :, 1]
            V[..., :, 3] = L[..., :, 1] - L[..., :, 3]

        # 2) 反传常量。
        half = xp.array(0.5, dtype=calc_dtype)

        # 3) dY -> dM（与前向 Y = A^T M A 显式公式对应）。
        gy_h = tile_h * 2
        gy_w = tile_w * 2
        if out_h == gy_h and out_w == gy_w:
            gy_src = gy_cast
        else:
            gy_buffer = workspace['gy_buffer']
            gy_buffer.fill(0)
            gy_buffer[:, :, :out_h, :out_w] = gy_cast
            gy_src = gy_buffer

        gy00 = gy_src[:, :, 0::2, 0::2]
        gy01 = gy_src[:, :, 0::2, 1::2]
        gy10 = gy_src[:, :, 1::2, 0::2]
        gy11 = gy_src[:, :, 1::2, 1::2]

        dM = workspace['dM']
        dM0 = dM[..., 0, :]
        dM3 = dM[..., 3, :]

        dM0[..., 0] = gy00
        dM0[..., 1] = gy00 + gy01
        dM0[..., 2] = gy00 - gy01
        dM0[..., 3] = -gy01

        dM3[..., 0] = -gy10
        dM3[..., 1] = -(gy10 + gy11)
        dM3[..., 2] = -(gy10 - gy11)
        dM3[..., 3] = gy11

        dM[..., 1, :] = dM0 - dM3
        dM[..., 2, :] = dM0 + dM3

        # 4) dU：M = U @ V 的反传（仅计算 gU 用于 gW）。
        gU = workspace['gU']
        xp.einsum('nohwab,nchwab->ocab', dM, V, optimize=False, out=gU)

        # 5) gW：U = G g G^T 的显式反传。
        S0 = gU[..., 0, :] + half * (gU[..., 1, :] + gU[..., 2, :])
        S1 = half * (gU[..., 1, :] - gU[..., 2, :])
        S2 = half * (gU[..., 1, :] + gU[..., 2, :]) + gU[..., 3, :]

        gW_np = xp.empty((OC, C, 3, 3), dtype=calc_dtype)
        gW_np[..., 0, 0] = S0[..., 0] + half * (S0[..., 1] + S0[..., 2])
        gW_np[..., 0, 1] = half * (S0[..., 1] - S0[..., 2])
        gW_np[..., 0, 2] = half * (S0[..., 1] + S0[..., 2]) + S0[..., 3]
        gW_np[..., 1, 0] = S1[..., 0] + half * (S1[..., 1] + S1[..., 2])
        gW_np[..., 1, 1] = half * (S1[..., 1] - S1[..., 2])
        gW_np[..., 1, 2] = half * (S1[..., 1] + S1[..., 2]) + S1[..., 3]
        gW_np[..., 2, 0] = S2[..., 0] + half * (S2[..., 1] + S2[..., 2])
        gW_np[..., 2, 1] = half * (S2[..., 1] - S2[..., 2])
        gW_np[..., 2, 2] = half * (S2[..., 1] + S2[..., 2]) + S2[..., 3]

        # 6) gx：与 deconv2d 前向等价的直接数组实现，避免 Function 调度开销。
        kh, kw = W_np.shape[2:]
        gcol = xp.tensordot(W_np, gy_cast, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        gx_np = col2im_array(
            gcol,
            (N, C, x.shape[2], x.shape[3]),
            (kh, kw),
            self.stride,
            self.pad,
            to_matrix=False,
        )
        gx = as_Tensor(gx_np.astype(x_np.dtype, copy=False))

        gb = None
        if b is not None and getattr(b, 'data', None) is not None:
            b_dtype = b.data.dtype if isinstance(b.data, np.ndarray) else calc_dtype
            gb_np = gy_cast.sum(axis=(0, 2, 3)).astype(b_dtype, copy=False)
            gb = as_Tensor(gb_np)

        gW = as_Tensor(gW_np.astype(W_np.dtype, copy=False))
        self._fw_workspace = None
        self._fw_workspace_version = None
        return gx, gW, gb

    def im2col_conv2d_forward(self, x, W, b):
        """
        使用im2col方法实现卷积的前向传播。

        Args:
            x: 输入数据，形状 (N, C, H, W)
            W: 卷积核权重，形状 (OC, C, KH, KW)
            b: 偏置，形状 (OC,)，可为None

        Returns:
            卷积结果，形状 (N, OC, OH, OW)

        GPU适配说明:
            - 优先从权重W获取数组模块类型，因为输出格式由权重决定
            - 当模型在GPU上但输入数据还在CPU上时，W会是cupy数组
            - 需要将xp设置为cupy，以便正确执行tensordot操作
        """
        KH, KW = W.shape[2:]

        # 关键修复：优先从权重W获取数组模块，而非输入x
        # 这样可以确保当x是numpy但W是cupy时，使用cupy进行计算
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 使用指定的xp模块执行im2col，确保所有操作在同一个设备上
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation, xp=xp)
        y = xp.tensordot(col, W_data, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b.data if isinstance(b, Tensor) else b
        y = xp.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def fft_conv2d_forward(self, x, W, b):
        """使用 FFT 实现卷积前向传播。"""
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        x_data = to_xp(x, xp)
        W_data = to_xp(W, xp)

        if not isinstance(x_data, xp.ndarray):
            x_data = xp.array(x_data)
        if not isinstance(W_data, xp.ndarray):
            W_data = xp.array(W_data)

        SH, SW = self.stride
        PH, PW = self.pad
        if self.dilation != (1, 1):
            return self.im2col_conv2d_forward(x, W, b)

        N, C, H, W_in = x_data.shape
        OC, _, KH, KW = W_data.shape

        OH = get_conv_outsize(H, KH, SH, PH, 1)
        OW = get_conv_outsize(W_in, KW, SW, PW, 1)
        if OH <= 0 or OW <= 0:
            return xp.zeros((N, OC, builtins.max(OH, 0), builtins.max(OW, 0)), dtype=x_data.dtype)

        x_pad = xp.pad(x_data, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')
        fft_h = x_pad.shape[2] + KH - 1
        fft_w = x_pad.shape[3] + KW - 1

        # FFT 对应的是卷积；当前实现是 cross-correlation，因此先翻转核。
        W_flip = W_data[:, :, ::-1, ::-1]
        try:
            x_freq = xp.fft.rfftn(x_pad, s=(fft_h, fft_w), axes=(2, 3))
            w_freq = xp.fft.rfftn(W_flip, s=(fft_h, fft_w), axes=(2, 3))
            y_freq = xp.einsum('nchw,ochw->nohw', x_freq, w_freq, optimize=True)
            y_full = xp.fft.irfftn(y_freq, s=(fft_h, fft_w), axes=(2, 3))
        except Exception as exc:
            if has_cupy and isinstance(exc, cp.cuda.memory.OutOfMemoryError):
                return self.im2col_conv2d_forward(x, W, b)
            raise

        h_start = KH - 1
        w_start = KW - 1
        y = y_full[:, :, h_start:h_start + OH * SH:SH, w_start:w_start + OW * SW:SW]

        if b is not None:
            b_data = to_xp(b, xp)
            y += b_data.reshape(1, -1, 1, 1)

        return y.astype(x_data.dtype, copy=False)

    def gemm_conv2d_forward(self, x, W, b):
        """
        使用GEMM方法实现卷积的前向传播（基于滑动窗口视图）。

        Args:
            x: 输入数据，形状 (N, C, H, W)
            W: 卷积核权重，形状 (OC, C, KH, KW)
            b: 偏置，形状 (OC,)，可为None

        Returns:
            卷积结果，形状 (N, OC, OH, OW)

        GPU适配说明:
            - 优先从权重W获取数组模块类型，因为输出格式由权重决定
            - 当模型在GPU上但输入数据还在CPU上时，W会是cupy数组
            - 需要将xp设置为cupy，以便正确执行einsum操作
        """
        N, OC, KH, KW, OH, OW = self._gemm_output_shape(x, W)

        # 关键修复：优先从权重W获取数组模块，而非输入x
        # 这样可以确保当x是numpy但W是cupy时，使用cupy进行计算
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        if OH <= 0 or OW <= 0:
            oh = OH if OH > 0 else 0
            ow = OW if OW > 0 else 0
            return xp.zeros((N, OC, builtins.max(oh, 0), builtins.max(ow, 0)), dtype=x.dtype)

        SH, SW = self.stride
        PH, PW = self.pad
        DH, DW = self.dilation

        # 如果x不是xp类型的数组，需要先转换
        # 使用to_xp辅助函数正确处理numpy/cupy之间的转换
        x_data = to_xp(x, xp)

        x_pad = xp.pad(x_data, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')
        eff_kh = DH * (KH - 1) + 1
        eff_kw = DW * (KW - 1) + 1

        windows = xp.lib.stride_tricks.sliding_window_view(x_pad, (eff_kh, eff_kw), axis=(2, 3))
        windows = windows[:, :, :OH * SH:SH, :OW * SW:SW, :, :]
        patches = windows[..., ::DH, ::DW]

        y = xp.einsum('nchwkl,ockl->nohw', patches, W_data, optimize=True)
        if b is not None:
            b_data = to_xp(b, xp)
            y += b_data.reshape(1, -1, 1, 1)
        return y

    def winograd_conv2d_forward(self, x, W, b):
        # --- 1. 数据提取与类型转换 ---
        x_np = x.data if isinstance(x, Tensor) else x
        W_np = W.data if isinstance(W, Tensor) else W
        
        xp = get_array_module(x_np)

        if not isinstance(x_np, xp.ndarray):
            x_np = xp.array(x_np)
        if not isinstance(W_np, xp.ndarray):
            W_np = xp.array(W_np)

        N, C, H_in, W_in = x_np.shape
        OC, _, KH, KW = W_np.shape

        ph, pw = self.pad
        dtype = x_np.dtype
        calc_dtype = xp.result_type(x_np.dtype, W_np.dtype)

        # --- 2. Padding 与 tile 布局 ---
        out_h = (H_in + 2 * ph - KH) + 1
        out_w = (W_in + 2 * pw - KW) + 1

        if out_h <= 0 or out_w <= 0:
            oh = out_h if out_h > 0 else 0
            ow = out_w if out_w > 0 else 0
            return xp.zeros((N, OC, oh, ow), dtype=dtype)

        tile_h = (out_h + 1) // 2
        tile_w = (out_w + 1) // 2

        req_h = (tile_h - 1) * 2 + 4
        req_w = (tile_w - 1) * 2 + 4

        pad_bottom = req_h - H_in - ph
        if pad_bottom < 0:
            pad_bottom = 0
        pad_right = req_w - W_in - pw
        if pad_right < 0:
            pad_right = 0

        x_cast = x_np.astype(calc_dtype, copy=False)
        xh = H_in + ph + pad_bottom
        xw = W_in + pw + pad_right
        W_work = W_np.astype(calc_dtype, copy=False)

        # --- 3. 变换矩阵（按 dtype 缓存） ---
        cls = self.__class__
        if not hasattr(cls, '_winograd_consts_by_dtype'):
            cls._winograd_consts_by_dtype = {}
        if not hasattr(cls, '_winograd_u_cache'):
            cls._winograd_u_cache = {}
        if not hasattr(cls, '_winograd_workspace_cache'):
            cls._winograd_workspace_cache = {}
        
        dtype_key = xp.dtype(calc_dtype).str
        workspace_key = (N, C, OC, tile_h, tile_w, xh, xw, dtype_key)
        workspace = cls._winograd_workspace_cache.get(workspace_key)
        if workspace is None:
            tile_count = N * tile_h * tile_w
            workspace = {
                'x_work': xp.empty((N, C, xh, xw), dtype=calc_dtype),
                'L': xp.empty((N, C, tile_h, tile_w, 4, 4), dtype=calc_dtype),
                'V': xp.empty((N, C, tile_h, tile_w, 4, 4), dtype=calc_dtype),
                'M16': xp.empty((16, OC, tile_count), dtype=calc_dtype),
                'y_buffer': xp.empty((N, OC, tile_h * 2, tile_w * 2), dtype=calc_dtype),
                'version': 0,
            }
            cls._winograd_workspace_cache[workspace_key] = workspace
            if len(cls._winograd_workspace_cache) > 4:
                cls._winograd_workspace_cache.pop(next(iter(cls._winograd_workspace_cache)))

        x_work = workspace['x_work']
        h0, h1 = ph, ph + H_in
        w0, w1 = pw, pw + W_in
        x_work[:, :, h0:h1, w0:w1] = x_cast
        if h0 > 0:
            x_work[:, :, :h0, :] = 0
        if w0 > 0:
            x_work[:, :, :, :w0] = 0
        if h1 < xh:
            x_work[:, :, h1:, :] = 0
        if w1 < xw:
            x_work[:, :, :, w1:] = 0
        consts = cls._winograd_consts_by_dtype.get(dtype_key)
        if consts is None:
            half = xp.array(0.5, dtype=calc_dtype)
            one = xp.array(1.0, dtype=calc_dtype)
            consts = (half, one)
            cls._winograd_consts_by_dtype[dtype_key] = consts
        half, _ = consts

        # --- 4. 权重变换 U = G g G^T（仅用稳定轻量 key 缓存） ---
        # 这里不回退到 im2col；仅优化 Winograd 本路径。
        w_ptr = int(W_work.__array_interface__['data'][0])
        u_cache_key = (w_ptr, W_work.shape, W_work.dtype.str, dtype_key)
        cached_u = cls._winograd_u_cache.get(u_cache_key)
        if cached_u is None:
            g0 = W_work[:, :, 0, :]
            g1 = W_work[:, :, 1, :]
            g2 = W_work[:, :, 2, :]

            T = xp.empty((OC, C, 4, 3), dtype=calc_dtype)
            T[:, :, 0, :] = g0
            T[:, :, 1, :] = half * (g0 + g1 + g2)
            T[:, :, 2, :] = half * (g0 - g1 + g2)
            T[:, :, 3, :] = g2

            U = xp.empty((OC, C, 4, 4), dtype=calc_dtype)
            U[:, :, :, 0] = T[:, :, :, 0]
            U[:, :, :, 1] = half * (T[:, :, :, 0] + T[:, :, :, 1] + T[:, :, :, 2])
            U[:, :, :, 2] = half * (T[:, :, :, 0] - T[:, :, :, 1] + T[:, :, :, 2])
            U[:, :, :, 3] = T[:, :, :, 2]

            U16 = xp.ascontiguousarray(U.reshape(OC, C, 16).transpose(2, 0, 1))
            cls._winograd_u_cache[u_cache_key] = (U, U16)
            if len(cls._winograd_u_cache) > 8:
                cls._winograd_u_cache.pop(next(iter(cls._winograd_u_cache)))
        else:
            U, U16 = cached_u

        # --- 5. 输入 tile 变换 V = B^T d B（显式向量化公式） ---
        tiles = xp.lib.stride_tricks.sliding_window_view(x_work, (4, 4), axis=(2, 3))
        d = tiles[:, :, 0:2 * tile_h:2, 0:2 * tile_w:2, :, :]

        d0 = d[..., 0, :]
        d1 = d[..., 1, :]
        d2 = d[..., 2, :]
        d3 = d[..., 3, :]

        L = workspace['L']
        L[..., 0, :] = d0 - d2
        L[..., 1, :] = d1 + d2
        L[..., 2, :] = d2 - d1
        L[..., 3, :] = d1 - d3

        V = workspace['V']
        V[..., :, 0] = L[..., :, 0] - L[..., :, 2]
        V[..., :, 1] = L[..., :, 1] + L[..., :, 2]
        V[..., :, 2] = L[..., :, 2] - L[..., :, 1]
        V[..., :, 3] = L[..., :, 1] - L[..., :, 3]

        workspace['version'] = workspace.get('version', 0) + 1
        self._fw_workspace = workspace
        self._fw_workspace_version = workspace['version']

        # --- 6. 核心乘法 M（全量 matmul，避免分块循环开销） ---
        V16T = V.reshape(N, C, tile_h, tile_w, 16).transpose(4, 1, 0, 2, 3)
        V16T = xp.ascontiguousarray(V16T.reshape(16, C, N * tile_h * tile_w))

        M16 = workspace['M16']
        np.matmul(U16, V16T, out=M16)
        M = M16.reshape(4, 4, OC, N, tile_h, tile_w).transpose(3, 2, 4, 5, 0, 1)

        # --- 7. 输出逆变换 Y = A^T M A（显式向量化公式） ---
        r0 = M[..., 0, :] + M[..., 1, :] + M[..., 2, :]
        r1 = M[..., 1, :] - M[..., 2, :] - M[..., 3, :]

        Y00 = r0[..., 0] + r0[..., 1] + r0[..., 2]
        Y01 = r0[..., 1] - r0[..., 2] - r0[..., 3]
        Y10 = r1[..., 0] + r1[..., 1] + r1[..., 2]
        Y11 = r1[..., 1] - r1[..., 2] - r1[..., 3]

        y_buffer = workspace['y_buffer']
        y_buffer[:, :, 0::2, 0::2] = Y00
        y_buffer[:, :, 0::2, 1::2] = Y01
        y_buffer[:, :, 1::2, 0::2] = Y10
        y_buffer[:, :, 1::2, 1::2] = Y11

        y_res = y_buffer[:, :, :out_h, :out_w]

        if b is not None:
            b_val = b.data if isinstance(b, Tensor) else b
            if b_val is not None:
                if not isinstance(b_val, xp.ndarray):
                    b_val = xp.array(b_val)
                y_res += b_val.reshape(1, -1, 1, 1)

        return y_res.astype(dtype, copy=False)

def conv2d(x, W, b=None, stride=(1,1), pad=(0,0), dilation=1, visualize=False):
    return Conv2d(stride, pad, dilation, visualize)(x, W, b)


def conv2d_backward_input_array(gy, W, stride=(1, 1), pad=(0, 0), dilation=(1, 1), out_h=None, out_w=None):
    """计算卷积对输入 x 的梯度（支持 dilation）。

    Args:
        gy: 上游梯度，形状 (N, OC, OH, OW)
        W: 卷积核，形状 (OC, C, KH, KW)
    """
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    DH, DW = pair(dilation)

    N, OC, OH, OW = gy.shape
    OC_W, C, KH, KW = W.shape
    assert OC == OC_W

    if out_h is None or out_w is None:
        out_h = SH * (OH - 1) - 2 * PH + DH * (KH - 1) + 1
        out_w = SW * (OW - 1) - 2 * PW + DW * (KW - 1) + 1
        
    xp = get_array_module(gy)

    gx_pad = xp.zeros((N, C, out_h + 2 * PH + SH - 1, out_w + 2 * PW + SW - 1), dtype=gy.dtype)

    for kh in range(KH):
        h_start = kh * DH
        h_end = h_start + SH * OH
        for kw in range(KW):
            w_start = kw * DW
            w_end = w_start + SW * OW

            # (N, OC, OH, OW) x (OC, C) -> (N, OH, OW, C) -> (N, C, OH, OW)
            contrib = xp.tensordot(gy, W[:, :, kh, kw], axes=(1, 0)).transpose(0, 3, 1, 2)
            gx_pad[:, :, h_start:h_end:SH, w_start:w_end:SW] += contrib

    return gx_pad[:, :, PH:PH + out_h, PW:PW + out_w]


class GroupedConv2d(Function):
    def __init__(self, stride=(1,1), pad=(0,0), groups=1, dilation=1, visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.groups = groups
        self.dilation = pair(dilation)
        self.visualize = visualize

    def forward(self, *xs):
        """
        分组卷积的前向传播。

        Args:
            x: 输入数据，形状 (N, C, H, W)
            W: 权重，形状 (OC, C_per_group, KH, KW)
            b: 偏置，可为None

        Returns:
            输出数据

        GPU适配说明:
            - 优先从权重W获取数组模块类型
            - 当W是cupy数组时，需要先将x转换为cupy数组
        """
        x = xs[0]
        W = xs[1]
        b = xs[2]

        N, C, H, W_in = x.shape
        OC, C_per_group, KH, KW = W.shape

        assert C % self.groups == 0, "Input channels must be divisible by groups"
        assert OC % self.groups == 0, "Output channels must be divisible by groups"

        # 关键修复：优先从权重W获取数组模块，而非输入x
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 如果x不是xp类型的数组，需要先转换
        # 使用to_xp辅助函数正确处理numpy/cupy之间的转换
        x_data = to_xp(x, xp)

        OC_per_group = OC // self.groups
        OH = get_conv_outsize(H, KH, self.stride[0], self.pad[0], self.dilation[0])
        OW = get_conv_outsize(W_in, KW, self.stride[1], self.pad[1], self.dilation[1])
        y = xp.zeros((N, OC, OH, OW), dtype=x_data.dtype)

        for i in range(self.groups):
            x_group = x_data[:, i*C_per_group:(i+1)*C_per_group, :, :]
            W_group = W_data[i*OC_per_group:(i+1)*OC_per_group, :, :, :]

            col = im2col_array(
                x_group, (KH, KW), self.stride, self.pad,
                to_matrix=False, dilation=self.dilation, xp=xp
            )
            y_group = xp.tensordot(col, W_group, ((1, 2, 3), (1, 2, 3)))
            y_group = xp.rollaxis(y_group, 3, 1)
            y[:, i*OC_per_group:(i+1)*OC_per_group, :, :] = y_group

        if b is not None:
            b_data = to_xp(b, xp)
            y += b_data.reshape(1, -1, 1, 1)

        return y

    def backward(self, gys):
        x, W, b = self.inputs
        x_data = x.data
        W_data = W.data
        gy = gys.data

        N, C, H, W_in = x_data.shape
        OC, C_per_group, KH, KW = W_data.shape
        OC_per_group = OC // self.groups
        
        xp = get_array_module(gy)

        gx = xp.zeros_like(x_data)
        gW = xp.zeros_like(W_data)

        for i in range(self.groups):
            c0, c1 = i * C_per_group, (i + 1) * C_per_group
            oc0, oc1 = i * OC_per_group, (i + 1) * OC_per_group

            x_group = x_data[:, c0:c1, :, :]
            gy_group = gy[:, oc0:oc1, :, :]
            W_group = W_data[oc0:oc1, :, :, :]

            gx_group = conv2d_backward_input_array(
                gy_group,
                W_group,
                stride=self.stride,
                pad=self.pad,
                dilation=self.dilation,
                out_h=H,
                out_w=W_in,
            )
            gx[:, c0:c1, :, :] += gx_group

            col = im2col_array(
                x_group, (KH, KW), self.stride, self.pad,
                to_matrix=False, dilation=self.dilation
            )
            gW_group = xp.tensordot(gy_group, col, ((0, 2, 3), (0, 4, 5)))
            gW[oc0:oc1, :, :, :] = gW_group

        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return as_Tensor(gx), as_Tensor(gW), (as_Tensor(gb) if gb is not None else None)


def grouped_conv2d(x, W, b=None, stride=(1,1), pad=(0,0), groups=1, dilation=1, visualize=False):
    return GroupedConv2d(stride, pad, groups, dilation, visualize)(x, W, b)


# 深度卷积（本质上是 groups == in_channels 的分组卷积）
def depthwise_conv2d(x, W, b=None, stride=(1,1), pad=(0,0), dilation=1, visualize=False):
    x = as_Tensor(x)
    groups = x.shape[1]
    return grouped_conv2d(x, W, b, stride=stride, pad=pad, groups=groups, dilation=dilation, visualize=visualize)


class Deconv2d(Function):
    def __init__(self, stride=(1,1), pad=(0,0), outsize=None, dilation=1, visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize
        self.dilation = pair(dilation)
        self.visualize = visualize


    def forward(self, *xs):
        """
        转置卷积的前向传播。

        Args:
            x: 输入数据，形状 (N, C, H, W)
            W: 权重，形状 (C, OC, KH, KW)
            b: 偏置，可为None

        Returns:
            输出数据

        GPU适配说明:
            - 优先从权重W获取数组模块类型
            - 当W是cupy数组时，需要先将x转换为cupy数组
        """
        x = xs[0]
        W = xs[1]
        b = xs[2]
        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        DH, DW = self.dilation
        C, OC, KH, KW = Weight.shape
        N, C, H, W_in = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W_in, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        # 关键修复：优先从权重W获取数组模块，而非输入x
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 如果x不是xp类型的数组，需要先转换
        # 使用to_xp辅助函数正确处理numpy/cupy之间的转换
        x_data = to_xp(x, xp)

        gcol = xp.tensordot(W_data, x_data, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        if b is not None:
            self.no_bias = True
            b_data = to_xp(b, xp)
            y += b_data.reshape((1, b_data.size, 1, 1))
        return y

    def backward(self, gys):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gys, W, b=None, stride=self.stride, pad=self.pad, dilation=self.dilation)
        # ==== gW ====
        gW = Conv2DGradW(self)(gys, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gys.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=(1,1), pad=(0,0), outsize=None, dilation=1, visualize=False):
    return Deconv2d(stride, pad, outsize, dilation, visualize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d,visualize=False):
        super().__init__()
        self.visualize = visualize
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad
        self.dilation = getattr(conv2d, 'dilation', (1, 1))

    def forward(self, *xs):
        """
        计算卷积核梯度的前向传播。

        Args:
            x: 输入数据，形状 (N, C, H, W)
            gy: 梯度输出，形状 (N, OC, OH, OW)

        Returns:
            权重梯度

        GPU适配说明:
            - 优先从gy获取数组模块类型
            - 当gy是cupy数组时，需要先将x和col转换为cupy数组
        """
        x = xs[0]
        gy = xs[1]

        gy_data = gy.data if isinstance(gy, Tensor) else gy
        xp = get_array_module(gy_data)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False, dilation=self.dilation, xp=xp)
        gW = xp.tensordot(gy_data, col, ((0, 2, 3), (0, 4, 5)))
        return gW
    #貌似用不上 也就是gw关于gy和x的倒数
        def backward(self, gys):
             pass
    #     x, gy = self.inputs
    #     gW, = self.outputs

    #     xh, xw = x.shape[2:]
    #     gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
    #                   outsize=(xh, xw))
    #     ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
    #     return gx, ggy
    
#改变维度，池化操作找到KH * KW最大值的位置
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0, visualize=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.visualize = visualize

    def forward(self, *xs):
        x = xs[0]
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gys):
        return Pooling2DGrad(self)(gys)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d,visualize=False):
        super().__init__()
        self.visualize = visualize
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, *xs):
        gy = xs[0]
        

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)
        
        xp = get_array_module(gy)
        
        #与池化操作的维度相同 
        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        #也可以用transpose
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, gys):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(gys)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d,visualize=False):
        super().__init__()
        self.visualize = visualize
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, *xs):
        x = xs[0]
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0, visualize=False):
    return Pooling(kernel_size, stride, pad, visualize)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0, visualize=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, *xs):
        x = xs[0]
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gys):
        N, C, OH, OW = gys.shape
        KW, KH = pair(self.kernel_size)
        gys /= (KW*KH)
        gcol = broadcast_to(gys.reshape(-1), (KH, KW, N*C*OH*OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)

class GlobalAveragePooling(Function):
    def forward(self, *xs):
        x = xs[0]
        self.input_shape = x.shape
        y = x.mean(axis=(2, 3), keepdims=True)
        return y
    
    def backward(self, gy):
        # 全局平均池化的反向传播是将梯度广播回原始输入形状
        gx = broadcast_to(gy, self.input_shape)
        return gx

def global_average_pooling(x):
    return GlobalAveragePooling()(x)

class BatchNormFunction(Function):
    # 注意：该类已被弃用，请使用 BatchNorm2d 替代
    def __init__(self, eps=1e-5, momentum=0.9, training=True, moving_mean=None, moving_var=None):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = training
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.mean = None
        self.var = None
        self.x_hat = None
    
    def forward(self, *xs):
        x, gamma, beta = xs
        
        if self.training:
            # 计算当前批次的均值和方差
            if len(x.shape) == 2:
                # 全连接层
                self.mean = x.mean(axis=0, keepdims=True)
                self.var = ((x - self.mean) ** 2).mean(axis=0, keepdims=True)
            else:
                # 卷积层
                self.mean = x.mean(axis=(0, 2, 3), keepdims=True)
                self.var = ((x - self.mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            
            # 更新移动平均
            if self.moving_mean is not None:
                self.moving_mean[:] = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
            if self.moving_var is not None:
                self.moving_var[:] = self.momentum * self.moving_var + (1 - self.momentum) * self.var
            
            # 使用当前批次的均值和方差
            mean = self.mean
            var = self.var
        else:
            # 预测模式，使用移动平均
            mean = self.moving_mean
            var = self.moving_var
        
        # 标准化
        self.x_hat = (x - mean) / (var + self.eps) ** 0.5
        
        # 缩放和移位
        y = gamma * self.x_hat + beta
        return y
    
    def backward(self, gy):
        # 只处理y的梯度，忽略moving_mean和moving_var的梯度
        # 因为它们不是可训练参数
        x, gamma, beta = self.inputs
        
        if len(x.shape) == 4:
            axes = (0, 2, 3)
            m = x.shape[0] * x.shape[2] * x.shape[3]
        else:
            axes = (0,)
            m = x.shape[0]

        # 对gamma和beta的梯度
        dbeta = gy.sum(axis=axes, keepdims=True)
        dgamma = (gy * self.x_hat).sum(axis=axes, keepdims=True)

        # 对x的梯度（标准 BN 公式）
        inv_std = as_Tensor(1.0 / np.sqrt(self.var + self.eps))
        x_hat = as_Tensor(self.x_hat)
        sum_gy = gy.sum(axis=axes, keepdims=True)
        sum_gy_xhat = (gy * x_hat).sum(axis=axes, keepdims=True)
        dx = (gamma * inv_std / m) * (m * gy - sum_gy - x_hat * sum_gy_xhat)
        
        return dx, dgamma, dbeta

def batch_norm(x, gamma, beta, moving_mean=None, moving_var=None, eps=1e-5, momentum=0.9, training=True):
    # 兼容旧接口：内部改为 BatchNorm2d 实现
    x = as_Tensor(x)
    gamma = as_Tensor(gamma)
    beta = as_Tensor(beta)

    is_2d = (x.ndim == 2)
    if is_2d:
        n, c = x.shape
        x_in = x.reshape(n, c, 1, 1)
    else:
        x_in = x
        c = x.shape[1]

    # 兼容 moving_mean / moving_var 既可能是 ndarray 也可能是 Parameter
    from .parameter import Parameter

    xp = get_array_module(x)

    if moving_mean is None:
        running_mean = Parameter(xp.zeros(c, dtype=x.data.dtype), name='running_mean')
        running_mean.requires_grad = False
    elif hasattr(moving_mean, 'data'):
        running_mean = moving_mean
    else:
        mm = xp.asarray(moving_mean)
        running_mean = Parameter(mm.reshape(c).astype(x.data.dtype), name='running_mean')
        running_mean.requires_grad = False

    if moving_var is None:
        running_var = Parameter(xp.ones(c, dtype=x.data.dtype), name='running_var')
        running_var.requires_grad = False
    elif hasattr(moving_var, 'data'):
        running_var = moving_var
    else:
        mv = xp.asarray(moving_var)
        running_var = Parameter(mv.reshape(c).astype(x.data.dtype), name='running_var')
        running_var.requires_grad = False

    y = batch_norm2d((x_in, gamma.reshape(c), beta.reshape(c)), running_mean, running_var, momentum, eps)

    # 若传入的是 ndarray，回写统计量，保持旧行为
    if moving_mean is not None and not hasattr(moving_mean, 'data'):
        if xp.asarray(moving_mean).ndim == 1:
            moving_mean[...] = running_mean.data
        else:
            moving_mean[...] = running_mean.data.reshape(moving_mean.shape)
    if moving_var is not None and not hasattr(moving_var, 'data'):
        if xp.asarray(moving_var).ndim == 1:
            moving_var[...] = running_var.data
        else:
            moving_var[...] = running_var.data.reshape(moving_var.shape)

    if is_2d:
        return y.reshape(n, c)
    return y
    


class BatchNorm2d(Function):
    def __init__(self, running_mean, running_var, momentum=0.9, eps=1e-5):
        super().__init__()
        self.running_mean = running_mean
        self.running_var = running_var
        self.momentum = momentum
        self.eps = eps

    def forward(self, *xs):
        # x: (N, C, H, W)
        # gamma, beta: (C,)
        x, gamma, beta = xs
        #print(f"x.dtype = {x.dtype}")
        N, C, H, W = x.shape
        self.x_shape = x.shape
        self.x = x

        # 计算当前 batch 的均值和方差
        # 在通道维度上计算，保留 H,W 用于广播
        mean = x.mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
        var = x.var(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)

        # 更新 running 统计量（训练时）
        if Config.train:
            # 优先从running_mean获取数组模块（因为它在GPU上）
            # 如果running_mean是numpy，则使用numpy
            running_mean_data = self.running_mean.data
            xp = get_array_module(running_mean_data)

            # 确保所有数组都是xp类型
            if xp is not np:
                # 使用to_xp确保所有数组都正确转换
                m = to_xp(mean.reshape(C), xp)
                v = to_xp(var.reshape(C), xp)
                running_mean_data = to_xp(running_mean_data, xp)
                running_var_data = to_xp(self.running_var.data, xp)
                momentum = xp.asarray(self.momentum)
                one_minus_momentum = xp.asarray(1) - momentum
            else:
                m = mean.reshape(C)
                v = var.reshape(C)
                running_var_data = self.running_var.data
                momentum = self.momentum
                one_minus_momentum = 1 - self.momentum

            new_mean = momentum * running_mean_data + one_minus_momentum * m
            new_var = momentum * running_var_data + one_minus_momentum * v

            # 更新running统计量
            self.running_mean.data = new_mean
            self.running_var.data = new_var

            # 保存当前 batch 的统计量用于反向传播
            self.mean = mean
            self.var = var
        else:
            # 测试模式：使用 running 统计量，形状广播到 (1, C, 1, 1)
            mean = self.running_mean.data.reshape(1, C, 1, 1)
            var = self.running_var.data.reshape(1, C, 1, 1)
            self.mean = mean
            self.var = var

        # 归一化
        # 优先从running_mean获取xp，确保一致性
        xp = get_array_module(self.running_mean.data)
        if xp is np:
            xp = get_array_module(x)

        # 确保所有数组都是xp类型
        if xp is not np:
            # 将所有数组转换为cupy
            x = to_xp(x, xp)
            mean = to_xp(mean, xp)
            var = to_xp(var, xp)
            gamma_data = to_xp(gamma, xp)
            beta_data = to_xp(beta, xp)
        else:
            gamma_data = gamma
            beta_data = beta

        x_hat = (x - mean) / xp.sqrt(var + self.eps)
        # 缩放和偏移
        out = gamma_data.reshape(1, C, 1, 1) * x_hat + beta_data.reshape(1, C, 1, 1)
        self.x_hat = x_hat
        self.gamma = gamma_data
        return out

    def backward(self, gys):
        x = self.x
        gamma = self.gamma.reshape(1, -1, 1, 1)  # (1, C, 1, 1)
        mean = self.mean
        var = self.var
        eps = self.eps
        N, C, H, W = x.shape
        M = N * H * W  # 每个通道的像素总数

        # 计算中间变量
        xp = get_array_module(x)
        std_inv = 1.0 / xp.sqrt(var + eps)
        x_hat = self.x_hat
        # 对 gamma 和 beta 的梯度
        gbeta = gys.sum(axis=(0, 2, 3), keepdims=False)  # (C,)
        ggamma = (gys * x_hat).sum(axis=(0, 2, 3), keepdims=False)  # (C,)

        # 对 x_hat 的梯度
        gx_hat = gys * gamma
        # 对 var 的梯度
        gvar = (gx_hat * (x - mean) * (-0.5) * std_inv**3).sum(axis=(0, 2, 3), keepdims=True)
        # 对 mean 的梯度
        gmean = (gx_hat * (-std_inv)).sum(axis=(0, 2, 3), keepdims=True) + \
                gvar * (-2.0 / M) * (x - mean).sum(axis=(0, 2, 3), keepdims=True)
        # 对输入 x 的梯度
        gx = gx_hat * std_inv + gvar * (2.0 / M) * (x - mean) + gmean / M

        # 返回 gx, ggamma, gbeta (顺序与 forward 输入一致)
        return as_Tensor(gx), as_Tensor(ggamma), as_Tensor(gbeta)

def batch_norm2d(x, running_mean, running_var, momentum=0.9, eps=1e-5):
    return BatchNorm2d(running_mean, running_var, momentum, eps)(*x)


class FusedConvReLU(Function):
    """
    融合 Conv2d + ReLU 的算子。
    前向：卷积后原地应用 ReLU，只保存掩码（bool 数组）。
    反向：利用掩码直接计算梯度，并复用底层 numpy 函数，不创建额外计算图节点。
    """
    def __init__(self, stride=(1,1), pad=(0,0), dilation=(1,1), visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.dilation = pair(dilation) # 扩张卷积参数
        self.visualize = visualize

    def forward(self, *xs):
        """
        融合 Conv2d + ReLU 的前向传播。

        Args:
            x: 输入数据
            W: 权重
            b: 偏置

        GPU适配说明:
            - 优先从权重W获取数组模块类型
            - 当W是cupy数组时，需要先将x转换为cupy数组
        """
        x, W, b = xs
        KH, KW = W.shape[2:]

        # 关键修复：优先从权重W获取数组模块，而非输入x
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 如果x不是xp类型的数组，需要先转换
        # 注意：x可能是Tensor对象，需要先提取其data
        x_data = to_xp(x, xp)

        # 1. im2col + 卷积
        col = im2col_array(x_data, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation, xp=xp)
        conv_out = xp.tensordot(col, W_data, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            b_data = to_xp(b, xp)
            conv_out += b_data
        conv_out = xp.rollaxis(conv_out, 3, 1)

        # 2. 融合 ReLU：计算掩码并原地修改 conv_out
        self.mask = conv_out > 0
        conv_out[conv_out < 0] = 0          # 原地 ReLU，conv_out 变为最终输出

        return conv_out

    def backward(self, gys):
        # gy: 输出梯度 Tensor (N, OC, OH, OW)
        x, W, b = self.inputs
        KH, KW = W.shape[2:]

        # ReLU 梯度：gys * mask
        g_conv = gys.data * self.mask

        # 1. 计算 gW: 使用 im2col(x) 与 g_conv 的 tensordot
        col_x = im2col_array(x.data, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation)
        gW = np.tensordot(g_conv, col_x, ((0,2,3), (0,4,5)))   # (OC, C, KH, KW)

        # 2. 计算 gb (如果有偏置)
        gb = None
        if b is not None:
            gb = g_conv.sum(axis=(0,2,3))

        # 3. 计算 gx —— 改用支持 dilation 的函数
        gx = conv2d_backward_input_array(
            g_conv, W.data,
            stride=self.stride,
            pad=self.pad,
            dilation=self.dilation,
            out_h=x.shape[2],
            out_w=x.shape[3],
        )

        # 返回梯度（与 forward 输入顺序一致）
        return as_Tensor(gx), as_Tensor(gW), (as_Tensor(gb) if gb is not None else None)

def fused_conv_relu(x, W, b=None, stride=1, pad=0, dilation=1, visualize=False):
    return FusedConvReLU(stride, pad, dilation, visualize)(x, W, b)

class FusedConvBNReLU(Function):
    """
    融合 Conv2d + BatchNorm2d + ReLU 的算子。
    前向：卷积 → 批量归一化 → ReLU
    反向：ReLU 梯度 → BN 梯度 → 卷积梯度
    """
    def __init__(self, stride=(1,1), pad=(0,0), dilation=(1,1),
                  running_mean=None, running_var=None, momentum=0.9, eps=1e-5, visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.dilation = pair(dilation)   # 扩张卷积参数
        self.running_mean = running_mean
        self.running_var = running_var
        self.momentum = momentum
        self.eps = eps
        self.visualize = visualize

    def forward(self, *xs):
        """
        融合 Conv2d + BatchNorm2d + ReLU 的前向传播。

        Args:
            x: 输入数据
            W: 权重
            b: 偏置
            gamma: BN 缩放参数
            beta: BN 偏移参数

        GPU适配说明:
            - 优先从权重W获取数组模块类型
            - 当W是cupy数组时，需要先将x转换为cupy数组
        """
        # 输入：x, W, b, gamma, beta
        x, W, b, gamma, beta = xs
        OC, _, KH, KW = W.shape

        # 用于融合算子时的初始化
        if not hasattr(self,'outsize'):
            self.outsize = OC

        # 关键修复：优先从权重W获取数组模块，而非输入x
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 如果x不是xp类型的数组，需要先转换
        # 使用to_xp辅助函数正确处理numpy/cupy之间的转换
        x_data = to_xp(x, xp)

        # ---------- 1. 卷积 ----------
        # im2col
        col = im2col_array(x_data, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation, xp=xp)
        # 卷积输出 (N, OH, OW, OC)
        conv_out = xp.tensordot(col, W_data, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            b_data = to_xp(b, xp)
            conv_out += b_data
        conv_out = xp.rollaxis(conv_out, 3, 1)

        # 保存卷积输出和输入，用于反向
        self.x = x_data
        self.W = W_data
        self.b = b_data if b is not None else None
        self.conv_out = conv_out

        # ---------- 2. 批量归一化 ----------
        # 计算均值和方差 (在 (N, H, W) 维度上)
        mean = conv_out.mean(axis=(0,2,3), keepdims=True)   # (1, OC, 1, 1)
        var = conv_out.var(axis=(0,2,3), keepdims=True)     # (1, OC, 1, 1)

        # 训练时更新 running 统计量
        if Config.train:
            m = mean.reshape(OC)
            v = var.reshape(OC)
            
            if self.running_mean is None:
                self.running_mean = Tensor(xp.zeros(OC, dtype='float32'), requires_grad=False, name='running_mean')
            if self.running_var is None:
                self.running_var = Tensor(xp.ones(OC, dtype='float32'), requires_grad=False, name='running_mean')

            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * m
            self.running_var.data  = self.momentum * self.running_var.data  + (1 - self.momentum) * v
            self.mean = mean
            self.var = var
        else:
            # 测试模式：使用 running 统计量
            if self.running_mean is None:
                self.running_mean = Tensor(xp.zeros(OC, dtype='float32'), requires_grad=False, name='running_mean')
            if self.running_var is None:
                self.running_var = Tensor(xp.ones(OC, dtype='float32'), requires_grad=False, name='running_mean')

            mean = self.running_mean.data.reshape(1, OC, 1, 1)
            var = self.running_var.data.reshape(1, OC, 1, 1)
            self.mean = mean
            self.var = var

        # 归一化、缩放、平移
        std_inv = 1.0 / xp.sqrt(var + self.eps)
        x_hat = (conv_out - mean) * std_inv
        gamma_reshaped = gamma.reshape(1, OC, 1, 1)
        beta_reshaped  = beta.reshape(1, OC, 1, 1)
        bn_out = gamma_reshaped * x_hat + beta_reshaped

        # 保存 BN 中间变量
        self.gamma = gamma
        self.beta = beta
        self.x_hat = x_hat
        self.std_inv = std_inv

        # ---------- 3. ReLU ----------
        self.mask = bn_out > 0
        out = bn_out * self.mask

        return out

    def backward(self, gys):
        # gys: 输出梯度 (N, OC, OH, OW)
        # ---------- 1. ReLU 梯度 ----------
        g_relu = gys.data * self.mask   # 对 bn_out 的梯度

        # ---------- 2. BN 梯度 ----------
        x = self.conv_out               # 卷积输出，BN 的输入
        gamma = self.gamma.reshape(1, -1, 1, 1)
        mean = self.mean
        var = self.var
        eps = self.eps
        N, C, H, W = x.shape
        M = N * H * W   # 每个通道的像素总数
        OC, _, KH, KW = self.W.shape
        
        xp = get_array_module(x)

        # 计算 gamma 和 beta 的梯度
        gbeta = g_relu.sum(axis=(0,2,3), keepdims=False)          # (OC,)
        ggamma = (g_relu * self.x_hat).sum(axis=(0,2,3), keepdims=False)  # (OC,)

        # 对 x_hat 的梯度
        gx_hat = g_relu * gamma
        # 对方差和均值的梯度
        gvar = (gx_hat * (x - mean) * (-0.5) * self.std_inv**3).sum(axis=(0,2,3), keepdims=True)
        gmean = (gx_hat * (-self.std_inv)).sum(axis=(0,2,3), keepdims=True) + \
                gvar * (-2.0 / M) * (x - mean).sum(axis=(0,2,3), keepdims=True)
        # 对 BN 输入（即卷积输出）的梯度
        g_conv_out = gx_hat * self.std_inv + gvar * (2.0 / M) * (x - mean) + gmean / M

        # ---------- 3. 卷积梯度 ----------
        # 使用卷积的反向传播公式
        # gW: (OC, C, KH, KW)
        col_x = im2col_array(self.x, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation)
        gW = xp.tensordot(g_conv_out, col_x, ((0,2,3), (0,4,5)))   # (OC, C, KH, KW)

        # gb (如果有偏置)
        gb = None
        if self.b is not None:
            gb = g_conv_out.sum(axis=(0,2,3))

        # gx
        gx = conv2d_backward_input_array(
            g_conv_out, self.W,
            stride=self.stride,
            pad=self.pad,
            dilation=self.dilation,
            out_h=self.x.shape[2],
            out_w=self.x.shape[3],
        )

        # 返回梯度，顺序与 forward 输入一致
        # 返回：gx, gW, gb, ggamma, gbeta
        return (as_Tensor(gx), as_Tensor(gW), 
                as_Tensor(gb) if gb is not None else None,
                as_Tensor(ggamma), 
                as_Tensor(gbeta)
                )
    
def fused_conv_bn_relu(x, W, b, gamma, beta, running_mean, running_var, stride=1, pad=0, dilation=1, momentum=0.9, eps=1e-5, visualize=False):
    return FusedConvBNReLU(stride, pad, dilation, running_mean, running_var, momentum, eps, visualize)(x, W, b, gamma, beta)

class CastRegistry:
    '''
    注册自动精度
    resist_cast: 从Config.current_dtype提高到float32
    cancast: 降低精度至Config.current_dtype (一般是float16)
    '''
    cancast = [
        Tanh,
        MatMul,
        Linear,
        Sigmoid,
        ReLU,
        Conv2d,
        GroupedConv2d,
        Deconv2d,
        Pooling,
        AveragePooling,
        GlobalAveragePooling,
        FusedConvReLU
    ]

    resist_cast = [
        Sin,
        Cos,
        Exp,
        Log,
        Mean,
        Softmax,
        BatchNorm2d,
        FusedConvBNReLU
    ]

def type_precedence(dtype):
    order = {'b':0, 'i':1, 'u':2, 'f':3, 'c':4}
    return (order[dtype.kind], dtype.itemsize)

def compare_dtype_greater(dtype1, dtype2):
    if isinstance(dtype1, str):
        dtype1 = np.dtype(dtype1)
    if isinstance(dtype2, str):
        dtype2 = np.dtype(dtype2)
        
    return True if type_precedence(dtype1) > type_precedence(dtype2) else False

def compare_dtype_equal(dtype1, dtype2):
    if isinstance(dtype1, str):
        dtype1 = np.dtype(dtype1)
    if isinstance(dtype2, str):
        dtype2 = np.dtype(dtype2)

    return True if type_precedence(dtype1) == type_precedence(dtype2) else False

class Cast(Function):
    def __init__(self, dtype='float16'):
        super().__init__()
        self.dtype = dtype

    def forward(self, *xs):
        if len(xs) > 1:
            raise ValueError("Cast function only supports single input.")
        x = xs[0]
        # 改变精度
        x = x.astype(self.dtype) if not compare_dtype_equal(x.dtype, self.dtype) else x
        return x

    def backward(self, gys):
        return gys

def cast(x):
    return Cast()(x)


class DownCast(Function):
    def __init__(self, dtype='float16'):
        super().__init__()
        self.dtype = dtype

    def forward(self, *xs):
        if len(xs) > 1:
            raise ValueError("DownCast function only supports single input.")
        x = xs[0]
        # 降低精度
        x = x.astype(self.dtype) if compare_dtype_greater(x.dtype, self.dtype) else x
        return x

    def backward(self, gys):
        return gys

def downcast(x):
    return DownCast()(x)

class UpCast(Function):
    def __init__(self, dtype='float32'):
        super().__init__()
        self.dtype = dtype

    def forward(self, *xs):
        if len(xs) > 1:
            raise ValueError("UpCast function only supports single input.")
        x = xs[0]
        # 提高精度
        x = x.astype(self.dtype) if compare_dtype_greater(self.dtype, x.dtype) else x 
        return x

    def backward(self, gys):
        return gys

def upcast(x):
    return UpCast()(x)

def get_qmin_qmax(dtype):
    if type(dtype) == str and dtype.startswith('int'):
        num_bits = int(dtype[3:])
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
    elif type(dtype) == str and dtype.startswith('uint'):
        num_bits = int(dtype[4:])
        qmin = 0
        qmax = 2 ** num_bits - 1
    elif isinstance(dtype, np.dtype):
        qmin = np.iinfo(dtype).min
        qmax = np.iinfo(dtype).max
    elif has_cupy and isinstance(dtype, cp.dtype):
        qmin = cp.iinfo(dtype).min
        qmax = cp.iinfo(dtype).max
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")
    return qmin, qmax

class FakeQuantize(Function):
    def __init__(self, dtype='int8', gamma=0.9):
        super().__init__()
        self.dtype = dtype
        self.qmin, self.qmax = get_qmin_qmax(dtype)
        self.gamma = gamma

        self.scale = None
        self.zero_point = None
        self.alpha = None
        self.beta = None

    def forward(self, *xs):
        x = xs[0]
        xp = get_array_module(x)
        
        # 计算输入的最小值和最大值
        x_min = x.min()
        x_max = x.max()
        # 更新alpha和beta
        if self.alpha is None:
            self.alpha = x_min
        elif x_min < self.alpha:
            self.alpha = x_min
        else:
            self.alpha = self.gamma * self.alpha + (1 - self.gamma) * x_min

        if self.beta is None:
            self.beta = x_max
        elif x_max > self.beta:
            self.beta = x_max
        else:
            self.beta = self.gamma * self.beta + (1 - self.gamma) * x_max
        
        # 计算缩放因子
        self.scale = (self.beta - self.alpha) / (self.qmax - self.qmin) if self.beta > self.alpha else 1.0
        self.zero_point = xp.round(self.qmin - (self.alpha / self.scale)).clip(self.qmin, self.qmax) if self.scale > 0 else 0

        # 量化并反量化
        q_x = xp.round(x / self.scale + self.zero_point).clip(self.qmin, self.qmax) if self.scale > 0 else x
        fq_x = (q_x - self.zero_point) * self.scale if self.scale > 0 else x
        #print(f'xmin - alpha = {x_min - self.alpha}')
        #print(f'beta - max = {self.beta - x_max}')
        #print(f'average offset = {xp.average(fq_x - x)}')
        return fq_x

    def backward(self, gys):
        # Straight-Through Estimator: 梯度直接传递，不考虑量化误差
        return gys
    
def fake_quantize(x, dtype='int8', gamma=0.9):
    return FakeQuantize(dtype=dtype, gamma=gamma)(x)

class FakeDequantize(Function):
    def forward(self, *xs):
        x = xs[0]
        # 反量化：直接返回输入，保持与 FakeQuantize 的输出一致
        return x

    def backward(self, gys):
        # Straight-Through Estimator: 梯度直接传递，不考虑量化误差
        return gys
    
def fake_dequantize(x):
    return FakeDequantize()(x)

class QuantizeRegistry:
    can_quantize = [
        Conv2d,
        GroupedConv2d,
        Deconv2d,
        MatMul,
        Linear,
        FusedConvReLU
    ]

class Quantize(Function):
    def __init__(self, scale, zero_point, dtype='int8'):
        super().__init__()
        self.dtype = dtype
        self.qmin, self.qmax = get_qmin_qmax(dtype)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, *xs):
        x = xs[0]
        xp = get_array_module(x)

        # 量化
        q_x = xp.round(x / self.scale + self.zero_point).clip(self.qmin, self.qmax)
        return q_x.astype(self.dtype)

    def backward(self, gys):
        return gys

def quantize(x, scale, zero_point, dtype='int8'):
    return Quantize(scale, zero_point, dtype=dtype)(x)

class Dequantize(Function):
    def __init__(self, scale, zero_point, dtype='int8'):
        super().__init__()
        self.dtype = dtype
        self.qmin, self.qmax = get_qmin_qmax(dtype)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, *xs):
        x = xs[0]
        # 反量化
        deq_x = (x.astype('float32') - self.zero_point) * self.scale
        return deq_x

    def backward(self, gys):
        return gys

def dequantize(x, scale, zero_point, dtype='int8'):
    return Dequantize(scale, zero_point, dtype)(x)