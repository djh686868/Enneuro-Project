import weakref
from .core import Tensor 
from .core import as_Tensor, as_array
from .core import Function
import numpy as np
from .core import Config

'''
other functions
'''

class Sin(Function):
    def forward(self, *xs):
        xs = xs[0]
        return np.sin(xs)

    def backward(self, gys):
        x = self.inputs[0].data
        gx = gys * np.cos(x)
        return gx
    
class Cos(Function):
    def forward(self,*xs):
        xs = xs[0]
        return np.cos(xs)
    def backward(self, gys):
        x = self.inputs[0].data
        gx = gys * -np.sin(x)
        return gx

class Exp(Function):
    def forward(self,*xs):
        xs = xs[0]
        return np.exp(xs)
    def backward(self, gys):
        x = self.inputs[0].data
        gx = gys * np.exp(x)
        return gx

class Tanh(Function):
    def forward(self,*xs):
        xs=xs[0]
        return np.tanh(xs)
    def backward(self, gys):
        y = self.outputs[0].data
        gx = gys * (1 - y**2)
        return gx
    
class Log(Function):
    def forward(self,*xs):
        xs=xs[0]
        return np.log(xs)
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
        gx = np.zeros(self.x_shape)
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
        self.visualize = visualize
        self.shape = shape
    def forward (self,*xs):
        xs=xs[0]
        self.x_shape = xs.shape
        return np.broadcast_to(xs,self.shape)
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
    def forward(self, *xs):
        x = xs[0]
        W = xs[1]
        y = x.dot(W)
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
    def forward(self,*xs):
        x = xs[0]
        w = xs[1]
        b = xs[2]
        y = x.dot(w)
        if b is None:
            return y
        else:
            return y + b 
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
        y = np.tanh(x * 0.5) * 0.5 + 0.5  #使用numpy函数而非自定义函数
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
        y = np.maximum(x, 0.0)
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
        self.axis = axis
        self.visualize = visualize

    def forward(self, *xs):
        x = xs[0]
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
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
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p

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

def im2col_array(img, kernel_size, stride, pad, to_matrix=True, dilation=(1,1)):

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    DH, DW = pair(dilation)
    OH = get_conv_outsize(H, KH, SH, PH, DH)
    OW = get_conv_outsize(W, KW, SW, PW, DW)

    # CPU-only implementation (use NumPy). Pad and extract patches.
    img = np.pad(img,
                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
    # 计算每个patch在输入图像中的位置，支持扩张卷积
    for j in range(KH):
        j_lim = j * DH + SH * OH
        for i in range(KW):
            i_lim = i * DW + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j*DH:j_lim:SH, i*DW:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # Ensure `col` has shape (N, C, KH, KW, OH, OW)
    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    
    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                   dtype=col.dtype)
    #适合小图像处理
    # for oh in range(OH):
    #     for ow in range(OW):
    #         # 考虑填充偏移
    #         j_start = oh * SH
    #         i_start = ow * SW
    #         img[:, :, j_start:j_start+KH, i_start:i_start+KW] += col[:, :, :, :, oh, ow]
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
    return img[:, :, PH:H + PH, PW:W + PW]



#正式函数#
#卷积函数和反卷积函数对称性强，大部分代码互为镜像
class Conv2d(Function):
    def __init__(self, stride=(1,1), pad=(0,0), dilation=1, visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.dilation = pair(dilation)
        self.visualize = visualize


    def forward(self, *xs):
        x = xs[0]
        W = xs[1]
        b = xs[2]
        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation)

        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = np.rollaxis(y, 3, 1)
        return y

    def backward(self, gys):
        x, W, b = self.inputs
        # ==== gx ====
        gx_data = conv2d_backward_input_array(
            gys.data, W.data,
            stride=self.stride,
            pad=self.pad,
            dilation=self.dilation,
            out_h=x.shape[2],
            out_w=x.shape[3],
        )
        gx = as_Tensor(gx_data)
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gys)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gys.sum(axis=(0, 2, 3))
        return gx, gW, gb


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

    gx_pad = np.zeros((N, C, out_h + 2 * PH + SH - 1, out_w + 2 * PW + SW - 1), dtype=gy.dtype)

    for kh in range(KH):
        h_start = kh * DH
        h_end = h_start + SH * OH
        for kw in range(KW):
            w_start = kw * DW
            w_end = w_start + SW * OW

            # (N, OC, OH, OW) x (OC, C) -> (N, OH, OW, C) -> (N, C, OH, OW)
            contrib = np.tensordot(gy, W[:, :, kh, kw], axes=(1, 0)).transpose(0, 3, 1, 2)
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
        x = xs[0]
        W = xs[1]
        b = xs[2]

        N, C, H, W_in = x.shape
        OC, C_per_group, KH, KW = W.shape

        assert C % self.groups == 0, "Input channels must be divisible by groups"
        assert OC % self.groups == 0, "Output channels must be divisible by groups"

        OC_per_group = OC // self.groups
        OH = get_conv_outsize(H, KH, self.stride[0], self.pad[0], self.dilation[0])
        OW = get_conv_outsize(W_in, KW, self.stride[1], self.pad[1], self.dilation[1])
        y = np.zeros((N, OC, OH, OW), dtype=x.dtype)

        for i in range(self.groups):
            x_group = x[:, i*C_per_group:(i+1)*C_per_group, :, :]
            W_group = W[i*OC_per_group:(i+1)*OC_per_group, :, :, :]

            col = im2col_array(
                x_group, (KH, KW), self.stride, self.pad,
                to_matrix=False, dilation=self.dilation
            )
            y_group = np.tensordot(col, W_group, ((1, 2, 3), (1, 2, 3)))
            y_group = np.rollaxis(y_group, 3, 1)
            y[:, i*OC_per_group:(i+1)*OC_per_group, :, :] = y_group

        if b is not None:
            y += b.reshape(1, -1, 1, 1)

        return y

    def backward(self, gys):
        x, W, b = self.inputs
        x_data = x.data
        W_data = W.data
        gy = gys.data

        N, C, H, W_in = x_data.shape
        OC, C_per_group, KH, KW = W_data.shape
        OC_per_group = OC // self.groups

        gx = np.zeros_like(x_data)
        gW = np.zeros_like(W_data)

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
            gW_group = np.tensordot(gy_group, col, ((0, 2, 3), (0, 4, 5)))
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
    def __init__(self, stride=(1,1), pad=(0,0), outsize=None,visualize=False):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize
        self.visualize = visualize


    def forward(self, *xs):
        x = xs[0]
        W = xs[1]
        b = xs[2]
        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = np.tensordot(Weight, x, (0, 1))
        gcol = np.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gys):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gys, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        gW = Conv2DGradW(self)(gys, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gys.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=(1,1), pad=(0,0), outsize=None,visualize=False):
    return Deconv2d(stride, pad, outsize,visualize)(x, W, b)


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
        x = xs[0]
        gy = xs[1]

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False, dilation=self.dilation)
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
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
        #与池化操作的维度相同 
        gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + np.arange(0, self.indexes.size * KH * KW, KH * KW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        #也可以用transpose
        gcol = np.swapaxes(gcol, 2, 4)
        gcol = np.swapaxes(gcol, 3, 5)

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
    def __init__(self):
        super().__init__()
    
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
    # 调用BatchNormFunction，获取返回值
    y = BatchNormFunction(eps, momentum, training, moving_mean, moving_var)(x, gamma, beta)
    # 只返回y值，moving_mean和moving_var在函数内部直接更新
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
        N, C, H, W = x.shape
        self.x_shape = x.shape
        self.x = x

        # 计算当前 batch 的均值和方差
        # 在通道维度上计算，保留 H,W 用于广播
        mean = x.mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
        var = x.var(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)

        # 更新 running 统计量（训练时）
        if Config.train:
            # 转为 (C,) 方便存储
            m = mean.reshape(C)
            v = var.reshape(C)
            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * m
            self.running_var.data  = self.momentum * self.running_var.data  + (1 - self.momentum) * v
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
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        # 缩放和偏移
        out = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
        self.x_hat = x_hat
        self.gamma = gamma
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
        std_inv = 1.0 / np.sqrt(var + eps)
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
        x, W, b = xs
        KH, KW = W.shape[2:]

        # 1. im2col + 卷积
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation)
        conv_out = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            conv_out += b
        conv_out = np.rollaxis(conv_out, 3, 1)

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
    def __init__(self, stride=(1,1), pad=(0,0), dilation=(1,1), running_mean=None, running_var=None, momentum=0.9, eps=1e-5, visualize=False):
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
        # 输入：x, W, b, gamma, beta
        x, W, b, gamma, beta = xs
        OC, _, KH, KW = W.shape

        # 用于融合算子时的初始化
        if not hasattr(self,'outsize'):
            self.outsize = OC

        # ---------- 1. 卷积 ----------
        # im2col
        col = im2col_array(x.data, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation)
        # 卷积输出 (N, OH, OW, OC)
        conv_out = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            conv_out += b
        conv_out = np.rollaxis(conv_out, 3, 1)

        # 保存卷积输出和输入，用于反向
        self.x = x
        self.W = W
        self.b = b
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
                self.running_mean = Tensor(np.zeros(OC, dtype=np.float32), requires_grad=False, name='running_mean')
            if self.running_var is None:
                self.running_var = Tensor(np.ones(OC, dtype=np.float32), requires_grad=False, name='running_mean')

            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * m
            self.running_var.data  = self.momentum * self.running_var.data  + (1 - self.momentum) * v
            self.mean = mean
            self.var = var
        else:
            # 测试模式：使用 running 统计量
            if self.running_mean is None:
                self.running_mean = Tensor(np.zeros(OC, dtype=np.float32), requires_grad=False, name='running_mean')
            if self.running_var is None:
                self.running_var = Tensor(np.ones(OC, dtype=np.float32), requires_grad=False, name='running_mean')

            mean = self.running_mean.data.reshape(1, OC, 1, 1)
            var = self.running_var.data.reshape(1, OC, 1, 1)
            self.mean = mean
            self.var = var

        # 归一化、缩放、平移
        std_inv = 1.0 / np.sqrt(var + self.eps)
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
        col_x = im2col_array(self.x.data, (KH, KW), self.stride, self.pad, to_matrix=False, dilation=self.dilation)
        gW = np.tensordot(g_conv_out, col_x, ((0,2,3), (0,4,5)))   # (OC, C, KH, KW)

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
