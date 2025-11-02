import weakref
from .tensor import Tensor 
from .tensor import as_Tensor, as_array
import numpy as np




class Function:
    def __call__(self, *inputs):
        inputs = [as_Tensor(x) for x in inputs]#判断or转化类型

        xs = [x.data for x in inputs]()
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Tensor(as_array(y)) for y in ys]

       
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]#弱引用，避免循环引用

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
class Cos(Function):
    def forward(self,x):
         return np.cos(x)
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * -np.sin(x)
        return gx

class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.exp(x)
        return gx

class Tanh(Function):
    def forward(self,x):
        return np.tanh(x)
    def backward(self, gy):
        y = self.outputs[0].data
        gx = gy * (1 - y**2)
        return gx
    
class Log(Function):
    def forward(self,x):
        return np.log(x)
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy / x
        return gx


class Reshape(Function):
    def __init__(self,shape):
        self.shape = shape
    def forward(self,x):
        self.x_shape = x.shape
        return x.reshape(self.shape)
    def backward(self,gy):
        return  reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return x
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self,axes = None):
        self.axes = axes
    def forward(self,x):
        return x.transpose(self.axes)
    def backward(self,gy):
        if self.axes is None:
            return gy.transpose()
        axes_len = len(self.axes)
        #计算逆转置的轴
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
#转置操作的便捷函数,Tensor
#的transpose方法会调用它    
def transpose(x, axes = None):
    return Transpose(axes)(x)
#切片操作
class GetItem(Function):
    def __init__ (self,slices):
        self.slices = slices
    def forward (self,x):
        return x[self.slices]
    def backward (self,gy):
        x=self.inputs[0]
        gx=GetItemGrad(self.slices,x.shape)(gy)
        return gx   
    
#切片操作的反向传播
class GetItemGrad(Function):
    def __init__ (self,slices,x_shape):
        self.slices = slices
        self.x_shape = x_shape
    def forward (self,gy):
        gx = np.zeros(self.x_shape)
        gx[self.slices] = gy
        return gx
    #切片操作的反向传播
    def backward (self,ggx):
        return GetItem(self.slices)(ggx)
    

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
    def __init__ (self,axis,keepdims):
        self.axis = axis
        self.keepdims = keepdims   
    def forward (self,x):
        self.x_shape = x.shape
        return x.sum(x,self.axis,self.keepdims)
    def backward (self,gy):
        ndim = len(self.x_shape)
        tupled_axis = self.axis
        if self.axis is None:
            tupled_axis = None
        elif not isinstance(self.axis, tuple):
            tupled_axis = (self.axis,)

        if not (ndim == 0 or tupled_axis is None or self.keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
            shape = list(gy.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gy.shape
        gy = gy.reshape(shape)  # reshape
        return gy

def sum (x,axis = None,keepdims = False):
    return Sum(axis,keepdims)(x)


#求和到目标形状
class SumTo(Function):
    def __init__ (self,shape):
        self.shape = shape
    def forward (self,x):
        self.x_shape = x.shape
        ndim = len(self.shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
             y = y.squeeze(lead_axis)
        return y

    def backward (self,gy):
        return reshape(gy,self.x_shape)
 
def sum_to(x, shape):
    if x.shape == shape:
        return as_Tensor(x)
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__ (self,shape):
        self.shape = shape
    def forward (self,x):
        self.x_shape = x.shape
        return np.broadcast_to(x,self.shape)
    def backward (self,gy):
        return sum_to(gy,self.x_shape)
  

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_Tensor(x)
    return BroadcastTo(shape)(x)


def average(x, axis=None, keepdims=False):
    x = as_Tensor(x)
    y = sum(x, axis, keepdims)
    return y * (y.data.size / x.data.size)



class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        #调用的是自定义的matmul函数
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)

##线性变换函数###

class Linear(Function):
    def forward(self,x,w,b):
        y = x.dot(w)
        if b is None:
            return y
        else:
            return y + b 
    def backward (self,gy):
        x,w,b = self.inputs
        gx = matmul(gy,w.T)
        gw = matmul(x.T,gy)
        if b is None:
            gb = None
        else:
            gb = gy.sum(axis=0)

#封装成线性变换的便捷函数
def linear(x, W, b=None):
    return Linear()(x, W, b)

class Sigmoid(Function):
    def forward(self, x):
        y = np.tanh(x * 0.5) * 0.5 + 0.5  #使用numpy函数而非自定义函数
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

# Sigmoid函数的便捷函数
def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)








##卷积部分函数###

#准备函数1#
#反卷积输出尺寸，即前向传播输入的未填充的尺寸
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p

#卷积输出尺寸，即前向传播输出的尺寸
def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

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
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad, self.to_matrix)
        return gx

#依旧是im2col的便捷函数
def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
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
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def im2col_array(img, kernel_size, stride, pad, to_matrix=True):

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # CPU-only implementation (use NumPy). Pad and extract patches.
    img = np.pad(img,
                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
    # 计算每个patch在输入图像中的位置，第一种按卷积核遍历，第二个按输出位置遍历
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    #适合小图像处理，大图像小卷积核用上面那种，但是不如下面这个理解性强
    # for oh in range(OH):           # 遍历输出高度
    #     for ow in range(OW):       # 遍历输出宽度
    #         # 计算当前patch的起始位置
    #         j_start = oh * SH
    #         i_start = ow * SW
    #         # 提取整个patch
    #         col[:, :, :, :, oh, ow] = img[:, :, j_start:j_start+KH, i_start:i_start+KW]

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
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = np.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # ==== gx ====
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
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

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        gW = Conv2DGradW(self)(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
       

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
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
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        

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

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW*KH)
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N*C*OH*OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)

