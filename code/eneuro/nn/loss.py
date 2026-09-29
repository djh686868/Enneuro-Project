import numpy as np
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
from ..base.functions import Function, get_array_module
from ..base import Tensor, as_Tensor

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

class MSELoss(Function):
    def forward(self, x, t):
        """
        均方误差损失函数的前向传播。

        GPU适配说明:
            - 优先从输入x获取数组模块类型
            - 当x是cupy数组时，需要先将t转换为cupy数组
        """
        xp = get_array_module(x)
        if t.ndim == 1 and x.ndim == 2:
            t = t.reshape(len(t), 1)
        # 关键修复：使用to_xp函数正确转换数组类型
        t_data = to_xp(t, xp)
        # Keep the target in the prediction dtype.  In particular, CuPy's
        # scalar division can otherwise promote a float32 regression graph to
        # float64 before the first convolution backward pass.
        if hasattr(t_data, 'dtype') and t_data.dtype != x.dtype:
            t_data = xp.asarray(t_data, dtype=x.dtype)
        self.x = x
        self.t = t_data
        self.diff = x - t_data
        batch_scale = xp.asarray(len(x), dtype=x.dtype)
        y = xp.sum(self.diff ** 2) / batch_scale
        return y

    def backward(self, dout=1):
        xp = get_array_module(self.diff)
        dtype = self.diff.dtype
        two = xp.asarray(2, dtype=dtype)
        batch_scale = xp.asarray(len(self.diff), dtype=dtype)
        dx = two * self.diff / batch_scale
        dout = to_xp(dout, xp)
        if hasattr(dout, 'dtype') and dout.dtype != dtype:
            dout = xp.asarray(dout, dtype=dtype)
        return as_Tensor(dx) * dout

class SoftmaxWithLoss(Function):
    def forward(self, x, t):
        """
        Softmax交叉熵损失函数的前向传播。

        GPU适配说明:
            - 优先从输入x获取数组模块类型
            - 当x是cupy数组时，需要先将t转换为cupy数组
        """
        xp = get_array_module(x)
        maxX = x.max(axis=1, keepdims=True)
        expX = xp.exp(x - maxX)
        sumExp = xp.sum(expX, axis=1, keepdims=True)
        self.y = expX / sumExp
        # 交叉熵损失
        if t.ndim == 1:
            t = t.reshape(len(t), 1)
        # 关键修复：使用to_xp函数正确转换数组类型
        t_data = to_xp(t, xp)
        batchSize = x.shape[0]
        self.t = t_data
        loss = -xp.sum(xp.log(self.y[xp.arange(batchSize), t_data.flatten()] + 1e-7)) / batchSize
        return loss
    def backward(self, dout=1):
        xp = get_array_module(self.y)
        batchSize = self.t.shape[0]
        
        if self.t.size == self.y.size:  # one-hot编码
            dx = (self.y - self.t) / batchSize
        else:  # 标签索引
            dx = self.y.copy()
            dx[xp.arange(batchSize), self.t.flatten()] -= 1
            dx = dx / batchSize
            
        dout = to_xp(dout, xp)
        return as_Tensor(dx) * dout

class SigmoidWithLoss(Function):
    def forward(self, x, t):
        """
        Sigmoid交叉熵损失函数的前向传播（用于二元分类）。

        GPU适配说明:
            - 优先从输入x获取数组模块类型
            - 当x是cupy数组时，需要先将t转换为cupy数组
        """
        xp = get_array_module(x)
        # sigmoid函数
        self.y = 1 / (1 + xp.exp(-x))
        # 关键修复：使用to_xp函数正确转换数组类型
        t_data = to_xp(t, xp)
        self.t = t_data

        # 二元交叉熵损失
        loss = -xp.sum(t_data * xp.log(self.y + 1e-7) + (1 - t_data) * xp.log(1 - self.y + 1e-7)) / len(x)
        return loss

    def backward(self, dout=1):
        batchSize = self.t.shape[0]
        dx = (self.y - self.t) / batchSize

        xp = get_array_module(self.y)
        dout = to_xp(dout, xp)
        return as_Tensor(dx) * dout

class CrossEntropyLoss(Function):
    def forward(self, x, t):
        """
        交叉熵损失函数的前向传播。

        GPU适配说明:
            - 优先从输入x获取数组模块类型
            - 当x是cupy数组时，需要先将t转换为cupy数组
        """
        xp = get_array_module(x)
        maxX = x.max(axis=1, keepdims=True)
        logZ = xp.log(xp.sum(xp.exp(x - maxX), axis=1, keepdims=True))
        logSoftmax = x - maxX - logZ

        # 关键修复：使用to_xp函数正确转换数组类型
        t_data = to_xp(t, xp)

        if t_data.ndim == 1:
            # 标签索引形式
            batchSize = x.shape[0]
            loss = -xp.sum(logSoftmax[xp.arange(batchSize), t_data]) / batchSize
        else:
            # one-hot编码形式
            loss = -xp.sum(t_data * logSoftmax) / len(x)

        self.logSoftmax = logSoftmax
        self.t = t_data
        return loss

    def backward(self, dout=1):
        xp = get_array_module(self.logSoftmax)
        batchSize = self.t.shape[0]
        
        if self.t.ndim == 1:
            # 从logSoftmax推导梯度
            dx = xp.exp(self.logSoftmax)
            dx[xp.arange(batchSize), self.t] -= 1
            dx = dx / batchSize
        else:
            dx = (xp.exp(self.logSoftmax) - self.t) / batchSize
        
        dout = to_xp(dout, xp)
        #tmp = as_Tensor(dx) * dout
        return as_Tensor(dx) * dout

# 便捷函数
def meanSquaredError(x, t):
    return MSELoss()(x, t)

def softmaxCrossEntropy(x, t):
    return SoftmaxWithLoss()(x, t)

def sigmoidCrossEntropy(x, t):
    return SigmoidWithLoss()(x, t)

def crossEntropyError(x, t):
    return CrossEntropyLoss()(x, t)
