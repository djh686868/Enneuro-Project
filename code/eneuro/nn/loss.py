import numpy as np
from ..base.functions import Function
from ..base import Tensor, as_Tensor

class MSELoss(Function):
    def forward(self, x, t):
        if t.ndim == 1 and x.ndim == 2:
            t = t.reshape(len(t), 1)
        self.x = x
        self.t = t
        self.diff = x - t
        y = np.sum(self.diff ** 2) / len(x)
        return y

    def backward(self, dout=1):
        dx = 2 * self.diff / len(self.diff)
        return as_Tensor(dx) * dout

class SoftmaxWithLoss(Function):
    def forward(self, x, t):
        # 稳定化softmax计算
        maxX = x.max(axis=1, keepdims=True)
        expX = np.exp(x - maxX)
        sumExp = np.sum(expX, axis=1, keepdims=True)
        self.y = expX / sumExp
        # 交叉熵损失
        if t.ndim == 1:
            t = t.reshape(len(t), 1)
        batchSize = x.shape[0]
        self.t = t
        loss = -np.sum(np.log(self.y[np.arange(batchSize), t.flatten()] + 1e-7)) / batchSize
        return loss
    def backward(self, dout=1):
        batchSize = self.t.shape[0]
        
        if self.t.size == self.y.size:  # one-hot编码
            dx = (self.y - self.t) / batchSize
        else:  # 标签索引
            dx = self.y.copy()
            dx[np.arange(batchSize), self.t.flatten()] -= 1
            dx = dx / batchSize
            
        return as_Tensor(dx) * dout

class SigmoidWithLoss(Function):
    def forward(self, x, t):
        # sigmoid函数
        self.y = 1 / (1 + np.exp(-x))
        self.t = t

        # 二元交叉熵损失
        loss = -np.sum(t * np.log(self.y + 1e-7) + (1 - t) * np.log(1 - self.y + 1e-7)) / len(x)
        return loss

    def backward(self, dout=1):
        batchSize = self.t.shape[0]
        dx = (self.y - self.t) / batchSize
        return as_Tensor(dx) * dout

class CrossEntropyLoss(Function):
    def forward(self, x, t):
        # 稳定化log softmax
        maxX = x.max(axis=1, keepdims=True)
        logZ = np.log(np.sum(np.exp(x - maxX), axis=1, keepdims=True))
        logSoftmax = x - maxX - logZ
        
        if t.ndim == 1:
            # 标签索引形式
            batchSize = x.shape[0]
            loss = -np.sum(logSoftmax[np.arange(batchSize), t]) / batchSize
        else:
            # one-hot编码形式
            loss = -np.sum(t * logSoftmax) / len(x)
            
        self.logSoftmax = logSoftmax
        self.t = t
        return loss

    def backward(self, dout=1):
        batchSize = self.t.shape[0]
        
        if self.t.ndim == 1:
            # 从logSoftmax推导梯度
            dx = np.exp(self.logSoftmax)
            dx[np.arange(batchSize), self.t] -= 1
            dx = dx / batchSize
        else:
            dx = (np.exp(self.logSoftmax) - self.t) / batchSize
        
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