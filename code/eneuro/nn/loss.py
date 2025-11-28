import numpy as np
from .tensor import Tensor
from .functions import Function
from abc import abstractmethod

class Loss(Function):
    """损失函数基类"""
    
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, pred, target):
        pass
    
    @abstractmethod
    def backward(self, gys):
        pass
    
    def __call__(self, pred, target):
        # 确保输入是Tensor
        pred = Tensor(pred) if not isinstance(pred, Tensor) else pred
        target = Tensor(target) if not isinstance(target, Tensor) else target
        
       
        return super().__call__(pred, target)

class MSELoss(Loss):
    """均方误差损失"""

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        diff = pred - target

        if self.reduction == 'mean':
            loss = np.mean(diff ** 2)
        elif self.reduction == 'sum':
            loss = np.sum(diff ** 2)
        else:
            loss = diff ** 2

        return loss

    def backward(self, gys):
        batch_size = self.pred.shape[0] if len(self.pred.shape) > 0 else 1
        diff = self.pred - self.target

        if self.reduction == 'mean':
            grad_input = 2 * diff / batch_size
        elif self.reduction == 'sum':
            grad_input = 2 * diff
        else:
            grad_input = 2 * diff

        # 乘以上游梯度
        grad_input = grad_input * gys
        return grad_input

class CrossEntropyLoss(Loss):
    """交叉熵损失"""

    def forward(self, pred, target):
        self.pred = pred
        self.target = target

        # 数值稳定的softmax
        exp_vals = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        self.softmax = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        # 计算交叉熵损失
        if len(target.shape) == 1:
            # 类别索引
            batch_size = pred.shape[0]
            loss = -np.log(self.softmax[np.arange(batch_size), target.astype(int)] + 1e-8)
        else:
            # one-hot编码
            loss = -np.sum(target * np.log(self.softmax + 1e-8), axis=1)

        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)

        return loss

    def backward(self, gys):
        batch_size = self.pred.shape[0]

        if len(self.target.shape) == 1:
            # 类别索引
            grad_input = self.softmax.copy()
            grad_input[np.arange(batch_size), self.target.astype(int)] -= 1
        else:
            # one-hot编码
            grad_input = self.softmax - self.target

        if self.reduction == 'mean':
            grad_input /= batch_size

        # 乘以上游梯度
        grad_input = grad_input * gys
        return grad_input

class BCELoss(Loss):
    """二元交叉熵损失"""
    
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        
        # 应用sigmoid将预测值映射到(0,1)范围
        self.pred_sigmoid = 1 / (1 + np.exp(-pred))
        
        # 数值稳定性：防止log(0)
        epsilon = 1e-8
        pred_clipped = np.clip(self.pred_sigmoid, epsilon, 1 - epsilon)
        
        # 计算二元交叉熵
        bce = - (target * np.log(pred_clipped) + 
                (1 - target) * np.log(1 - pred_clipped))
        
        if self.reduction == 'mean':
            loss = np.mean(bce)
        elif self.reduction == 'sum':
            loss = np.sum(bce)
        else:
            loss = bce
            
        return loss
    
    def backward(self, gys):
        batch_size = self.pred.shape[0] if len(self.pred.shape) > 0 else 1
        
        # BCE梯度: dL/dpred = (pred_sigmoid - target)
        grad_input = self.pred_sigmoid - self.target
        
        if self.reduction == 'mean':
            grad_input /= batch_size
        
        # 乘以上游梯度
        grad_input = grad_input * gys
        return grad_input

# 便捷函数
def mse_loss(pred, target, reduction='mean'):
    return MSELoss(reduction)(pred, target)

def cross_entropy_loss(pred, target, reduction='mean'):
    return CrossEntropyLoss(reduction)(pred, target)

def bce_loss(pred, target, reduction='mean'):
    return BCELoss(reduction)(pred, target)
