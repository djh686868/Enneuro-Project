from ..core import Tensor
import numpy as np
from abc import ABC,abstractmethod

class Loss(ABC):

    @abstractmethod
    def __call__(self, pred, target):
        pass

class MSELoss(Loss):
    """均方误差损失"""

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred, target):
        self.pred, self.target = pred, target
        diff = pred.data - target.data

        if self.reduction == 'mean':
            loss = np.mean(diff ** 2)
        elif self.reduction == 'sum':
            loss = np.sum(diff ** 2)
        else:
            loss = diff ** 2

        loss_tensor = Tensor(loss, requires_grad=pred.requires_grad)
        if pred.requires_grad:
            loss_tensor.set_creator = self  

        return loss_tensor

    def backward(self, grad):
        batch_size = self.pred.shape[0] if len(self.pred.shape) > 0 else 1
        diff = self.pred.data - self.target.data

        if self.reduction == 'mean':
            grad_input = 2 * diff / batch_size
        elif self.reduction == 'sum':
            grad_input = 2 * diff
        else:
            grad_input = 2 * diff


        grad_input = grad_input * grad
        
        if self.pred.requires_grad:
            # 累加梯度
            if self.pred.grad is None:
                self.pred.grad = grad_input
            else:
                self.pred.grad += grad_input

class CrossEntropyLoss(Loss):
    """交叉熵损失"""

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred, target):
        self.pred, self.target = pred, target

        # 数值稳定的softmax
        exp_vals = np.exp(pred.data - np.max(pred.data, axis=1, keepdims=True))
        self.softmax = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        # 计算交叉熵损失
        if len(target.shape) == 1:
            # 类别索引
            batch_size = pred.shape[0]
            loss = -np.log(self.softmax[np.arange(batch_size), target.data.astype(int)] + 1e-8)
        else:
            # one-hot编码
            loss = -np.sum(target.data * np.log(self.softmax + 1e-8), axis=1)

        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)

        loss_tensor = Tensor(loss, requires_grad=pred.requires_grad)
        if pred.requires_grad:
            loss_tensor.set_creator = self  
        return loss_tensor

    def backward(self, grad):
        batch_size = self.pred.shape[0]

        if len(self.target.shape) == 1:
            # 类别索引
            grad_input = self.softmax.copy()
            grad_input[np.arange(batch_size), self.target.data.astype(int)] -= 1
        else:
            # one-hot编码
            grad_input = self.softmax - self.target.data

        if self.reduction == 'mean':
            grad_input /= batch_size

        grad_input = grad_input * grad
        
        if self.pred.requires_grad:
            # 累加梯度
            if self.pred.grad is None:
                self.pred.grad = grad_input
            else:
                self.pred.grad += grad_input

class BCELoss(Loss):
    """二元交叉熵损失"""
    
    def __init__(self, reduction='mean'):
        self.reduction = reduction
    
    def __call__(self, pred, target):
        self.pred, self.target = pred, target
        
        # 应用sigmoid将预测值映射到(0,1)范围
        self.pred_sigmoid = 1 / (1 + np.exp(-pred.data))
        
        # 数值稳定性：防止log(0)
        epsilon = 1e-8
        pred_clipped = np.clip(self.pred_sigmoid, epsilon, 1 - epsilon)
        
        # 计算二元交叉熵
        bce = - (target.data * np.log(pred_clipped) + 
                (1 - target.data) * np.log(1 - pred_clipped))
        
        if self.reduction == 'mean':
            loss = np.mean(bce)
        elif self.reduction == 'sum':
            loss = np.sum(bce)
        else:
            loss = bce
            
        loss_tensor = Tensor(loss, requires_grad=pred.requires_grad)
        if pred.requires_grad:
            loss_tensor.set_creator = self 
        return loss_tensor
    
    def backward(self, grad):
        batch_size = self.pred.shape[0] if len(self.pred.shape) > 0 else 1
        
        # BCE梯度: dL/dpred = (pred_sigmoid - target)
        grad_input = self.pred_sigmoid - self.target.data
        
        if self.reduction == 'mean':
            grad_input /= batch_size
        

        grad_input = grad_input * grad
            
        if self.pred.requires_grad:
         if self.pred.grad is None:
                self.pred.grad = grad_input
        else:
                self.pred.grad += grad_input
