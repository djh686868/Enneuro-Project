from __future__ import annotations   # 统一把注解变成字符串,在类定义后才解析gradient类型

from ..utils import StateDict
import numpy as np
from typing import List

class Tensor(StateDict):
    def __init__(self) -> None:
        self.data : np.ndarray
        self.shape : tuple
        self.grad : Tensor | None
        self.requires_grad : bool = False
    
    def backward(self, gradient : Tensor | None = None) -> None:
        pass

    def to_dict(self) -> dict: #序列化保存数据
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None: #序列化读取数据
        return super().from_dict(d)

    #数据加载器需要用的方法
    @staticmethod
    def stack(tensors: List['Tensor']) -> 'Tensor':
        """堆叠多个Tensor"""
        if not tensors:
            return Tensor(np.array([]))
        return Tensor(np.stack([t.data for t in tensors]))
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype})"