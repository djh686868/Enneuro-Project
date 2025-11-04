from __future__ import annotations   # 统一把注解变成字符串,在类定义后才解析gradient类型

from ..utils import StateDict
import numpy as np

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
def as_array(x):
    """将输入转换为numpy数组"""
    if np.isscalar(x):
        return np.array(x)
    return x

def as_Tensor(x):
    """将输入转换为Tensor对象"""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)
        

