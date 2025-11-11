from __future__ import annotations   # 统一把注解变成字符串,在类定义后才解析gradient类型

from ..utils import StateDict
import numpy as np




class Tensor(StateDict):
    def __init__(self) -> None:
        self.data : np.ndarray
        self.shape : tuple
        self.grad : Tensor | None
        self.requires_grad : bool = False
        self.creator = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

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


        

