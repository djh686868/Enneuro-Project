from __future__ import annotations  # 统一把注解变成字符串,在类定义后才解析gradient类型
from ..utils import StateDict
import numpy as np
import weakref
import contextlib
import code


class Config:
    enable_backprop = True
    train = True

    @contextlib.contextmanager
    def using_config(name, value):
        old_value = getattr(Config, name)
        setattr(Config, name, value)
        try:
            yield
        finally:
            setattr(Config, name, old_value)

    def no_grad(self):
        return self.using_config('enable_backprop', False)

    def test_mode(self):
        return self.using_config('train', False)


class Tensor(StateDict):
    def __init__(self, data, name=None) -> None:
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

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

            with self.using_config('enable_backprop', create_graph):
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

        def unchain_backward(self):
            if self.creator is not None:
                funcs = [self.creator]
                while funcs:
                    f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        add_func(x.creator)
                        funcs.append(x.creator)
                        x.unchain()


def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return code.functions.reshape(self, shape)


def transpose(self, *axes):
    if len(axes) == 0:
        axes = None
    elif len(axes) == 1:
        if isinstance(axes[0], (tuple, list)) or axes[0] is None:
            axes = axes[0]
    return code.functions.transpose(self, axes)


def to_dict(self) -> dict:  # 序列化保存数据
    return super().to_dict()


def from_dict(self, d: dict) -> None:  # 序列化读取数据
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


  #数据加载器需要用的方法
    @staticmethod
    def stack(tensors: List['Tensor']) -> 'Tensor':
        """堆叠多个Tensor"""
        if not tensors:
            return Tensor(np.array([]))
        return Tensor(np.stack([t.data for t in tensors]))
        

