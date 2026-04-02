from __future__ import annotations  # 统一把注解变成字符串,在类定义后才解析gradient类型
from typing import List
from ..utils import StateDict
import numpy as np
import weakref
import contextlib
from functools import total_ordering
from eneuro.base import functions as f
from eneuro.global_config import VISUAL_CONFIG
import cv2


class Config:
    enable_backprop = True
    train = True

    record_graph = False
    current_tracer = None   # 当前活动的 Tracer

    @classmethod
    @contextlib.contextmanager
    def using_config(cls, attr, value):
        old_value = getattr(Config, attr)
        setattr(Config, attr, value)
        try:
            yield
        finally:
            setattr(Config, attr, old_value)

    @classmethod
    def no_grad(cls):
        return cls.using_config('enable_backprop', False)

    @classmethod
    def test_mode(cls):
        return cls.using_config('train', False)


@total_ordering
class Tensor(StateDict):
    def __init__(self, data, requires_grad=False, name=None) -> None:
        self.data = data
        self.requires_grad = requires_grad
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

    @property
    def stride(self):
        return self.data.strides

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Tensor(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Tensor(' + p + ')'
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.data == other.data
        else:
            return self.data == other
    
    def __lt__(self, other):
        """只需要实现 < 和 ==，total_ordering会自动生成其他比较方法"""
        if isinstance(other, Tensor):
            return self.data < other.data
        else:
            return self.data < other

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None
    
    def argmax(self, axis=None):
        return self.data.argmax(axis=axis)
    
    def sum(self, axis=None, keepdims=False):
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

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
        '''
            testing
        '''
        if self.creator is not None:  # 创建者不为空
            add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with Config.using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    # if not isinstance(gx, Tensor):
                    #     gx = Tensor(gx)
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
            seen_set = set()

            def add_func(f):
                if f not in seen_set:
                    seen_set.add(f)
                    funcs.append(f)

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
        # 从本地functions模块导入reshape函数
        from .functions import reshape
        return reshape(self, shape)


    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        # 从本地functions模块导入transpose函数
        from .functions import transpose
        return transpose(self, axes)
    @property
    def T(self):
        return self.transpose()

  #数据加载器需要用的方法
    @staticmethod
    def stack(tensors: List['Tensor']) -> 'Tensor':
        """堆叠多个Tensor"""
        if not tensors:
            return Tensor(np.array([]))
        return Tensor(np.stack([t.data for t in tensors]))

    def using_config(self, param, create_graph):
        pass


def to_dict() -> dict:  # 序列化保存数据
    return super().to_dict()


def from_dict(d: dict) -> None:  # 序列化读取数据
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


class Function:
    def __init__(self):
        self.visualize=False

    def __call__(self, *inputs):
        inputs = [as_Tensor(x) for x in inputs]#判断or转化类型

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Tensor(as_array(y)) for y in ys]

        '''
            这里需要使用python原装的max
            否则会出现list没有max方法的问题
        '''
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]#弱引用，避免循环引用
        current_layer_name = self.__class__.__name__.lower()
        if VISUAL_CONFIG["ENABLE_ALL_LAYERS"] and self.visualize:
            self._print_output(outputs[0])

        if Config.record_graph and Config.current_tracer is not None:
            Config.current_tracer.record(self, inputs, outputs)

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

    def _print_output(self, output_tensor):
        # 和Layer类的_print_output逻辑完全一致（复制粘贴）
        data = output_tensor.data if hasattr(output_tensor, 'data') else output_tensor
        if data.ndim != 4:
            return

        N, C, H, W = data.shape
        sample_idx = 0
        scale = 20
        layer_name = self.__class__.__name__

        for channel_idx in range(C):
            img = data[sample_idx, channel_idx]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = cv2.resize(img.astype(np.uint8), (W * scale, H * scale))

            window_name = f"{layer_name} channel{channel_idx + 1} image"
            cv2.putText(img, window_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, img)
            print(f"[可视化] {window_name}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

'''
base operators
'''
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
def square(x):
    return Square()(x) # you can use square(x) to call the forward method of the Square class
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
def exp(x):
    return Exp()(x) # you can use exp(x) to call the forward method of the Exp clasa

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = f.sum_to(gx0, self.x0_shape)
            gx1 = f.sum_to(gx1, self.x1_shape)
        return gx0, gx1
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)
# Variable.__add__ = add
# Variable.__radd__ = add

class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0, gx1 = gy, gy
        x0, x1 = self.inputs # cause gy is a variable now, you can use self.inputs to get the data
        if self.x0_shape != self.x1_shape:
            gx0 = f.sum_to(gx0, self.x0_shape)
            gx1 = f.sum_to(gx1, self.x1_shape)
        return gx0 * x1, gx1 * x0
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)
# Variable.__mul__ = mul
# Variable.__rmul__ = mul

class Neg(Function):
    def forward(self, x):

        return -x
    
    def backward(self, gy):
        return -gy
def neg(x):
    return Neg()(x)
# Variable.__neg__ = neg

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = f.sum_to(gx0, self.x0_shape)
            gx1 = f.sum_to(gx1, self.x1_shape)
        return gx0, -gx1

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)
# Variable.__sub__ = sub
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # rsub is the same as sub but with the arguments reversed
# Variable.__rsub__ = rsub

class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0, gx1 = gy, gy
        x0, x1 = self.inputs
        if self.x0_shape != self.x1_shape:
            gx0 = f.sum_to(gx0, self.x0_shape)
            gx1 = f.sum_to(gx1, self.x1_shape)
        return gx0 / x1, -gx1 * x0 / x1 ** 2

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)
# Variable.__truediv__ = div
def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) # rdiv is the same as div but with the arguments reversed
# Variable.__rtruediv__ = rdiv

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
def pow(x, c):
    return Pow(c)(x)
# Variable.__pow__ = pow

def setup_tensor():
    # 从.functions模块导入所需的函数
    from .functions import get_item, matmul
    
    Tensor.__add__ = add
    Tensor.__radd__ = add
    Tensor.__mul__ = mul
    Tensor.__rmul__ = mul
    Tensor.__neg__ = neg
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv
    Tensor.__pow__ = pow

    Tensor.__getitem__ = get_item
    Tensor.matmul = matmul
    Tensor.dot = matmul
    # Tensor.max = core.functions.max
    # Tensor.min = core.functions.min
