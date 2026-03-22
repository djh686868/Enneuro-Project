import numpy as np
import sys
import os

# 直接导入core.py文件
sys.path.append(os.path.abspath('./code/eneuro/base'))

# 手动导入所需的模块
import numpy as np
import weakref
import contextlib
from functools import total_ordering

# 简化版的Config类
class Config:
    enable_backprop = True
    train = True

# 简化版的Tensor类，只包含我们需要测试的部分
class Tensor:
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

# 测试基本的stride属性
print("测试基本的stride属性:")
t = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
print(f"Tensor shape: {t.shape}")
print(f"Tensor stride: {t.stride}")
print(f"Numpy array stride: {t.data.strides}")

# 测试不同形状的tensor
print("\n测试不同形状的tensor:")
t2 = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
print(f"Tensor shape: {t2.shape}")
print(f"Tensor stride: {t2.stride}")

# 测试转置后的stride
print("\n测试转置后的stride:")
t3 = Tensor(t.data.transpose())
print(f"Original shape: {t.shape}")
print(f"Original stride: {t.stride}")
print(f"Transposed shape: {t3.shape}")
print(f"Transposed stride: {t3.stride}")

# 测试reshape后的stride
print("\n测试reshape后的stride:")
t4 = Tensor(t.data.reshape(1, 2, 3))
print(f"Original shape: {t.shape}")
print(f"Original stride: {t.stride}")
print(f"Reshaped shape: {t4.shape}")
print(f"Reshaped stride: {t4.stride}")
