import numpy as np
import sys
import os

# 添加code目录到Python路径
sys.path.append(os.path.abspath('./code'))
from eneuro.base.core import Tensor

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
t3 = t.transpose()
print(f"Original shape: {t.shape}")
print(f"Original stride: {t.stride}")
print(f"Transposed shape: {t3.shape}")
print(f"Transposed stride: {t3.stride}")

# 测试reshape后的stride
print("\n测试reshape后的stride:")
t4 = t.reshape(1, 2, 3)
print(f"Original shape: {t.shape}")
print(f"Original stride: {t.stride}")
print(f"Reshaped shape: {t4.shape}")
print(f"Reshaped stride: {t4.stride}")
