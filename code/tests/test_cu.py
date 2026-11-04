import ctypes
import numpy as np
import os

# 加载库
lib_path = './code/eneuro/cuda/so/libvector_add.so' if os.name != 'nt' else './code/eneuro/cuda/dll/vector_add.dll'
lib = ctypes.cdll.LoadLibrary(lib_path)

# 设置函数签名
lib.launch_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
lib.launch_add.restype = None

# 准备数据
n = 1000000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

# 调用
lib.launch_add(
    a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    n
)

# 验证
print(c)
print(a+b)
print("Max difference:", np.max(np.abs(c - (a + b))))