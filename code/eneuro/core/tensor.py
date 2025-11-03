import numpy as np
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

class Tensor:
    def __init__(self, data):
        pass
    pass