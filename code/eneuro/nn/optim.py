from ..core import Tensor,Parameter
from ..utils import StateDict
import numpy as np

class Optimizer(StateDict):
    params: list[Parameter]
    lr: float

    _state: dict[Parameter, dict] # 各参数对应的缓存（如 momentum、adam 的 m/v）

    def __init__(self, params: list[Parameter], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        """执行单步参数更新，子类必须实现"""
        raise NotImplementedError
    
    def zero_grad(self):
        """清空梯度（委托给模型）"""
        for param in self.params:
            if not param.grad:
                param.grad = Tensor()

            param.grad.data = np.zeros_like(param.data)

    # to_dict 把 Parameter 映射到 id(str)；from_dict 按 id 还原引用
    def to_dict(self) -> dict:
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None:
        return super().from_dict(d)
    
class SGD(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 0.01):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            if not param.grad:
                continue
            param.data -= self.lr * np.array(param.grad)
    
    def to_dict(self) -> dict:
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None:
        return super().from_dict(d)
    
class MomentumSGD(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(params, lr)
        self.momentum: float = momentum
        self.velocities: dict[Parameter, np.ndarray] = {}

    def step(self):
        for param in self.params:
            if not param.grad:
                continue
            
            if param not in self.velocities:
                self.velocities[param] = np.zeros_like(param.data)

            # v = v*m - lr*grad
            self.velocities[param] = self.velocities[param] * self.momentum - self.lr * param.grad.data
            # w = w + v
            param.data += self.velocities[param]
    
    def to_dict(self) -> dict:
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None:
        return super().from_dict(d)
    
class Adam(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.9, eps = 1e-10):
        super().__init__(params, lr)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps

        self.s: dict[Parameter, np.ndarray] = {}
        self.v: dict[Parameter, np.ndarray] = {}

    def step(self):
        for param in self.params:
            if not param.grad:
                continue

            if param not in self.s:
                self.s[param] = np.zeros_like(param.data)

            if param not in self.v:
                self.v[param] = np.zeros_like(param.data)

            # v = v*beta1 + (1-beta1)*grad
            self.v[param] = self.v[param] * self.beta1 + (1-self.beta1) * param.grad.data
            # s = s*beta2 + (1-beta2)*grad*grad
            self.s[param] = self.s[param] * self.beta2 + (1-self.beta2) * np.power(param.grad.data, 2)
            # w = w - lr*v / sqrt(s + eps)
            param.data -= self.lr * self.v[param] / np.sqrt(self.s[param] + self.eps)
    
    def to_dict(self) -> dict:
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None:
        return super().from_dict(d)