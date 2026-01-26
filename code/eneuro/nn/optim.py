from ..base import Tensor,Parameter
from ..utils import StateDict
import numpy as np
from typing import Any

LR_KEY = 'lr'
M_KEY = 'm'
S_KEY = 's'
T_KEY = 't'
V_KEY = 'v'
BETA1_KEY = 'beta1'
BETA2_KEY = 'beta2'
EPS_KEY = 'eps'
L2_LAMBDA_KEY = 'l2_lambda'
L1_LAMBDA_KEY = 'l1_lambda'

class Optimizer(StateDict):
    params: list[Parameter]
    _state: dict[str, Any] = {}

    def __init__(self, params: list[Parameter], lr: float = 0.01,
                 l2_lambda: float = 0.0, l1_lambda: float = 0.0):
        self.params = list(params)
        self._state.update(
            lr=lr, 
            l2_lambda=l2_lambda, 
            l1_lambda=l1_lambda
        )
    
    def step(self):
        """执行单步参数更新，子类必须实现"""
        raise NotImplementedError
    
    def zero_grad(self):
        """清空梯度"""
        for param in self.params:
            param.grad = None

    def to_dict(self) -> dict:
        state = {}
        for k, v in self._state.items():
            if isinstance(v, dict):
                if k not in state:
                    state[k] = {}
                for idx, arr in v.items():
                    # 'key': {'idx': ndarray} -> 'key': {'idx': list}
                    if isinstance(arr, np.ndarray):
                        state[k][idx] = arr.tolist()
                    else:
                        state[k][idx] = arr
            else:
                state[k] = v
        return state

    def from_dict(self, d: dict) -> None:
        new_state = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in new_state:
                    new_state[k] = {}
                for idx, arr in v.items():
                    # 'key': {'idx': list} -> 'key': {'idx': ndarray}
                    if isinstance(arr, list):
                        new_state[k][idx] = np.array(arr)
                    else:
                        new_state[k][idx] = arr
            else:
                new_state[k] = v
        self._state = new_state

    def _apply_regularization(self, param: Parameter, grad_data: np.ndarray) -> np.ndarray:
        """应用正则化到梯度"""
        l2_lambda = self._state.get(L2_LAMBDA_KEY, 0.0)
        l1_lambda = self._state.get(L1_LAMBDA_KEY, 0.0)
        
        # L2正则化 (权重衰减)
        # Loss = Loss + l2_lambda * ||w||^2
        # dLoss/dw = dLoss/dw + 2 * l2_lambda * w
        if l2_lambda > 0:
            grad_data += 2 * l2_lambda * param.data
        
        # L1正则化
        # Loss = Loss + l1_lambda * ||w||_1
        # dLoss/dw = dLoss/dw + l1_lambda * sign(w)
        if l1_lambda > 0:
            # 使用次梯度处理0值
            sign = np.sign(param.data)
            sign[param.data == 0] = 0  # 在0处，次梯度可以是[-1, 1]之间的任意值，通常取0
            grad_data += l1_lambda * sign
        
        return grad_data
    
class SGD(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 0.01,
                 l2_lambda: float = 0.0, l1_lambda: float = 0.0):
        super().__init__(params, lr, l2_lambda, l1_lambda)

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            # 应用正则化
            grad = self._apply_regularization(param, param.grad.data)
            param.grad.data = grad

            param.data -= self._state[LR_KEY] * param.grad.data
            
            '''
            print(type(param.grad.data))
            print(param.grad.data)
            t = self._state[LR_KEY] * param.grad.data
            t = param.data - t
            param.data = t
            '''
    
class MomentumSGD(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 0.01,
                 l2_lambda: float = 0.0, l1_lambda: float = 0.0, momentum: float = 0.9):
        super().__init__(params, lr, l2_lambda, l1_lambda)
        self._state.update(m=momentum)
        
        if V_KEY not in self._state:
            self._state[V_KEY] = {}

    def step(self):
        _lr, _m = self._state[LR_KEY], self._state[M_KEY]
        _vdict: dict[str,np.ndarray] = self._state[V_KEY]

        for idx in range(len(self.params)):
            param = self.params[idx]
            if param.grad is None:
                continue

            # 应用正则化
            grad = self._apply_regularization(param, param.grad.data)
            param.grad.data = grad

            _v = _vdict.get(str(idx), np.zeros_like(param.data))

            # v = v*m - lr*grad
            _v = _v * _m - _lr * param.grad.data
            # w = w + v
            param.data += _v

            self._state[V_KEY][str(idx)] = _v
    
class Adam(Optimizer):
    def __init__(self, params: list[Parameter], lr: float = 0.01,
                 l2_lambda: float = 0.0, l1_lambda: float = 0.0, beta1: float = 0.9, beta2: float = 0.999, eps = 1e-10):
        super().__init__(params, lr, l2_lambda, l1_lambda)
        self._state.update(beta1=beta1, beta2=beta2, eps=eps)

        if S_KEY not in self._state:
            self._state[S_KEY] = {}
        if V_KEY not in self._state:
            self._state[V_KEY] = {}

        if T_KEY not in self._state:
            self._state[T_KEY] = 0

    def step(self):
        _lr, _b1, _b2, _eps = self._state[LR_KEY], self._state[BETA1_KEY], self._state[BETA2_KEY], self._state[EPS_KEY]
        _vdict: dict[str,np.ndarray] = self._state[V_KEY]
        _sdict: dict[str,np.ndarray] = self._state[S_KEY]

        self._state[T_KEY] += 1
        _t = self._state[T_KEY]

        for idx in range(len(self.params)):
            param = self.params[idx]
            if param.grad is None:
                continue

            # 应用正则化
            grad = self._apply_regularization(param, param.grad.data)
            param.grad.data = grad

            _v = _vdict.get(str(idx), np.zeros_like(param.data))
            _s = _sdict.get(str(idx), np.zeros_like(param.data))

            # v = v*beta1 + (1-beta1)*grad
            _v = _v * _b1 + (1-_b1) * param.grad.data
            # s = s*beta2 + (1-beta2)*grad*grad
            _s = _s * _b2 + (1-_b2 ) * np.power(param.grad.data, 2)

            # 计算修正
            v_hat = _v / (1 - _b1 ** _t)
            s_hat = _s / (1 - _b2 ** _t)

            # w = w - lr*v / sqrt(s + eps)
            param.data -= _lr * v_hat / np.sqrt(s_hat + _eps)

            self._state[V_KEY][str(idx)], self._state[S_KEY][str(idx)] = _v, _s