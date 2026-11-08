"""Model parameter snapshots and strict loading by stable name."""

import numpy as np
from ..base import Tensor


def named_state(model):
    return {name: np.array(param.data, copy=True) for name, param in model.named_parameters()}


def load_state(model, arrays):
    parameters = dict(model.named_parameters())
    if set(arrays) != set(parameters):
        raise ValueError('parameter names differ')
    for name, param in parameters.items():
        value = np.asarray(arrays[name])
        if param.data is not None and (value.shape != param.data.shape or value.dtype != param.data.dtype):
            raise ValueError(f'parameter shape/dtype mismatch: {name}')
        param.data = value.copy()
        param.grad = None


def install_gradients(model, gradients):
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(gradients) != set(parameters):
        raise ValueError('gradient names differ')
    model.cleargrads()
    for name, param in parameters.items():
        value = np.asarray(gradients[name])
        if value.shape != param.data.shape:
            raise ValueError(f'gradient shape mismatch: {name}')
        param.grad = Tensor(value.astype(param.data.dtype, copy=True), requires_grad=False)
