import numpy as np
import os
import sys

# 添加code目录到Python搜索路径，这样就能找到eneuro模块
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base.core import Tensor, Config
from eneuro.base.functions import BatchNormFunction
from eneuro.nn.module import BatchNorm


def numerical_grad(param: np.ndarray, eval_loss, h: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = param[idx]

        param[idx] = old + h
        l1 = eval_loss()

        param[idx] = old - h
        l2 = eval_loss()

        grad[idx] = (l1 - l2) / (2 * h)
        param[idx] = old
        it.iternext()
    return grad


def test_batchnorm_function_smoke_and_moving_stats_update():
    np.random.seed(0)

    moving_mean = np.zeros((1, 3), dtype=np.float32)
    moving_var = np.ones((1, 3), dtype=np.float32)

    x = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    gamma = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
    beta = Tensor(np.zeros((1, 3), dtype=np.float32), requires_grad=True)

    y = BatchNormFunction(training=True, moving_mean=moving_mean, moving_var=moving_var)(x, gamma, beta)
    y.backward()

    assert y.shape == (4, 3)
    assert x.grad is not None
    assert gamma.grad is not None
    assert beta.grad is not None
    assert not np.allclose(moving_mean, np.zeros((1, 3), dtype=np.float32))
    assert not np.allclose(moving_var, np.ones((1, 3), dtype=np.float32))


def test_batchnorm_function_numerical_gradients_2d_and_4d():
    np.random.seed(2026)
    tol = 1e-6

    for shape, num_dims in [((4, 3), 2), ((2, 3, 2, 2), 4)]:
        x_np = np.random.randn(*shape).astype(np.float64)
        gshape = (1, 3) if num_dims == 2 else (1, 3, 1, 1)
        gamma_np = np.random.randn(*gshape).astype(np.float64)
        beta_np = np.random.randn(*gshape).astype(np.float64)
        gy_np = np.random.randn(*shape).astype(np.float64)

        f = BatchNormFunction(eps=1e-5, training=True, moving_mean=None, moving_var=None)
        _ = f(Tensor(x_np.copy()), Tensor(gamma_np.copy()), Tensor(beta_np.copy()))
        dx, dgamma, dbeta = f.backward(Tensor(gy_np))
        dx, dgamma, dbeta = dx.data, dgamma.data, dbeta.data

        def loss_x():
            y = BatchNormFunction(eps=1e-5, training=True, moving_mean=None, moving_var=None)(
                Tensor(x_np), Tensor(gamma_np), Tensor(beta_np)
            ).data
            return float((y * gy_np).sum())

        def loss_gamma():
            y = BatchNormFunction(eps=1e-5, training=True, moving_mean=None, moving_var=None)(
                Tensor(x_np), Tensor(gamma_np), Tensor(beta_np)
            ).data
            return float((y * gy_np).sum())

        def loss_beta():
            y = BatchNormFunction(eps=1e-5, training=True, moving_mean=None, moving_var=None)(
                Tensor(x_np), Tensor(gamma_np), Tensor(beta_np)
            ).data
            return float((y * gy_np).sum())

        num_dx = numerical_grad(x_np, loss_x)
        num_dgamma = numerical_grad(gamma_np, loss_gamma)
        num_dbeta = numerical_grad(beta_np, loss_beta)

        dx_err = np.max(np.abs(dx - num_dx))
        dgamma_err = np.max(np.abs(dgamma - num_dgamma))
        dbeta_err = np.max(np.abs(dbeta - num_dbeta))

        assert dx_err < tol, f"dx error too large for shape={shape}: {dx_err}"
        assert dgamma_err < tol, f"dgamma error too large for shape={shape}: {dgamma_err}"
        assert dbeta_err < tol, f"dbeta error too large for shape={shape}: {dbeta_err}"


def test_batchnorm_layer_train_eval_behavior():
    np.random.seed(7)

    bn = BatchNorm(num_features=3, num_dims=2, dtype=np.float64)

    old_train_flag = Config.train
    try:
        Config.train = True
        x_train = Tensor(np.random.randn(8, 3).astype(np.float64), requires_grad=True)
        y_train = bn(x_train)
        y_train.backward()

        assert y_train.shape == (8, 3)
        assert x_train.grad is not None
        assert bn.gamma.grad is not None
        assert bn.beta.grad is not None

        moving_mean_before_eval = bn.moving_mean.copy()
        moving_var_before_eval = bn.moving_var.copy()

        Config.train = False
        x_eval_np = (np.random.randn(8, 3) * 2.0 + 5.0).astype(np.float64)
        x_eval = Tensor(x_eval_np, requires_grad=False)
        y_eval = bn(x_eval).data

        expected = bn.gamma.data * ((x_eval_np - moving_mean_before_eval) / np.sqrt(moving_var_before_eval + bn.eps)) + bn.beta.data

        assert np.allclose(y_eval, expected, atol=1e-10, rtol=1e-8)
        assert np.allclose(bn.moving_mean, moving_mean_before_eval)
        assert np.allclose(bn.moving_var, moving_var_before_eval)
    finally:
        Config.train = old_train_flag


if __name__ == '__main__':
    test_batchnorm_function_smoke_and_moving_stats_update()
    test_batchnorm_function_numerical_gradients_2d_and_4d()
    test_batchnorm_layer_train_eval_behavior()
    print('test_batchnorm.py: all tests passed')
