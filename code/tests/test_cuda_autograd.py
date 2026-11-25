"""Exercise CUDA gradients through the public Tensor operations."""

import numpy as np
import pytest

from eneuro.base import Tensor, conv2d, global_average_pooling, pooling
from eneuro.base.core import Function
from eneuro.base.cuda import dispatch


@pytest.fixture(params=["cpu", pytest.param("cupy", marks=pytest.mark.cuda),
                        pytest.param("rawmodule", marks=pytest.mark.cuda)])
def device(request, monkeypatch):
    backend = request.param
    if backend == "cpu":
        return "cpu"
    if not dispatch.is_available(backend):
        pytest.skip(f"{backend} unavailable")
    monkeypatch.setattr(dispatch, "_BACKEND", backend)
    return "cuda"


@pytest.mark.parametrize("operation", ["multiply", "divide"])
def test_broadcast_gradients(device, operation):
    a = np.array([[1], [5]], dtype=np.float32)
    b = np.array([2, 4, 8], dtype=np.float32)
    weights = np.array([[1, 2, 3], [-2, 4, 1]], dtype=np.float32)
    left, right = Tensor(a, device=device), Tensor(b, device=device)
    if operation == "multiply":
        output = left * right
        expected_left = (weights * b).sum(axis=1, keepdims=True)
        expected_right = (weights * a).sum(axis=0)
    else:
        output = left / right
        expected_left = (weights / b).sum(axis=1, keepdims=True)
        expected_right = (-weights * a / b**2).sum(axis=0)

    output.grad = Tensor(weights, device=device)
    output.backward()
    np.testing.assert_allclose(left.grad.to("cpu").data, expected_left)
    np.testing.assert_allclose(right.grad.to("cpu").data, expected_right)


def test_global_pool_backward(device, monkeypatch):
    monkeypatch.setattr(dispatch, "_RAW_CUSTOM_ELEMENTWISE_MAX_SIZE", 4096)
    values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    weights = np.array([1, 2, 3, 4], dtype=np.float32).reshape(2, 2, 1, 1)
    x = Tensor(values, device=device)
    pool = global_average_pooling(x)
    pool.grad = Tensor(weights, device=device)
    kernel = "global_avg_bwd_f32"
    before = dispatch.launch_counts().get(kernel, 0)
    pool.backward()
    after = dispatch.launch_counts().get(kernel, 0)

    expected = np.broadcast_to(weights / 12, x.shape)
    np.testing.assert_allclose(x.grad.to("cpu").data, expected)
    if device != "cpu":
        assert x.grad.dtype == x.dtype
    assert after - before == int(device == "cuda" and dispatch.get_backend() == "rawmodule")


@pytest.mark.parametrize("error", [AttributeError, ImportError])
def test_operator_errors_propagate_without_changing_backend(error):
    class BrokenOperator(Function):
        backend = "cupy"

        def forward(self, x):
            raise error("operator failure")

    previous = dispatch.get_backend()
    with pytest.raises(error, match="operator failure"):
        BrokenOperator()(Tensor(np.ones(1)))
    assert dispatch.get_backend() == previous


@pytest.mark.cuda
@pytest.mark.parametrize("operator,phase,kernel", [
    ("conv", "forward", "winograd_f32"),
    ("conv", "backward", "conv_bwd_x_f32"),
    ("pool", "forward", "pool_f32"),
    ("pool", "backward", "pool_bwd_f32"),
])
def test_auto_backend_recovers_from_launch_failure(monkeypatch, operator, phase, kernel):
    if not dispatch.is_available("rawmodule"):
        pytest.skip("CUDA/NVRTC unavailable")
    for threshold in ("_WINOGRAD_MAX_OUTPUT", "_RAW_REFERENCE_CONV_OUTPUT",
                      "_RAW_CUSTOM_ELEMENTWISE_MAX_SIZE"):
        monkeypatch.setattr(dispatch, threshold, 4096)
    launch_name = "_call_extra" if kernel == "winograd_f32" else "_call"
    original = getattr(dispatch, launch_name)
    attempts = []

    def fail_target(name, args, size):
        if name == kernel:
            attempts.append(dispatch.get_backend())
            raise RuntimeError("injected launch error")
        return original(name, args, size)

    monkeypatch.setattr(dispatch, launch_name, fail_target)
    monkeypatch.setattr(dispatch, "_BACKEND", "rawmodule")
    values = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    x = Tensor(values, device="cuda")
    if operator == "conv":
        weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
        weights[:, :, 1, 1] = 2
        weight = Tensor(weights, device="cuda")
        def forward(mode):
            return conv2d(x, weight, pad=1, backend=mode)
        expected_y = 2 * values
        expected_gx = np.full_like(values, 2)
    else:
        def forward(mode):
            return pooling(x, kernel_size=2, stride=2, backend=mode)
        expected_y = values[:, :, 1::2, 1::2]
        expected_gx = np.zeros_like(values)
        expected_gx[:, :, 1::2, 1::2] = 1

    previous = dispatch.get_backend()
    y = forward("auto")
    if phase == "forward":
        assert y.creator._backend_effective == "cupy_fallback"
    y.backward()
    np.testing.assert_allclose(y.to("cpu").data, expected_y)
    np.testing.assert_allclose(x.grad.to("cpu").data, expected_gx)
    assert "injected launch error" in y.creator._backend_fallback_reason
    assert dispatch.get_backend() == previous

    with pytest.raises(RuntimeError, match="injected launch error"):
        strict_output = forward("cuda_c")
        if phase == "backward":
            strict_output.backward()
    assert attempts == ["auto", "rawmodule"]
    assert dispatch.get_backend() == previous
