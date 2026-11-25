import numpy as np
import pytest

from eneuro.base.core import Config, Tensor
from eneuro.base.cuda import dispatch
from eneuro.base.functions import batch_norm2d, fused_conv_bn_relu


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("fused", [False, True], ids=["batchnorm", "fused"])
@pytest.mark.parametrize("backend", [
    "cpu",
    pytest.param("cupy", marks=pytest.mark.cuda),
    pytest.param("rawmodule", marks=pytest.mark.cuda),
])
def test_batchnorm_transposed_input_and_gradients(backend, training, fused, monkeypatch):
    device = "cpu" if backend == "cpu" else "cuda"
    xp = np if device == "cpu" else pytest.importorskip("cupy")
    if device == "cuda":
        if not dispatch.is_available(backend):
            pytest.skip(f"{backend} unavailable")
        monkeypatch.setattr(dispatch, "_BACKEND", backend)
    rng = np.random.default_rng(23)
    data, upstream = [rng.normal(size=(2, 2, 3, 4)).astype("f4") for _ in range(2)]
    # Transpose on the target device to exercise a non-C-contiguous NCHW view.
    x, gy = [Tensor(xp.asarray(a).swapaxes(2, 3), device=device) for a in (data, upstream)]
    data, upstream = data.swapaxes(2, 3), upstream.swapaxes(2, 3)
    gamma, beta, mean, var = [
        Tensor(xp.array(a, dtype="f4"), device=device)
        for a in ([2, -1], [1, -2], [-1, 3], [4, 0.25])
    ]
    with Config.using_config("train", training):
        if fused:
            weight = Tensor(xp.eye(2, dtype="f4").reshape(2, 2, 1, 1), device=device)
            bias = Tensor(xp.zeros(2, dtype="f4"), device=device)
            y = fused_conv_bn_relu(x, weight, bias, gamma, beta, mean, var)
        else:
            y = batch_norm2d((x, gamma, beta), mean, var)
    y.grad = gy
    # Backward must use the mode of its own forward invocation.
    with Config.using_config("train", not training):
        y.backward()

    axes = (0, 2, 3)
    shape = (1, 2, 1, 1)
    m = data.mean(axes) if training else np.array([-1, 3])
    v = data.var(axes) if training else np.array([4, 0.25])
    new_mean = 0.9 * np.array([-1, 3]) + 0.1 * m if training else m
    new_var = 0.9 * np.array([4, 0.25]) + 0.1 * v if training else v
    inv = 1 / np.sqrt(v.reshape(shape) + 1e-5)
    norm = (data - m.reshape(shape)) * inv
    scale = np.array([2, -1]).reshape(shape)
    output = scale * norm + np.array([1, -2]).reshape(shape)
    if fused:
        upstream = upstream * (output > 0)
        output = np.maximum(output, 0)
    gx = upstream
    if training:
        gx = gx - upstream.mean(axes, keepdims=True) - norm * (upstream * norm).mean(axes, keepdims=True)
    expected = (
        output,
        gx * scale * inv,
        (upstream * norm).sum(axes),
        upstream.sum(axes),
        new_mean,
        new_var,
    )
    for result, reference in zip((y, x.grad, gamma.grad, beta.grad, mean, var), expected):
        values = result.data if device == "cpu" else xp.asnumpy(result.data)
        np.testing.assert_allclose(values, reference, rtol=2e-5, atol=2e-5)
