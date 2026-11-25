"""Fast-route CUDA C smoke and numerical tests.

Run inside the EnNeuro environment.  GPU cases skip cleanly when CuPy/device
is unavailable; CPU import/fallback checks remain runnable everywhere.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cp = pytest.importorskip("cupy")
from eneuro.base.cuda import dispatch as cuda


pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _raw_backend():
    cuda.set_backend("rawmodule")
    cuda.diagnostics(reset=True)
    cuda.launch_counts(reset=True)
    yield
    cuda.set_backend("cupy")


def _gpu_available():
    return cuda.is_available("rawmodule")


@pytest.mark.parametrize("forward", [cuda.conv2d_forward, cuda.winograd_conv2d_forward])
def test_conv_validates_input_channels_before_dispatch(forward, monkeypatch):
    # Geometry validation is device independent and must precede any launch.
    monkeypatch.setattr(cuda, "_call_extra", lambda *args: pytest.fail("unexpected launch"))
    x = np.empty((2, 3, 5, 7), dtype=np.float32)
    w = np.empty((4, 1, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="channels"):
        forward(x, w)


def test_winograd_validates_kernel_shape_and_dtype():
    x = np.empty((1, 3, 5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="3x3"):
        cuda.winograd_conv2d_forward(x, np.empty((4, 3, 1, 1), dtype=np.float32))
    with pytest.raises(TypeError, match="float32"):
        cuda.winograd_conv2d_forward(x, np.empty((4, 3, 3, 3), dtype=np.float64))


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 5, 7)])
def test_bias_kernel_launch_matches_output_shape(monkeypatch, shape):
    # Verify the launch contract before executing the kernel. Correct values
    # alone cannot detect extra writes by threads in the final partial block.
    launch = cuda._call

    def checked_launch(name, args, count):
        _, _, output, batch, channels, spatial = args
        assert name == "bias_f32"
        assert batch == shape[0] and channels == shape[1]
        assert batch * channels * spatial == count == output.size
        launch(name, args, count)

    monkeypatch.setattr(cuda, "_call", checked_launch)
    x = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
    bias = cp.arange(shape[1], dtype=cp.float32)
    y = cuda.bias_add_forward(x, bias)
    expected = x + bias.reshape((1, shape[1]) + (1,) * (len(shape) - 2))
    cp.testing.assert_array_equal(y, expected)
    assert cuda.launch_counts().get("bias_f32") == 1


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_relu_nonfinite_matches_cupy():
    x = cp.asarray([cp.nan, -cp.inf, 0, -1, cp.inf], dtype=cp.float32)
    gy = cp.asarray([1, cp.inf, cp.nan, -2, -3], dtype=cp.float32)
    cp.testing.assert_array_equal(cuda.relu_forward(x), cp.maximum(x, 0))
    cp.testing.assert_array_equal(cuda.relu_backward(x, gy), gy * (x > 0))


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_pool_nan_propagation_and_first_index():
    x = cp.asarray([1, cp.nan, 5, cp.nan], dtype=cp.float32).reshape(1, 1, 2, 2)
    y, indexes = cuda.maxpool_forward(x, 2)
    assert cp.isnan(y).item()
    assert indexes.item() == 1
    gx = cuda.maxpool_backward(cp.ones_like(y), indexes, x.shape, 2)
    cp.testing.assert_array_equal(gx.ravel(), cp.asarray([0, 1, 0, 0]))


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
@pytest.mark.parametrize("shape", [(1,), (17,), (2, 3, 5)])
def test_elementwise_relu_exp(shape):
    rng = np.random.default_rng(20260909)
    x = cp.asarray(rng.normal(size=shape).astype(np.float32))
    np.testing.assert_allclose(cp.asnumpy(cuda.relu_forward(x)), np.maximum(cp.asnumpy(x), 0), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(cuda.exp_forward(x)), np.exp(cp.asnumpy(x)), atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_im2col_layout_and_values():
    x = cp.arange(1 * 2 * 5 * 7, dtype=cp.float32).reshape(1, 2, 5, 7)
    col = cuda.im2col_forward(x, (3, 2), stride=(2, 1), pad=(1, 2), dilation=(1, 2))
    from eneuro.base.functions import im2col_array
    ref = im2col_array(x, (3, 2), (2, 1), (1, 2), True, dilation=(1, 2), xp=cp)
    np.testing.assert_allclose(cp.asnumpy(col), cp.asnumpy(ref), atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_conv_forward_matches_cupy_reference():
    rng = np.random.default_rng(20260909)
    x = cp.asarray(rng.normal(size=(2, 3, 7, 5)).astype(np.float32))
    w = cp.asarray(rng.normal(size=(4, 3, 3, 3)).astype(np.float32))
    b = cp.asarray(rng.normal(size=(4,)).astype(np.float32))
    y = cuda.conv2d_forward(x, w, b, stride=(2, 1), pad=(1, 2))
    from eneuro.base.functions import im2col_array
    col = im2col_array(x, (3, 3), (2, 1), (1, 2), True, xp=cp)
    # OH=(H+2P-K)//S+1=4，OW=(W+2P-K)//S+1=7；不要使用固定的错误尺寸。
    oh, ow = y.shape[2:]
    ref = col.dot(w.reshape(4, -1).T).reshape(2, oh, ow, 4).transpose(0, 3, 1, 2) + b.reshape(1, 4, 1, 1)
    np.testing.assert_allclose(cp.asnumpy(y), cp.asnumpy(ref), atol=5e-5, rtol=2e-4)


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_pool_forward_tie_rule():
    x = cp.asarray([[[[1, 2], [2, 0]]]], dtype=cp.float32)
    y, idx = cuda.maxpool_forward(x, 2, 2)
    assert int(cp.asnumpy(idx)[0, 0, 0, 0]) == 1
    assert float(cp.asnumpy(y)[0, 0, 0, 0]) == 2.0


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_conv_backward_input_and_weights_match_cupy_reference():
    rng = np.random.default_rng(20260915)
    x = cp.asarray(rng.normal(size=(2, 2, 7, 5)).astype(np.float32))
    w = cp.asarray(rng.normal(size=(3, 2, 3, 2)).astype(np.float32))
    gy = cp.asarray(rng.normal(size=(2, 3, 4, 6)).astype(np.float32))
    gx, gw, gb = cuda.conv2d_backward(gy, x, w, cp.zeros(3, dtype=cp.float32),
                                      stride=(2, 1), pad=(1, 1))
    from eneuro.base.functions import conv2d_backward_input_array, im2col_array
    ref_gx = conv2d_backward_input_array(gy, w, stride=(2, 1), pad=(1, 1),
                                         out_h=x.shape[2], out_w=x.shape[3])
    col = im2col_array(x, (3, 2), (2, 1), (1, 1), True, xp=cp)
    ref_gw = gy.transpose(0, 2, 3, 1).reshape(-1, 3).T.dot(col).reshape(w.shape)
    np.testing.assert_allclose(cp.asnumpy(gx), cp.asnumpy(ref_gx), atol=1e-4, rtol=2e-4)
    np.testing.assert_allclose(cp.asnumpy(gw), cp.asnumpy(ref_gw), atol=1e-4, rtol=2e-4)
    np.testing.assert_allclose(cp.asnumpy(gb), cp.asnumpy(gy.sum(axis=(0, 2, 3))), atol=1e-5)
    assert cuda.launch_counts().get("conv_bwd_x_f32", 0) > 0


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_pool_backward_overlap_matches_cupy_reference():
    x = cp.asarray([[[[1, 3, 3], [2, 5, 4], [2, 5, 1]]]], dtype=cp.float32)
    gy = cp.asarray([[[[1, 2], [3, 4]]]], dtype=cp.float32)
    _, idx = cuda.maxpool_forward(x, 2, stride=1)
    gx = cuda.maxpool_backward(gy, idx, x.shape, 2, stride=1)
    # CPU oracle 遍历输出窗口，并按首次最大值索引把上游梯度散射回输入。
    ref = np.zeros(x.shape, dtype=np.float32)
    indices = cp.asnumpy(idx)
    for oh in range(2):
        for ow in range(2):
            kh, kw = divmod(int(indices[0, 0, oh, ow]), 2)
            ref[0, 0, oh + kh, ow + kw] += float(gy[0, 0, oh, ow].get())
    np.testing.assert_allclose(cp.asnumpy(gx), ref, atol=1e-5)
    assert cuda.launch_counts().get("pool_bwd_f32", 0) > 0


def test_backend_switch_without_gpu_requirement():
    cuda.set_backend("cupy")
    assert cuda.get_backend() == "cupy"
    with pytest.raises(ValueError):
        cuda.set_backend("invalid")


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_scalar_tensor_preserves_rank_through_backward():
    from eneuro.base import Tensor, functions as F

    x = Tensor(cp.asarray(2, dtype=cp.float32), device="cuda")
    assert cuda.pow_forward(x.data, 2).shape == ()
    activated = F.relu(x)
    assert activated.shape == ()
    total = activated + x
    assert total.shape == ()
    squared = total ** 2
    assert squared.shape == ()
    squared.backward()
    assert x.grad.shape == ()
    assert float(squared.data) == 16
    assert float(x.grad.data) == 16


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_relu_backward_broadcasts_scalar_gradient():
    x = cp.asarray([-1, 0, 3], dtype=cp.float32)
    gy = cp.asarray(4, dtype=cp.float32)
    cp.testing.assert_array_equal(cuda.relu_backward(x, gy), gy * (x > 0))


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_conv_backward_on_transposed_input():
    x = cp.arange(12, dtype=cp.float32).reshape(1, 1, 3, 4).swapaxes(2, 3)
    gy = cp.arange(x.size, dtype=x.dtype).reshape(x.shape)
    weights = cp.ones((1, 1, 1, 1), dtype=x.dtype)
    gx, _, _ = cuda.conv2d_backward(gy, x, weights)
    cp.testing.assert_array_equal(gx, gy)
    assert cuda.launch_counts().get("conv_bwd_x_f32") == 1


@pytest.mark.skipif(not _gpu_available(), reason="CUDA/NVRTC unavailable")
def test_empty_outputs_skip_raw_launches():
    x = cp.empty((0, 2, 3, 4), dtype=cp.float32)
    assert cuda.pow_forward(x, 2).shape == x.shape
    assert cuda.global_average_pool_forward(x).shape == (0, 2, 1, 1)
    assert cuda.launch_counts() == {}
