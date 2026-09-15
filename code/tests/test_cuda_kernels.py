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
    yield
    cuda.set_backend("cupy")


def _gpu_available():
    return cuda.is_available("rawmodule")


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


def test_backend_switch_without_gpu_requirement():
    cuda.set_backend("cupy")
    assert cuda.get_backend() == "cupy"
    with pytest.raises(ValueError):
        cuda.set_backend("invalid")
