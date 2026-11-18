"""Focused correctness checks for the ResNet18 CUDA extension stage.

These tests deliberately compare only the new kernels against CuPy formulas;
the existing fast-route test continues to cover the stage-one elementwise,
im2col and pooling kernels.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cp = pytest.importorskip("cupy")
from eneuro.base.cuda import dispatch  # noqa: E402


pytestmark = pytest.mark.cuda


def _require_gpu():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")


def test_winograd_forward_matches_im2col():
    _require_gpu()
    dispatch.set_backend("rawmodule")
    rng = np.random.default_rng(20260924)
    x = cp.asarray(rng.normal(size=(2, 3, 7, 5)).astype(np.float32))
    w = cp.asarray(rng.normal(size=(4, 3, 3, 3)).astype(np.float32))
    b = cp.asarray(rng.normal(size=(4,)).astype(np.float32))
    y = dispatch.conv2d_forward(x, w, b, stride=1, pad=1)
    from eneuro.base.functions import im2col_array
    col = im2col_array(x, (3, 3), (1, 1), (1, 1), True, xp=cp)
    ref = col.dot(w.reshape(4, -1).T).reshape(2, 7, 5, 4).transpose(0, 3, 1, 2)
    ref = ref + b.reshape(1, 4, 1, 1)
    np.testing.assert_allclose(cp.asnumpy(y), cp.asnumpy(ref), rtol=2e-5, atol=3e-5)


def test_batchnorm_and_global_average_pooling_match_cupy():
    _require_gpu()
    dispatch.set_backend("rawmodule")
    rng = np.random.default_rng(20260924)
    x = cp.asarray(rng.normal(size=(2, 3, 4, 5)).astype(np.float32))
    gy = cp.asarray(rng.normal(size=x.shape).astype(np.float32))
    mean, var = x.mean((0, 2, 3)), x.var((0, 2, 3))
    gamma = cp.asarray(rng.normal(size=3).astype(np.float32))
    beta = cp.asarray(rng.normal(size=3).astype(np.float32))
    y = dispatch.batchnorm_forward(x, mean, var, gamma, beta)
    ref = gamma.reshape(1, 3, 1, 1) * (x - mean.reshape(1, 3, 1, 1)) / cp.sqrt(var.reshape(1, 3, 1, 1) + 1e-5) + beta.reshape(1, 3, 1, 1)
    np.testing.assert_allclose(cp.asnumpy(y), cp.asnumpy(ref), rtol=2e-5, atol=3e-5)
    pooled = dispatch.global_average_pool_forward(x)
    np.testing.assert_allclose(cp.asnumpy(pooled), cp.asnumpy(x.mean((2, 3), keepdims=True)), rtol=1e-6, atol=1e-6)
