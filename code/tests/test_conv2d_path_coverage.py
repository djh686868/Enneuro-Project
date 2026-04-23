# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import contextlib

import numpy as np

# 确保能导入 code 下的包
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import as_Tensor  # noqa: E402
from eneuro.base import functions as F  # noqa: E402


@contextlib.contextmanager
def temp_winograd_threshold(shape):
    old = F.Conv2d.WINOGRAD_MIN_INPUT_SHAPE
    F.Conv2d.WINOGRAD_MIN_INPUT_SHAPE = shape
    try:
        yield
    finally:
        F.Conv2d.WINOGRAD_MIN_INPUT_SHAPE = old


def _max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def test_path_im2col_selected():
    np.random.seed(0)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)

    with temp_winograd_threshold((999, 999, 999, 999)):
        y = F.conv2d(as_Tensor(x), as_Tensor(w), as_Tensor(b), stride=1, pad=1, dilation=1)
        path = y.creator._select_forward_path(y.creator.inputs[0].data, y.creator.inputs[1].data)

    assert path == 'im2col', f"expected im2col, got {path}"
    assert y.shape == (2, 4, 8, 8), f"unexpected output shape: {y.shape}"


def test_path_gemm_selected_and_correct():
    np.random.seed(1)
    x = np.random.randn(2, 3, 16, 16).astype(np.float32)
    w = np.random.randn(5, 3, 5, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)

    with temp_winograd_threshold((1, 1, 1, 1)):
        y = F.conv2d(as_Tensor(x), as_Tensor(w), as_Tensor(b), stride=1, pad=2, dilation=1)
        path = y.creator._select_forward_path(y.creator.inputs[0].data, y.creator.inputs[1].data)

    assert path == 'gemm', f"expected gemm, got {path}"

    ref_fn = F.Conv2d(stride=1, pad=2, dilation=1)
    y_ref = ref_fn.im2col_conv2d_forward(x, w, b)
    diff = _max_abs_diff(y.data, y_ref)
    assert diff < 1e-4, f"gemm vs im2col mismatch too large: {diff}"


def test_path_winograd_selected_and_correct():
    np.random.seed(2)
    x = np.random.randn(1, 3, 20, 20).astype(np.float32)
    w = np.random.randn(6, 3, 3, 3).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)

    with temp_winograd_threshold((1, 1, 1, 1)):
        y = F.conv2d(as_Tensor(x), as_Tensor(w), as_Tensor(b), stride=1, pad=1, dilation=1)
        path = y.creator._select_forward_path(y.creator.inputs[0].data, y.creator.inputs[1].data)

    assert path == 'winograd', f"expected winograd, got {path}"

    ref_fn = F.Conv2d(stride=1, pad=1, dilation=1)
    y_ref = ref_fn.im2col_conv2d_forward(x, w, b)
    diff = _max_abs_diff(y.data, y_ref)
    # Winograd 与 im2col 允许略大数值误差
    assert diff < 1e-2, f"winograd vs im2col mismatch too large: {diff}"


def test_dilation_backward_smoke():
    np.random.seed(3)
    x = as_Tensor(np.random.randn(1, 3, 11, 11).astype(np.float32))
    w = as_Tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
    b = as_Tensor(np.random.randn(4).astype(np.float32))

    y = F.conv2d(x, w, b, stride=1, pad=2, dilation=2)
    loss = y * y

    x.cleargrad()
    w.cleargrad()
    b.cleargrad()
    loss.backward()

    assert x.grad is not None, "x.grad is None"
    assert w.grad is not None, "w.grad is None"
    assert b.grad is not None, "b.grad is None"
    assert x.grad.shape == x.shape, f"x.grad shape mismatch: {x.grad.shape} vs {x.shape}"
    assert w.grad.shape == w.shape, f"w.grad shape mismatch: {w.grad.shape} vs {w.shape}"
    assert b.grad.shape == b.shape, f"b.grad shape mismatch: {b.grad.shape} vs {b.shape}"


def main():
    test_path_im2col_selected()
    print("[PASS] test_path_im2col_selected")

    test_path_gemm_selected_and_correct()
    print("[PASS] test_path_gemm_selected_and_correct")

    test_path_winograd_selected_and_correct()
    print("[PASS] test_path_winograd_selected_and_correct")

    test_dilation_backward_smoke()
    print("[PASS] test_dilation_backward_smoke")


if __name__ == '__main__':
    main()
