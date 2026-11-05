# -*- coding: utf-8 -*-
"""最小 FFT 卷积路径回归测试。

目标：
1. 确认 Conv2d 的路径选择能够进入 FFT 分支。
2. 确认 FFT 前向输出与朴素卷积结果一致。

这个脚本直接加载 base 模块，避免被包入口里的可选依赖卡住。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"


def _pair(value):
    if isinstance(value, int):
        return value, value
    return value


def naive_conv2d(x, w, b=None, stride=1, pad=0, dilation=1):
    sh, sw = _pair(stride)
    ph, pw = _pair(pad)
    dh, dw = _pair(dilation)

    n, c_in, h, w_in = x.shape
    c_out, _, kh, kw = w.shape
    h_out = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    y = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)

    for ni in range(n):
        for oc in range(c_out):
            for oh in range(h_out):
                for ow in range(w_out):
                    acc = 0.0
                    for ic in range(c_in):
                        for kh_idx in range(kh):
                            for kw_idx in range(kw):
                                ih = oh * sh + kh_idx * dh
                                iw = ow * sw + kw_idx * dw
                                acc += x_pad[ni, ic, ih, iw] * w[oc, ic, kh_idx, kw_idx]
                    if b is not None:
                        acc += b[oc]
                    y[ni, oc, oh, ow] = acc
    return y


def load_base_modules():
    eneuro_pkg = types.ModuleType("eneuro")
    eneuro_pkg.__path__ = [str(CODE_DIR / "eneuro")]
    sys.modules["eneuro"] = eneuro_pkg

    base_pkg = types.ModuleType("eneuro.base")
    base_pkg.__path__ = [str(CODE_DIR / "eneuro" / "base")]
    sys.modules["eneuro.base"] = base_pkg

    utils_mod = types.ModuleType("eneuro.utils")

    class StateDict:
        pass

    utils_mod.StateDict = StateDict
    sys.modules["eneuro.utils"] = utils_mod

    global_config_mod = types.ModuleType("eneuro.global_config")
    global_config_mod.VISUAL_CONFIG = {"ENABLE_ALL_LAYERS": False}
    sys.modules["eneuro.global_config"] = global_config_mod

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.destroyAllWindows = lambda: None
    sys.modules["cv2"] = cv2_mod

    placeholder_functions = types.ModuleType("eneuro.base.functions")
    sys.modules["eneuro.base.functions"] = placeholder_functions

    core_spec = importlib.util.spec_from_file_location(
        "eneuro.base.core",
        CODE_DIR / "eneuro" / "base" / "core.py",
    )
    core_mod = importlib.util.module_from_spec(core_spec)
    sys.modules["eneuro.base.core"] = core_mod
    core_spec.loader.exec_module(core_mod)

    func_spec = importlib.util.spec_from_file_location(
        "eneuro.base.functions",
        CODE_DIR / "eneuro" / "base" / "functions.py",
    )
    func_mod = importlib.util.module_from_spec(func_spec)
    sys.modules["eneuro.base.functions"] = func_mod
    func_spec.loader.exec_module(func_mod)

    return core_mod, func_mod


def main():
    core_mod, func_mod = load_base_modules()
    as_Tensor = core_mod.as_Tensor
    conv2d = func_mod.conv2d
    Conv2d = func_mod.Conv2d

    # 把阈值调低，保证这个测试在小尺寸输入上也能稳定命中 FFT 分支。
    old_kernel_threshold = Conv2d.FFT_MIN_KERNEL_SIZE
    old_spatial_threshold = Conv2d.FFT_MIN_SPATIAL_SIZE
    Conv2d.FFT_MIN_KERNEL_SIZE = 1
    Conv2d.FFT_MIN_SPATIAL_SIZE = 1

    try:
        np.random.seed(0)
        x = np.random.randn(1, 2, 8, 8).astype(np.float32)
        w = np.random.randn(3, 2, 5, 5).astype(np.float32)
        b = np.random.randn(3).astype(np.float32)

        layer = Conv2d(stride=(1, 1), pad=(2, 2), dilation=1)
        path = layer._select_forward_path(x, w)
        assert path == "fft", f"expected fft path, got {path!r}"

        y = conv2d(
            as_Tensor(x),
            as_Tensor(w),
            as_Tensor(b),
            stride=(1, 1),
            pad=(2, 2),
            dilation=1,
        ).data
        y_ref = naive_conv2d(x, w, b, stride=1, pad=2, dilation=1)

        ok = np.allclose(y, y_ref, atol=1e-4, rtol=1e-4)
        max_diff = float(np.max(np.abs(y - y_ref)))

        print(f"path={path}")
        print(f"allclose={ok}")
        print(f"maxdiff={max_diff}")

        assert ok, f"fft conv2d mismatch, max diff={max_diff}"
    finally:
        Conv2d.FFT_MIN_KERNEL_SIZE = old_kernel_threshold
        Conv2d.FFT_MIN_SPATIAL_SIZE = old_spatial_threshold


if __name__ == "__main__":
    main()