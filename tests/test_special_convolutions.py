# -*- coding: gbk -*-
"""
特殊卷积单元测试：
1) grouped_conv2d 数值正确性
2) depthwise_conv2d 数值正确性（channel_multiplier=1）
3) dilation conv2d 数值正确性
4) 特殊卷积层反向传播梯度连通性检查
"""

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from eneuro.base import as_Tensor  # noqa: E402
from eneuro.base.functions import conv2d, grouped_conv2d, depthwise_conv2d  # noqa: E402
from eneuro.nn.module import Conv2d  # noqa: E402


def _pair(v):
    if isinstance(v, int):
        return v, v
    return v


def naive_conv2d(x, w, b=None, stride=1, pad=0, dilation=1, groups=1):
    """纯 numpy 朴素卷积，支持 groups / dilation。"""
    sh, sw = _pair(stride)
    ph, pw = _pair(pad)
    dh, dw = _pair(dilation)

    n, c_in, h, w_in = x.shape
    c_out, c_per_group, kh, kw = w.shape
    assert c_in % groups == 0
    assert c_out % groups == 0

    h_out = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    y = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)

    oc_per_group = c_out // groups
    cin_per_group = c_in // groups

    for ni in range(n):
        for g in range(groups):
            cin_beg = g * cin_per_group
            cin_end = (g + 1) * cin_per_group
            oc_beg = g * oc_per_group
            oc_end = (g + 1) * oc_per_group

            for oc in range(oc_beg, oc_end):
                for oh in range(h_out):
                    for ow in range(w_out):
                        acc = 0.0
                        for ic in range(cin_beg, cin_end):
                            kc = ic - cin_beg
                            for ki in range(kh):
                                for kj in range(kw):
                                    ih = oh * sh + ki * dh
                                    iw = ow * sw + kj * dw
                                    acc += x_pad[ni, ic, ih, iw] * w[oc, kc, ki, kj]
                        if b is not None:
                            acc += b[oc]
                        y[ni, oc, oh, ow] = acc
    return y


def run_unit_tests():
    np.random.seed(2026)
    reports = []

    # 1) grouped_conv2d correctness
    x = np.random.randn(2, 4, 7, 7).astype(np.float32)
    w = np.random.randn(6, 2, 3, 3).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)

    y_impl = grouped_conv2d(as_Tensor(x), as_Tensor(w), as_Tensor(b), stride=(1, 1), pad=(1, 1), groups=2, dilation=(1, 1)).data
    y_ref = naive_conv2d(x, w, b, stride=1, pad=1, dilation=1, groups=2)
    ok = np.allclose(y_impl, y_ref, atol=1e-5, rtol=1e-5)
    reports.append(("grouped_conv2d 前向数值正确性", ok, float(np.max(np.abs(y_impl - y_ref)))))

    # 2) depthwise_conv2d correctness (multiplier=1)
    x_dw = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w_dw = np.random.randn(3, 1, 3, 3).astype(np.float32)
    b_dw = np.random.randn(3).astype(np.float32)

    y_dw = depthwise_conv2d(as_Tensor(x_dw), as_Tensor(w_dw), as_Tensor(b_dw), stride=(1, 1), pad=(1, 1), dilation=(1, 1)).data

    # depthwise(multiplier=1) 等价于 groups=C 的 grouped conv
    w_group = np.zeros((3, 1, 3, 3), dtype=np.float32)
    w_group[:] = w_dw
    y_ref_dw = naive_conv2d(x_dw, w_group, b_dw, stride=1, pad=1, dilation=1, groups=3)
    ok = np.allclose(y_dw, y_ref_dw, atol=1e-5, rtol=1e-5)
    reports.append(("depthwise_conv2d 前向数值正确性", ok, float(np.max(np.abs(y_dw - y_ref_dw)))))

    # 3) dilation correctness (normal conv)
    x_di = np.random.randn(1, 2, 9, 9).astype(np.float32)
    w_di = np.random.randn(4, 2, 3, 3).astype(np.float32)
    b_di = np.random.randn(4).astype(np.float32)

    y_di = conv2d(as_Tensor(x_di), as_Tensor(w_di), as_Tensor(b_di), stride=(1, 1), pad=(2, 2), dilation=(2, 2)).data
    y_ref_di = naive_conv2d(x_di, w_di, b_di, stride=1, pad=2, dilation=2, groups=1)
    ok = np.allclose(y_di, y_ref_di, atol=1e-5, rtol=1e-5)
    reports.append(("dilation conv2d 前向数值正确性", ok, float(np.max(np.abs(y_di - y_ref_di)))))

    # 4) 梯度连通性检查
    xg = as_Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))

    conv_normal = Conv2d(6, kernel_size=3, pad=1, in_channels=4)
    y_normal = conv_normal(xg)
    loss_normal = (y_normal * y_normal)[0, 0, 0, 0]
    conv_normal.cleargrads()
    loss_normal.backward()
    normal_has_grad = conv_normal.W.grad is not None
    reports.append(("普通 Conv2d 梯度连通", normal_has_grad, 0.0))

    conv_group = Conv2d(6, kernel_size=3, pad=1, in_channels=4, groups=2)
    y_group = conv_group(xg)
    loss_group = (y_group * y_group)[0, 0, 0, 0]
    conv_group.cleargrads()
    loss_group.backward()
    group_has_grad = conv_group.W.grad is not None
    reports.append(("分组 Conv2d 梯度连通", group_has_grad, 0.0))

    conv_depth = Conv2d(4, kernel_size=3, pad=1, in_channels=4, depthwise=True)
    y_depth = conv_depth(xg)
    loss_depth = (y_depth * y_depth)[0, 0, 0, 0]
    conv_depth.cleargrads()
    loss_depth.backward()
    depth_has_grad = conv_depth.W.grad is not None
    reports.append(("深度卷积 Conv2d 梯度连通", depth_has_grad, 0.0))

    print("\n========== 特殊卷积单元测试报告 ==========")
    pass_count = 0
    for name, ok, stat in reports:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name} | stat={stat:.6f}")
        if ok:
            pass_count += 1

    total = len(reports)
    print(f"\nSummary: {pass_count}/{total} passed")

    # 至少保证前向数值测试都通过；梯度连通性允许暴露问题。
    forward_ok = all(item[1] for item in reports[:3])
    if not forward_ok:
        raise AssertionError("前向数值正确性测试存在失败，请优先修复卷积实现。")

    if not group_has_grad or not depth_has_grad:
        print("\n[WARN] 检测到分组/深度卷积参数未获得梯度，这会影响训练优化效果。")


if __name__ == "__main__":
    run_unit_tests()
