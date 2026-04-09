# -*- coding: gbk -*-
"""
�߼�������Ԫ���ԣ�
1) ��ȿɷ������ǰ����ֵ��ȷ��
2) ���ž���ǰ����ֵ��ȷ��
3) ת�þ���ǰ����ֵ��ȷ��
4) ��������ݶ���ͨ�Լ��
"""

import sys
from pathlib import Path

import numpy as np

# 添加code目录到Python搜索路径，这样就能找到eneuro模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import as_Tensor  # noqa: E402
from eneuro.base import functions as F  # noqa: E402
from eneuro.base.functions import conv2d, depthwise_conv2d, deconv2d  # noqa: E402
from eneuro.nn.module import Conv2d, Deconv2d, Module  # noqa: E402


def _pair(v):
    if isinstance(v, int):
        return v, v
    return v


def naive_conv2d(x, w, b=None, stride=1, pad=0, dilation=1, groups=1):
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

    cin_per_group = c_in // groups
    oc_per_group = c_out // groups

    for ni in range(n):
        for g in range(groups):
            cin0, cin1 = g * cin_per_group, (g + 1) * cin_per_group
            oc0, oc1 = g * oc_per_group, (g + 1) * oc_per_group

            for oc in range(oc0, oc1):
                for oh in range(h_out):
                    for ow in range(w_out):
                        acc = 0.0
                        for ic in range(cin0, cin1):
                            kc = ic - cin0
                            for ki in range(kh):
                                for kj in range(kw):
                                    ih = oh * sh + ki * dh
                                    iw = ow * sw + kj * dw
                                    acc += x_pad[ni, ic, ih, iw] * w[oc, kc, ki, kj]
                        if b is not None:
                            acc += b[oc]
                        y[ni, oc, oh, ow] = acc
    return y


def naive_deconv2d(x, w, b=None, stride=1, pad=0):
    sh, sw = _pair(stride)
    ph, pw = _pair(pad)

    n, c_in, h, w_in = x.shape
    c_in_w, c_out, kh, kw = w.shape
    assert c_in == c_in_w

    h_out = (h - 1) * sh - 2 * ph + kh
    w_out = (w_in - 1) * sw - 2 * pw + kw

    y = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)

    for ni in range(n):
        for ic in range(c_in):
            for ih in range(h):
                for iw in range(w_in):
                    base_h = ih * sh - ph
                    base_w = iw * sw - pw
                    for oc in range(c_out):
                        for ki in range(kh):
                            for kj in range(kw):
                                oh = base_h + ki
                                ow = base_w + kj
                                if 0 <= oh < h_out and 0 <= ow < w_out:
                                    y[ni, oc, oh, ow] += x[ni, ic, ih, iw] * w[ic, oc, ki, kj]

    if b is not None:
        y += b.reshape(1, -1, 1, 1)

    return y


class SeparableConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.dw = Conv2d(in_channels, kernel_size=kernel_size, stride=stride, pad=pad, in_channels=in_channels, depthwise=True)
        self.pw = Conv2d(out_channels, kernel_size=1, stride=1, pad=0, in_channels=in_channels)

    def forward(self, x):
        return self.pw(self.dw(x))


def run_unit_tests():
    np.random.seed(2026)
    reports = []

    # 1) ��ȿɷ������ǰ����ֵ
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w_dw = np.random.randn(3, 1, 3, 3).astype(np.float32)
    b_dw = np.random.randn(3).astype(np.float32)
    w_pw = np.random.randn(5, 3, 1, 1).astype(np.float32)
    b_pw = np.random.randn(5).astype(np.float32)

    y_impl = conv2d(
        depthwise_conv2d(as_Tensor(x), as_Tensor(w_dw), as_Tensor(b_dw), stride=(1, 1), pad=(1, 1), dilation=(1, 1)),
        as_Tensor(w_pw),
        as_Tensor(b_pw),
        stride=(1, 1),
        pad=(0, 0),
    ).data

    y_ref_dw = naive_conv2d(x, w_dw, b_dw, stride=1, pad=1, dilation=1, groups=3)
    y_ref = naive_conv2d(y_ref_dw, w_pw, b_pw, stride=1, pad=0, dilation=1, groups=1)

    ok = np.allclose(y_impl, y_ref, atol=1e-5, rtol=1e-5)
    reports.append(("��ȿɷ������ǰ����ֵ��ȷ��", ok, float(np.max(np.abs(y_impl - y_ref)))))

    # 2) ���ž���ǰ����ֵ
    x_di = np.random.randn(1, 2, 10, 10).astype(np.float32)
    w_di = np.random.randn(4, 2, 3, 3).astype(np.float32)
    b_di = np.random.randn(4).astype(np.float32)

    y_di = conv2d(as_Tensor(x_di), as_Tensor(w_di), as_Tensor(b_di), stride=(1, 1), pad=(2, 2), dilation=(2, 2)).data
    y_ref_di = naive_conv2d(x_di, w_di, b_di, stride=1, pad=2, dilation=2, groups=1)
    ok = np.allclose(y_di, y_ref_di, atol=1e-5, rtol=1e-5)
    reports.append(("���ž���ǰ����ֵ��ȷ��", ok, float(np.max(np.abs(y_di - y_ref_di)))))

    # 3) ת�þ���ǰ����ֵ
    x_de = np.random.randn(2, 3, 5, 5).astype(np.float32)
    w_de = np.random.randn(3, 4, 3, 3).astype(np.float32)
    b_de = np.random.randn(4).astype(np.float32)

    y_de = deconv2d(as_Tensor(x_de), as_Tensor(w_de), as_Tensor(b_de), stride=(2, 2), pad=(1, 1)).data
    y_ref_de = naive_deconv2d(x_de, w_de, b_de, stride=2, pad=1)
    ok = np.allclose(y_de, y_ref_de, atol=1e-5, rtol=1e-5)
    reports.append(("ת�þ���ǰ����ֵ��ȷ��", ok, float(np.max(np.abs(y_de - y_ref_de)))))

    # 4) �ݶ���ͨ��
    xg = as_Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))

    sep_block = SeparableConvBlock(3, 6)
    y_sep = sep_block(xg)
    loss_sep = (y_sep * y_sep)[0, 0, 0, 0]
    sep_block.cleargrads()
    loss_sep.backward()
    sep_dw_grad = sep_block.dw.W.grad is not None
    sep_pw_grad = sep_block.pw.W.grad is not None
    reports.append(("��ȿɷ�������ݶ���ͨ", sep_dw_grad and sep_pw_grad, 0.0))

    dil_conv = Conv2d(6, kernel_size=3, stride=1, pad=2, in_channels=3, dilation=2)
    y_dil = dil_conv(xg)
    loss_dil = (y_dil * y_dil)[0, 0, 0, 0]
    dil_conv.cleargrads()
    try:
        loss_dil.backward()
        reports.append(("���ž����ݶ���ͨ", dil_conv.W.grad is not None, 0.0))
    except Exception as e:
        reports.append(("���ž����ݶ���ͨ", False, 0.0))
        print(f"[WARN] ���ž������򴫲��쳣: {type(e).__name__}: {e}")

    deconv_layer = Deconv2d(4, kernel_size=3, stride=2, pad=1, in_channels=3)
    y_t = deconv_layer(xg)
    loss_t = (y_t * y_t)[0, 0, 0, 0]
    deconv_layer.cleargrads()
    loss_t.backward()
    reports.append(("ת�þ����ݶ���ͨ", deconv_layer.W.grad is not None, 0.0))

    print("\n========== �߼�������Ԫ���Ա��� ==========")
    pass_count = 0
    for name, ok, stat in reports:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name} | stat={stat:.6f}")
        if ok:
            pass_count += 1

    total = len(reports)
    print(f"\nSummary: {pass_count}/{total} passed")

    if not all(x[1] for x in reports[:3]):
        raise AssertionError("ǰ����ֵ��ȷ�Բ���ʧ�ܣ����޸�ʵ�֡�")

    if not all(x[1] for x in reports[3:]):
        print("[WARN] �ݶ���ͨ�Դ���ʧ����������Ӧ����ʵ�֡�")


if __name__ == "__main__":
    run_unit_tests()
