"""快速验证路线的独立测试脚本。

本脚本不依赖 pytest，适合直接在 EnNeuro 环境中运行：

    $env:PYTHONPATH = "code"
    python code/test_cuda_fast_route.py --backend cupy

指定 ``--backend rawmodule`` 时会强制编译并执行自研 CUDA C kernel；
``--backend auto`` 会在 kernel 编译失败时回退到 CuPy，但报告会明确标记
回退原因。所有数值 oracle 都使用独立的 CuPy 表达式计算。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def close(name, got, expected, atol=1e-5, rtol=1e-5):
    """比较 GPU 数组并返回可序列化的误差记录。"""
    import cupy as cp

    g = cp.asnumpy(got)
    e = cp.asnumpy(expected)
    diff = np.abs(g.astype(np.float64) - e.astype(np.float64))
    scale = np.maximum(np.abs(e), 1e-12)
    ok = bool(np.all(diff <= atol + rtol * scale))
    return {
        "name": name,
        "passed": ok,
        "max_abs_error": float(diff.max(initial=0.0)),
        "max_rel_error": float((diff / scale).max(initial=0.0)),
        "shape": list(g.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("cupy", "rawmodule", "auto"), default="cupy")
    ap.add_argument("--out", default="artifacts/cuda_stage1_probe/fast_route_test.json")
    args = ap.parse_args()

    report = {"backend": args.backend, "tests": [], "status": "started"}
    try:
        import cupy as cp
    except Exception as exc:
        report.update(status="no_cupy", error=repr(exc))
        return finish(report, args.out, 2)

    if int(cp.cuda.runtime.getDeviceCount()) == 0:
        report.update(status="no_cuda_device")
        return finish(report, args.out, 2)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eneuro.base.cuda import dispatch

    dispatch.set_backend(args.backend)
    report.update(cupy=cp.__version__, device=cp.cuda.Device().compute_capability)

    # 固定种子保证 CuPy 路线和 RawModule 路线每次比较同一批数据。
    rs = cp.random.RandomState(1234)
    x = rs.standard_normal((257,), dtype=cp.float32)
    a = rs.standard_normal((257,), dtype=cp.float32)
    b = rs.standard_normal((257,), dtype=cp.float32)

    ops = [
        ("add", dispatch.add_forward(a, b), a + b),
        ("sub", dispatch.sub_forward(a, b), a - b),
        ("mul", dispatch.mul_forward(a, b), a * b),
        ("div", dispatch.div_forward(a, b), a / b),
        ("neg", dispatch.neg_forward(x), -x),
        ("exp", dispatch.exp_forward(x), cp.exp(x)),
        ("log", dispatch.log_forward(cp.abs(x) + 0.1), cp.log(cp.abs(x) + 0.1)),
        ("pow", dispatch.pow_forward(cp.abs(x), 2.5), cp.power(cp.abs(x), 2.5)),
        ("relu", dispatch.relu_forward(x), cp.maximum(x, 0)),
        ("sigmoid", dispatch.sigmoid_forward(x), cp.tanh(x * 0.5) * 0.5 + 0.5),
    ]
    for name, got, expected in ops:
        report["tests"].append(close(name, got, expected))

    report["tests"].append(close(
        "relu_backward", dispatch.relu_backward(x, a), a * (x > 0),
    ))

    xb = rs.standard_normal((2, 3, 4, 5), dtype=cp.float32)
    bias = cp.asarray([0.2, -0.3, 0.7], dtype=cp.float32)
    report["tests"].append(close("bias_add", dispatch.bias_add_forward(xb, bias), xb + bias[None, :, None, None]))

    xi = rs.standard_normal((2, 2, 5, 6), dtype=cp.float32)
    col = dispatch.im2col_forward(xi, (3, 2), stride=(2, 1), pad=(1, 0), to_matrix=True)
    # 独立 oracle：逐窗口构造同样的零填充矩阵，避免复用被测 im2col 实现。
    rows = []
    for n in range(2):
        for oh in range(3):
            for ow in range(5):
                row = []
                for c in range(2):
                    for kh in range(3):
                        for kw in range(2):
                            ih, iw = oh * 2 + kh - 1, ow + kw
                            row.append(float(xi[n, c, ih, iw].get()) if 0 <= ih < 5 and 0 <= iw < 6 else 0.0)
                rows.append(row)
    report["tests"].append(close("im2col", col, cp.asarray(np.asarray(rows, dtype=np.float32))))

    xc = rs.standard_normal((1, 2, 5, 6), dtype=cp.float32)
    wc = rs.standard_normal((3, 2, 3, 2), dtype=cp.float32)
    bc = rs.standard_normal((3,), dtype=cp.float32)
    y = dispatch.conv2d_forward(xc, wc, bc, stride=(2, 1), pad=(1, 0))
    # 卷积 oracle 调用框架已有的 CuPy/NumPy 基线 im2col，不使用被测 conv2d_forward
    # 或 CUDA C im2col kernel，从而可以发现 kernel 的索引错误。
    from eneuro.base.functions import im2col_array
    ref_col = im2col_array(xc, (3, 2), (2, 1), (1, 0), to_matrix=True, xp=cp)
    # 记录卷积前的 im2col 误差；这样可区分索引 kernel 错误和 GEMM/reshape 错误。
    conv_col = dispatch.im2col_forward(xc, (3, 2), stride=(2, 1), pad=(1, 0), to_matrix=True)
    report["conv_im2col_max_abs_error"] = float(cp.max(cp.abs(conv_col - ref_col)).get())
    # GEMM 结果先是 (N,OH,OW,OC)，必须转回框架使用的 NCHW 布局。
    ref = ref_col.dot(wc.reshape(3, -1).T).reshape(1, 3, 5, 3).transpose(0, 3, 1, 2)
    ref = ref + bc[None, :, None, None]
    report["tests"].append(close("conv2d_forward", y, ref))

    xp = rs.standard_normal((1, 2, 5, 6), dtype=cp.float32)
    pool, indexes = dispatch.maxpool_forward(xp, 2, stride=2, pad=0)
    # 输入为 5x6、kernel=2、stride=2，因此输出空间尺寸是 2x3。
    ref_pool = cp.maximum(cp.maximum(xp[:, :, 0:4:2, 0:6:2], xp[:, :, 1:5:2, 0:6:2]),
                          cp.maximum(xp[:, :, 0:4:2, 1:6:2], xp[:, :, 1:5:2, 1:6:2]))
    report["tests"].append(close("maxpool_forward", pool, ref_pool))
    report["pool_index_dtype"] = str(indexes.dtype)

    report["diagnostics"] = dispatch.diagnostics()
    report["status"] = "passed" if all(t["passed"] for t in report["tests"]) else "failed"
    return finish(report, args.out, 0 if report["status"] == "passed" else 1)


def finish(report, output, code):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
