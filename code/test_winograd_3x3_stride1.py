import sys
import time
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eneuro.base import Tensor, as_Tensor  # noqa: E402
from eneuro.base.functions import conv2d  # noqa: E402


def run_simple_conv_timing(iters=50):
    np.random.seed(20260409)

    # 3x3, stride=1 的标准卷积调用路径
    w = Tensor(np.random.randn(32, 16, 3, 3).astype(np.float32))
    b = Tensor(np.random.randn(32).astype(np.float32))

    x = as_Tensor(np.random.randn(8, 16, 128, 128).astype(np.float32))

    # 预热
    for _ in range(5):
        _ = conv2d(x, w, b, stride=(1, 1), pad=(1, 1))

    t0 = time.perf_counter()
    for _ in range(iters):
        y = conv2d(x, w, b, stride=(1, 1), pad=(1, 1))
    t1 = time.perf_counter()

    total_sec = t1 - t0
    avg_ms = total_sec * 1000.0 / iters

    print("====== 3x3 stride=1 卷积运行时间 ======")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Iterations: {iters}")
    print(f"Total time: {total_sec:.6f} s")
    print(f"Avg time : {avg_ms:.6f} ms/iter")


if __name__ == "__main__":
    run_simple_conv_timing(iters=50)
