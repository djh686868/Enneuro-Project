"""Minimal LeNet fast-route smoke test; full comparison is run by the benchmark CLI."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cp = pytest.importorskip("cupy")

from eneuro.base import Tensor
from eneuro.nn.module import LeNet
from eneuro.nn.loss import CrossEntropyLoss


@pytest.mark.cuda
def test_lenet_forward_backward_smoke():
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")
    model = LeNet(in_channels=1, num_classes=10).to("cuda")
    x = Tensor(cp.zeros((2, 1, 28, 28), dtype=cp.float32), device="cuda")
    t = Tensor(cp.asarray([0, 1], dtype=cp.int32), device="cuda")
    y = model(x)
    assert y.shape == (2, 10)
    loss = CrossEntropyLoss()(y, t)
    loss.backward()
    assert np.isfinite(float(cp.asnumpy(loss.data)))
    assert any(p.grad is not None for p in model.params())
