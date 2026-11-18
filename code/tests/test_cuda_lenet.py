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
from eneuro.base.cuda import dispatch as cuda


@pytest.mark.cuda
def test_lenet_forward_backward_smoke():
    if not cuda.is_available("rawmodule"):
        pytest.skip("CUDA/NVRTC unavailable")
    previous_backend = cuda.get_backend()
    cuda.set_backend("rawmodule")
    cuda.launch_counts(reset=True)
    try:
        model = LeNet(in_channels=1, num_classes=10).to("cuda")
        x = Tensor(cp.zeros((2, 1, 28, 28), dtype=cp.float32), device="cuda")
        t = Tensor(cp.asarray([0, 1], dtype=cp.int32), device="cuda")
        y = model(x)
        assert y.shape == (2, 10)
        loss = CrossEntropyLoss()(y, t)
        loss.backward()
        assert np.isfinite(float(cp.asnumpy(loss.data)))
        assert any(p.grad is not None for p in model.params())
        counts = cuda.launch_counts()
        assert counts.get("conv_bwd_x_f32", 0) > 0
        assert counts.get("pool_bwd_f32", 0) > 0
    finally:
        cuda.set_backend(previous_backend)
