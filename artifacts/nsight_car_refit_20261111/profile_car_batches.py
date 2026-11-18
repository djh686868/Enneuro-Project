"""Capture a short, isolated copy of the Car refit workload in Nsight Systems."""

import json
import sys
import time
from pathlib import Path

import cupy as cp
from cupy.cuda import profiler
from cupyx.profiler import time_range

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "code"))
from eneuro.agent import CarBehavioralCloningExecutor
from eneuro.base import Tensor
from eneuro.train.trainer import Trainer

source = json.loads((root / "artifacts/car_beyesian_full_20261110_005640/report.json")
                    .read_text(encoding="utf-8"))
protocol = source["protocol"]
executor = CarBehavioralCloningExecutor(
    protocol["data"], refit_epochs=1, seed=source["seed"], device="cuda",
    backend=protocol["backend"], device_id=protocol["device_id"], model="resnet18",
    image_size=tuple(protocol["image_size"]), cameras=tuple(protocol["cameras"]),
    camera_correction=protocol["camera_correction"], max_samples=protocol["max_samples"],
)
train_epoch = Trainer._one_step
warmup_batches, profiled_batches = 16, 24


class CaptureFinished(Exception):
    pass


def capture(trainer, loader, **kwargs):
    trainer.model.forward = time_range("Model forward")(trainer.model.forward)
    trainer.loss_fn = time_range("Loss")(trainer.loss_fn)
    trainer.optimizer.step = time_range("Adam step")(trainer.optimizer.step)
    Tensor.backward = time_range("Autograd backward")(Tensor.backward)
    stream = cp.cuda.get_current_stream()
    elapsed = None

    def batches():
        nonlocal elapsed
        iterator = iter(loader)
        for index in range(warmup_batches + profiled_batches):
            if index == warmup_batches:
                stream.synchronize()
                profiler.start()
                started = time.perf_counter()
            with time_range("Training batch"):
                with time_range("JPEG decode and batch"):
                    batch = next(iterator)
                yield batch
        stream.synchronize()
        elapsed = time.perf_counter() - started
        profiler.stop()

    train_epoch(trainer, batches(), **kwargs)
    print(f"Captured {profiled_batches} batches after {warmup_batches} warm-up batches "
          f"in {elapsed:.3f}s", flush=True)
    raise CaptureFinished


Trainer._one_step = capture
try:
    executor.evaluate_final(source["best_config"])
except CaptureFinished:
    pass
