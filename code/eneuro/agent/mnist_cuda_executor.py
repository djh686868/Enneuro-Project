"""用 EnNeuro LeNet 在 CUDA 上执行 MNIST 搜索和独立外层评估。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..profiling import current_profile
from ..profiling.cuda import cuda_device_name
from .protocol import TrialResult


class _SearchDeadline(TimeoutError):
    """训练或验证已到达本次搜索的截止时间。"""


class _NonFiniteTrial(ValueError):
    """候选配置的训练损失或预测包含非有限值。"""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise _SearchDeadline("CUDA search budget expired before this trial completed")


class CudaMNISTExecutor:
    """接收已归一化的 NCHW 图片，使用固定划分和 LeNet 训练配方。"""

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, *,
                 epochs: int, seed: int, device_id: int = 0, backend: str = "cupy"):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CUDA MNIST requires CuPy in the current Python environment") from exc

        if epochs < 1:
            raise ValueError("epochs must be positive")
        if backend not in {"cupy", "rawmodule"}:
            raise ValueError("CUDA backend must be cupy or rawmodule")
        count = cp.cuda.runtime.getDeviceCount()
        if not 0 <= device_id < count:
            raise RuntimeError(f"CUDA device {device_id} is unavailable (visible device count: {count})")

        self._cp = cp
        self.device_id = device_id
        self.backend = backend
        self.epochs = epochs
        self.seed = seed
        self.device_name = cuda_device_name(device_id)
        for inputs, labels in ((x_train, y_train), (x_val, y_val), (x_test, y_test)):
            if inputs.shape[1:] != (1, 28, 28) or labels.shape != (len(inputs),) or not len(inputs):
                raise ValueError("MNIST splits must be non-empty NCHW (N, 1, 28, 28) with N labels")
        with cp.cuda.Device(device_id):
            self._train = (cp.asarray(x_train, dtype=cp.float32), cp.asarray(y_train, dtype=cp.int32))
            self._validation = (cp.asarray(x_val, dtype=cp.float32), cp.asarray(y_val, dtype=cp.int32))
        # 测试数据到外层评分阶段才复制到显存。
        self._test = (x_test, y_test)

    def execute(self, config: Mapping[str, Any], *, deadline: float | None = None,
                on_epoch: Callable[[int, float, float], None] | None = None) -> TrialResult:
        """使用验证错误率评分。超时的部分训练不产生可参与选择的分数。"""
        try:
            _, _, losses, accuracies = self._fit(
                config, self._train, validation=self._validation, deadline=deadline, on_epoch=on_epoch,
            )
        except _SearchDeadline as exc:
            return TrialResult(float("inf"), status="timed_out", error=str(exc))
        except _NonFiniteTrial as exc:
            return TrialResult(float("inf"), status="failed", error=str(exc))

        accuracy = accuracies[-1]
        return TrialResult(1.0 - accuracy, {
            "accuracy": accuracy, "error_rate": 1.0 - accuracy,
            "train_loss": losses[-1], "epoch_train_loss": losses,
            "epoch_val_accuracy": accuracies,
        })

    def evaluate_final(self, config: Mapping[str, Any], *,
                       on_epoch: Callable[[int, float, float | None], None] | None = None,
                       checkpoint: str | Path | None = None) -> dict[str, Any]:
        """固定配置后在整个开发集重训，仅在训练结束后评分测试集。"""
        cp = self._cp
        from eneuro.base.cuda import dispatch

        profile = current_profile()
        previous = dispatch.get_backend()
        try:
            dispatch.set_backend(self.backend)
            with cp.cuda.Device(self.device_id):
                with profile.span("automl.refit_data", cuda=True) if profile is not None else nullcontext():
                    development = tuple(cp.concatenate((a, b)) for a, b in zip(self._train, self._validation))
                model, optimizer, losses, _ = self._fit(config, development, on_epoch=on_epoch)
                images, labels = self._test
                with profile.span("automl.test", cuda=True) if profile is not None else nullcontext():
                    test = (cp.asarray(images, dtype=cp.float32), cp.asarray(labels, dtype=cp.int32))
                    accuracy = self._accuracy(model, test, deadline=None)
                if checkpoint is not None:
                    from eneuro.utils.serializer import Serializer

                    with profile.span("automl.checkpoint") if profile is not None else nullcontext():
                        Serializer.save_checkpoint_binary(model, optimizer, self.epochs, checkpoint)
        finally:
            dispatch.set_backend(previous)
        return {
            "test_accuracy": accuracy, "test_error_rate": 1.0 - accuracy,
            "train_loss": losses[-1], "epoch_train_loss": losses,
            "train_samples": len(development[0]), "test_samples": len(images),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        }

    def _fit(self, config, data, *, validation=None, deadline=None, on_epoch=None):
        cp = self._cp
        from eneuro.base import Config, Tensor
        from eneuro.base.cuda import dispatch
        from eneuro.nn.loss import CrossEntropyLoss
        from eneuro.nn.module import LeNet
        from eneuro.nn.optim import Adam, SGD

        optimizer_class = {"adam": Adam, "sgd": SGD}.get(str(config["optimizer"]).lower())
        batch_size = int(config["batch_size"])
        lr = float(config["lr"])
        if optimizer_class is None or batch_size < 1 or not isfinite(lr) or lr <= 0:
            raise ValueError("invalid optimizer, batch size, or learning rate")

        _check_deadline(deadline)
        profile = current_profile()
        batch_every = profile.config.batch_every if profile is not None else 0
        previous = dispatch.get_backend()
        try:
            dispatch.set_backend(self.backend)
            with cp.cuda.Device(self.device_id):
                # 每个 trial 的延迟层使用相同初始权重。前向训练和优化均使用 CUDA。
                state = np.random.get_state()
                try:
                    np.random.seed(self.seed)
                    model = LeNet(1, 10)
                    with Config.no_grad():
                        model(Tensor(np.zeros((1, 1, 28, 28), dtype=np.float32)))
                finally:
                    np.random.set_state(state)
                model.to("cuda")
                params = {}
                model._flatten_params(params)
                if not params or not all(isinstance(param.data, cp.ndarray) for param in params.values()):
                    raise RuntimeError("LeNet parameters failed to move to CUDA")
                # 用固定名称顺序传参，二进制 checkpoint 的 optimizer 状态索引可复现。
                optimizer = optimizer_class([params[name] for name in sorted(params)], lr=lr)
                rng = np.random.default_rng(self.seed)
                images, labels = data
                losses, accuracies = [], []
                for epoch in range(1, self.epochs + 1):
                    _check_deadline(deadline)
                    order = cp.asarray(rng.permutation(len(images)))
                    total_loss = 0.0
                    train_span = profile.span("train", cuda=True) if profile is not None else nullcontext()
                    with train_span, Config.using_config("train", True), Config.using_config("enable_backprop", True):
                        for start in range(0, len(images), batch_size):
                            _check_deadline(deadline)
                            indices = order[start:start + batch_size]
                            xb = Tensor(images[indices], device="cuda")
                            yb = Tensor(labels[indices], device="cuda")
                            model.cleargrads()
                            loss = CrossEntropyLoss()(model(xb), yb)
                            loss.backward()
                            optimizer.step()
                            # 取损失标量会同步本批 GPU 操作，截止时间不会积压多个批次。
                            value = float(loss.data.item())
                            if not isfinite(value):
                                raise _NonFiniteTrial("CUDA MNIST loss is not finite")
                            total_loss += value * len(indices)
                            if batch_every and (start // batch_size) % batch_every == 0:
                                profile.sample_cupy_pool()

                    if profile is not None:
                        profile.count("train_samples", len(images))
                    losses.append(total_loss / len(images))
                    accuracy = None
                    if validation is not None:
                        with profile.span("validation", cuda=True) if profile is not None else nullcontext():
                            accuracy = self._accuracy(model, validation, deadline=deadline)
                        if profile is not None:
                            profile.count("validation_samples", len(validation[0]))
                        accuracies.append(accuracy)
                    _check_deadline(deadline)
                    if on_epoch is not None:
                        on_epoch(epoch, losses[-1], accuracy)
                _check_deadline(deadline)
                return model, optimizer, losses, accuracies
        finally:
            dispatch.set_backend(previous)

    def _accuracy(self, model, data, *, deadline: float | None) -> float:
        cp = self._cp
        from eneuro.base import Config, Tensor

        images, labels = data
        correct = 0
        with Config.no_grad(), Config.test_mode():
            for start in range(0, len(images), 256):
                _check_deadline(deadline)
                batch = images[start:start + 256]
                logits = model(Tensor(batch, device="cuda")).data
                # CUDA 上计算统计量，每批只把一个计数传回主机。非有限输出视为失败。
                count = cp.count_nonzero(logits.argmax(axis=1) == labels[start:start + len(batch)])
                value = int(cp.where(cp.isfinite(logits).all(), count, -1).item())
                if value < 0:
                    raise _NonFiniteTrial("CUDA MNIST logits are not finite")
                correct += value
        _check_deadline(deadline)
        return correct / len(images)
