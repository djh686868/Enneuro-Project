"""基础 AutoML MVP 使用的 CPU 执行器。

这些执行器刻意只使用 NumPy：即使机器没有 CuPy，也能先验证调参 Agent
的完整流程；同时不会修改项目现有的 CUDA dispatch。真正的 CUDA 训练由
车辆执行器等其他适配层按需接入。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np

from ..profiling import current_profile
from .protocol import TrialResult


class SyntheticObjectiveExecutor:
    """带有已知最优点的确定性合成目标。

    该执行器不训练神经网络，而是根据学习率、batch size 和 optimizer
    与目标配置的距离构造损失。它的用途是快速测试：检查 seed 是否可复现、
    事件顺序是否完整、失败处理是否正确，而不是评估模型性能。
    """

    def __init__(self, target_lr: float = 1e-2, target_batch_size: int = 32,
                 target_optimizer: str = "adam", noise: float = 0.0, seed: int = 0):
        self.target_lr = float(target_lr)
        self.target_batch_size = int(target_batch_size)
        self.target_optimizer = str(target_optimizer)
        self.noise = float(noise)
        self.seed = int(seed)
        self.calls = 0

    def execute(self, config: Mapping[str, Any]) -> TrialResult:
        """计算一个配置的标量损失，并返回可记录的 TrialResult。"""
        lr = float(config["lr"])
        batch = int(config["batch_size"])
        optimizer = str(config["optimizer"])
        value = (np.log10(lr) - np.log10(self.target_lr)) ** 2
        value += 0.12 * abs(np.log2(batch / self.target_batch_size))
        value += 0.25 if optimizer != self.target_optimizer else 0.0
        if self.noise:
            rng = np.random.default_rng(self.seed + self.calls)
            value += float(rng.normal(0.0, self.noise))
        self.calls += 1
        return TrialResult(float(value), {"loss": float(value)})


def make_tiny_mnist(seed: int = 0, samples: int = 200, features: int = 28 * 28,
                    classes: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """在缺少真实文件时生成 MNIST 形状的可分数据。

    每个类别拥有一个随机原型，样本是在原型附近加入噪声得到的；这使
    线性 softmax 模型可以在很短时间内产生有意义的准确率变化。
    """

    rng = np.random.default_rng(seed)
    prototypes = rng.normal(0, 0.5, (classes, features)).astype(np.float32)
    labels = np.arange(samples, dtype=np.int64) % classes
    x = prototypes[labels] + rng.normal(0, 0.6, (samples, features)).astype(np.float32)
    return x, labels


class MNISTExecutor:
    """对真实 MNIST 数组或生成 fixture 执行小型 softmax 训练。

    输入既可以是 ``(N, 784)``，也可以是图像形状；初始化时会统一展平。
    内层默认使用单次留出，未传入验证集时从训练输入中切出后 20%。
    交叉验证将这两部分合并为开发集，使用固定的划分种子生成验证折。
    外层测试数据只交给 ``evaluate_final``，由它在配置确定后重训和评分。
    """

    def __init__(self, x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                 x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 *, epochs: int = 2, seed: int = 0, max_samples: int | None = 256,
                 num_classes: int = 10, x_test: np.ndarray | None = None,
                 y_test: np.ndarray | None = None, inner_folds: int = 1, split_seed: int = 0):
        if inner_folds < 1:
            raise ValueError("inner_folds must be positive")
        if (x_test is None) != (y_test is None):
            raise ValueError("x_test and y_test must be provided together")
        # 没有真实数据时退回到确定性的合成 fixture，保证演示仍可运行。
        if x_train is None or y_train is None:
            x_train, y_train = make_tiny_mnist(seed=seed, samples=max_samples or 256)
        x_train = self._features(x_train)
        y_train = np.asarray(y_train, dtype=np.int64)
        # 小规模 trial 可限制训练样本数，控制每次评估成本。
        if max_samples is not None:
            x_train, y_train = x_train[:max_samples], y_train[:max_samples]
        # 保留直接使用执行器时的默认划分。演示脚本会传入预先固定的验证集。
        if x_val is None or y_val is None:
            split = max(1, int(0.8 * len(x_train)))
            x_val, y_val = x_train[split:], y_train[split:]
            x_train, y_train = x_train[:split], y_train[:split]
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = self._features(x_val), np.asarray(y_val, dtype=np.int64)
        self._x_test = x_test
        self._y_test = y_test
        self._x_dev = np.concatenate((self.x_train, self.x_val))
        self._y_dev = np.concatenate((self.y_train, self.y_val))
        if inner_folds > len(self._x_dev):
            raise ValueError("inner_folds cannot exceed the development sample count")

        # 固定折只依赖 split_seed。换训练种子或比较两个配置时，各折仍完全一致。
        self._folds = ()
        if inner_folds > 1:
            indices = np.random.default_rng(split_seed).permutation(len(self._x_dev))
            self._folds = tuple(np.array_split(indices, inner_folds))
        self.epochs, self.seed = int(epochs), int(seed)
        self.num_classes = int(num_classes)
        self.calls = 0

    @staticmethod
    def _features(x: np.ndarray) -> np.ndarray:
        """将图像样本展平为线性分类器需要的二维矩阵。"""
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(len(x), -1) if x.ndim > 2 else x

    def execute(self, config: Mapping[str, Any]) -> TrialResult:
        """仅使用开发集进行内层评估，目标为 1 - 验证准确率。"""
        optimizer = str(config["optimizer"]).lower()
        if optimizer not in {"sgd", "adam"}:
            return TrialResult(float("inf"), status="failed", error=f"unknown optimizer: {optimizer}")
        if self._folds:
            accuracies = []
            for i, val_idx in enumerate(self._folds):
                train_idx = np.concatenate(self._folds[:i] + self._folds[i + 1:])
                accuracies.append(self._fit_and_score(
                    config, self._x_dev[train_idx], self._y_dev[train_idx],
                    self._x_dev[val_idx], self._y_dev[val_idx],
                ))
        else:
            accuracies = [self._fit_and_score(
                config, self.x_train, self.y_train, self.x_val, self.y_val,
            )]
        accuracy = float(np.mean(accuracies))
        objective = 1.0 - accuracy
        self.calls += 1
        metrics = {"accuracy": accuracy, "error_rate": objective}
        if self._folds:
            metrics["fold_accuracies"] = accuracies
        return TrialResult(objective, metrics)

    def evaluate_final(self, config: Mapping[str, Any]) -> dict[str, float]:
        """在完整开发集上重训选出的配置，再用外层测试集评估一次。"""
        if self._x_test is None or self._y_test is None or len(self._y_test) == 0:
            raise ValueError("final evaluation requires a non-empty outer test set")
        accuracy = self._fit_and_score(
            config, self._x_dev, self._y_dev,
            self._features(self._x_test), np.asarray(self._y_test, dtype=np.int64),
            evaluation_phase="automl.test",
        )
        return {"test_accuracy": accuracy, "test_error_rate": 1.0 - accuracy}

    def _fit_and_score(self, config: Mapping[str, Any], x_train: np.ndarray, y_train: np.ndarray,
                       x_eval: np.ndarray, y_eval: np.ndarray, *, evaluation_phase="validation") -> float:
        """内层和外层共用训练配方，配置只控制优化器、学习率和批大小。"""
        lr = float(config["lr"])
        batch_size = int(config["batch_size"])
        optimizer = str(config["optimizer"]).lower()
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(f"unknown optimizer: {optimizer}")
        # 同一运行的所有候选和基线从同一个种子开始，公平比较不同配置。
        rng = np.random.default_rng(self.seed)
        n_features = x_train.shape[1]
        weights = rng.normal(0, 0.01, (n_features, self.num_classes)).astype(np.float32)
        bias = np.zeros(self.num_classes, dtype=np.float32)
        m_w = np.zeros_like(weights); v_w = np.zeros_like(weights)
        m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
        step = 0
        profile = current_profile()
        # 对同一个训练子集，所有配置在每个 epoch 使用相同的样本排列。
        for _ in range(max(1, self.epochs)):
            with profile.span("train") if profile is not None else nullcontext():
                order = rng.permutation(len(x_train))
                for start in range(0, len(order), batch_size):
                    idx = order[start:start + batch_size]
                    x, y = x_train[idx], y_train[idx]
                    logits = x @ weights + bias
                    logits -= logits.max(axis=1, keepdims=True)
                    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
                    probs[np.arange(len(y)), y] -= 1.0
                    grad_w = x.T @ probs / len(y)
                    grad_b = probs.mean(axis=0)
                    if optimizer == "adam":
                        # Adam 的一阶/二阶动量；SGD 则直接使用当前梯度。
                        step += 1; beta1, beta2 = 0.9, 0.999
                        m_w = beta1 * m_w + (1 - beta1) * grad_w; v_w = beta2 * v_w + (1 - beta2) * grad_w**2
                        m_b = beta1 * m_b + (1 - beta1) * grad_b; v_b = beta2 * v_b + (1 - beta2) * grad_b**2
                        cw = m_w / (1 - beta1**step) / (np.sqrt(v_w / (1 - beta2**step)) + 1e-8)
                        cb = m_b / (1 - beta1**step) / (np.sqrt(v_b / (1 - beta2**step)) + 1e-8)
                    else:
                        cw, cb = grad_w, grad_b
                    weights -= lr * cw; bias -= lr * cb
            if profile is not None:
                profile.count("train_samples", len(x_train))
        # 评分样本从未进入这个模型的参数更新。调用方负责选择内层或外层数据。
        with profile.span(evaluation_phase) if profile is not None else nullcontext():
            logits = x_eval @ weights + bias
            if not np.isfinite(logits).all():
                raise ValueError("MNIST scores must be finite")
            pred = logits.argmax(axis=1)
            accuracy = float(np.mean(pred == y_eval)) if len(y_eval) else 0.0
        if profile is not None and evaluation_phase == "validation":
            profile.count("validation_samples", len(y_eval))
        return accuracy

    # 对外提供 fit 别名，方便将执行器接入习惯“fit”命名的调用方。
    fit = execute
