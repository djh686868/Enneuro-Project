"""基础 AutoML MVP 使用的 CPU 执行器。

这些执行器刻意只使用 NumPy：即使机器没有 CuPy，也能先验证调参 Agent
的完整流程；同时不会修改项目现有的 CUDA dispatch。真正的 CUDA 训练由
车辆执行器等其他适配层按需接入。
"""

from __future__ import annotations

import json
import zlib
from typing import Any, Mapping

import numpy as np

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
    如果调用方没有提供验证集，则按固定比例划分。内部实现了真实的
    minibatch SGD/Adam 更新，目标为 ``1 - validation_accuracy``，因此能够
    直接接入当前“目标越小越好”的搜索器。
    """

    def __init__(self, x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                 x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 *, epochs: int = 2, seed: int = 0, max_samples: int | None = 256,
                 num_classes: int = 10):
        # 没有真实数据时退回到确定性的合成 fixture，保证演示仍可运行。
        if x_train is None or y_train is None:
            x_train, y_train = make_tiny_mnist(seed=seed, samples=max_samples or 256)
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        # 线性分类器只接受二维矩阵，因此将图像样本展平。
        if x_train.ndim > 2:
            x_train = x_train.reshape(len(x_train), -1)
        # 小规模 trial 可限制训练样本数，控制每次评估成本。
        if max_samples is not None:
            x_train, y_train = x_train[:max_samples], y_train[:max_samples]
        # 未显式提供验证集时，按时间稳定的前后切分训练/验证数据。
        if x_val is None or y_val is None:
            split = max(1, int(0.8 * len(x_train)))
            x_val, y_val = x_train[split:], y_train[split:]
            x_train, y_train = x_train[:split], y_train[:split]
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = np.asarray(x_val, dtype=np.float32), np.asarray(y_val, dtype=np.int64)
        if self.x_val.ndim > 2:
            self.x_val = self.x_val.reshape(len(self.x_val), -1)
        self.epochs, self.seed = int(epochs), int(seed)
        self.num_classes = int(num_classes)
        self.calls = 0

    def execute(self, config: Mapping[str, Any]) -> TrialResult:
        """用给定超参数完成一次小型训练，并返回验证错误率。"""
        lr = float(config["lr"])
        batch_size = int(config["batch_size"])
        optimizer = str(config["optimizer"]).lower()
        if optimizer not in {"sgd", "adam"}:
            return TrialResult(float("inf"), status="failed", error=f"unknown optimizer: {optimizer}")
        # 将配置编码进随机种子，使同一配置不依赖 trial 执行顺序。
        key = json.dumps(dict(config), sort_keys=True).encode("utf-8")
        rng = np.random.default_rng(self.seed + zlib.crc32(key) % 1000003)
        n_features = self.x_train.shape[1]
        weights = rng.normal(0, 0.01, (n_features, self.num_classes)).astype(np.float32)
        bias = np.zeros(self.num_classes, dtype=np.float32)
        m_w = np.zeros_like(weights); v_w = np.zeros_like(weights)
        m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
        step = 0
        # 每个 epoch 重新打乱样本，但仍由配置专属 RNG 控制，保证复现。
        for _ in range(max(1, self.epochs)):
            order = rng.permutation(len(self.x_train))
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                x, y = self.x_train[idx], self.y_train[idx]
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
        # 训练结束后只在验证集上计算 objective，避免把测试集用于决策。
        val_logits = self.x_val @ weights + bias
        pred = val_logits.argmax(axis=1)
        accuracy = float(np.mean(pred == self.y_val)) if len(self.y_val) else 0.0
        objective = 1.0 - accuracy
        self.calls += 1
        return TrialResult(objective, {"accuracy": accuracy, "error_rate": objective})

    # 对外提供 fit 别名，方便将执行器接入习惯“fit”命名的调用方。
    fit = execute
