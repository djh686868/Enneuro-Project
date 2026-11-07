"""CPU executors used by the AutoML MVP.

They intentionally use NumPy only.  This keeps the agent useful on machines
without CuPy and avoids changing the project's CUDA dispatch layer.
"""

from __future__ import annotations

import json
import zlib
from typing import Any, Mapping

import numpy as np

from .protocol import TrialResult


class SyntheticObjectiveExecutor:
    """A deterministic objective with a known optimum for smoke tests."""

    def __init__(self, target_lr: float = 1e-2, target_batch_size: int = 32,
                 target_optimizer: str = "adam", noise: float = 0.0, seed: int = 0):
        self.target_lr = float(target_lr)
        self.target_batch_size = int(target_batch_size)
        self.target_optimizer = str(target_optimizer)
        self.noise = float(noise)
        self.seed = int(seed)
        self.calls = 0

    def execute(self, config: Mapping[str, Any]) -> TrialResult:
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
    """Create MNIST-shaped data when a downloaded MNIST file is unavailable."""

    rng = np.random.default_rng(seed)
    prototypes = rng.normal(0, 0.5, (classes, features)).astype(np.float32)
    labels = np.arange(samples, dtype=np.int64) % classes
    x = prototypes[labels] + rng.normal(0, 0.6, (samples, features)).astype(np.float32)
    return x, labels


class MNISTExecutor:
    """Tiny softmax classifier executor for MNIST arrays or generated fixtures.

    Inputs can be flattened or image-shaped NumPy arrays.  A validation split is
    made deterministically if one is not supplied.  The implementation is
    intentionally small but performs real minibatch optimization for both SGD
    and Adam.
    """

    def __init__(self, x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                 x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                 *, epochs: int = 2, seed: int = 0, max_samples: int | None = 256,
                 num_classes: int = 10):
        if x_train is None or y_train is None:
            x_train, y_train = make_tiny_mnist(seed=seed, samples=max_samples or 256)
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        if x_train.ndim > 2:
            x_train = x_train.reshape(len(x_train), -1)
        if max_samples is not None:
            x_train, y_train = x_train[:max_samples], y_train[:max_samples]
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
        lr = float(config["lr"])
        batch_size = int(config["batch_size"])
        optimizer = str(config["optimizer"]).lower()
        if optimizer not in {"sgd", "adam"}:
            return TrialResult(float("inf"), status="failed", error=f"unknown optimizer: {optimizer}")
        # Stable per-configuration randomness makes order changes harmless.
        key = json.dumps(dict(config), sort_keys=True).encode("utf-8")
        rng = np.random.default_rng(self.seed + zlib.crc32(key) % 1000003)
        n_features = self.x_train.shape[1]
        weights = rng.normal(0, 0.01, (n_features, self.num_classes)).astype(np.float32)
        bias = np.zeros(self.num_classes, dtype=np.float32)
        m_w = np.zeros_like(weights); v_w = np.zeros_like(weights)
        m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
        step = 0
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
                    step += 1; beta1, beta2 = 0.9, 0.999
                    m_w = beta1 * m_w + (1 - beta1) * grad_w; v_w = beta2 * v_w + (1 - beta2) * grad_w**2
                    m_b = beta1 * m_b + (1 - beta1) * grad_b; v_b = beta2 * v_b + (1 - beta2) * grad_b**2
                    cw = m_w / (1 - beta1**step) / (np.sqrt(v_w / (1 - beta2**step)) + 1e-8)
                    cb = m_b / (1 - beta1**step) / (np.sqrt(v_b / (1 - beta2**step)) + 1e-8)
                else:
                    cw, cb = grad_w, grad_b
                weights -= lr * cw; bias -= lr * cb
        val_logits = self.x_val @ weights + bias
        pred = val_logits.argmax(axis=1)
        accuracy = float(np.mean(pred == self.y_val)) if len(self.y_val) else 0.0
        objective = 1.0 - accuracy
        self.calls += 1
        return TrialResult(objective, {"accuracy": accuracy, "error_rate": objective})

    fit = execute
