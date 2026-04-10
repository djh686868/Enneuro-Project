# -*- coding: utf-8 -*-
"""
����������ɲ��ԣ�
- ��׼��������
- �����������
- ��Ⱦ�������

�����
1) ÿ��ģ�͵� loss/acc ����
2) ��������ѵ��ʱ��Ա�
3) ���Ż�Ч�������ԣ��Զ�����ԭ�����
"""

import csv
import gzip
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 添加code目录到Python搜索路径，这样就能找到eneuro模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from eneuro.base import Config, as_Tensor  # noqa: E402
from eneuro.base import functions as F  # noqa: E402
from eneuro.nn.loss import crossEntropyError  # noqa: E402
from eneuro.nn.module import Module, Conv2d, Linear  # noqa: E402
from eneuro.nn.optim import SGD  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "tests" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def find_mnist_pkl() -> Path:
    candidates = [
        PROJECT_ROOT / "tests" / "testdata" / "MNIST_data" / "mnist.pkl",
        PROJECT_ROOT / "code" / "testdata" / "MNIST_data" / "mnist.pkl",
        PROJECT_ROOT / "testdata" / "MNIST_data" / "mnist.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("MNIST pickle file not found")


def load_mnist_from_pkl(pkl_path: Path):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        with gzip.open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

    if isinstance(data, tuple) and len(data) == 2:
        (x_train, y_train), (x_test, y_test) = data
    elif isinstance(data, tuple) and len(data) == 3:
        (x_train, y_train), _, (x_test, y_test) = data
    elif isinstance(data, dict):
        x_train = data.get("train_img", data.get("x_train"))
        y_train = data.get("train_label", data.get("y_train"))
        x_test = data.get("test_img", data.get("x_test"))
        y_test = data.get("test_label", data.get("y_test"))
    else:
        raise ValueError(f"Unknown MNIST data format: {type(data)}")

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if x_train.ndim == 2 and x_train.shape[1] == 784:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)
    elif x_train.ndim == 3:
        x_train = x_train[:, None, :, :]
        x_test = x_test[:, None, :, :]

    return x_train, y_train, x_test, y_test


class NumpyDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            bidx = idx[i:i + self.batch_size]
            yield self.dataset.x[bidx], self.dataset.y[bidx]


class StandardConvNet(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(8, kernel_size=3, stride=1, pad=1, in_channels=1)
        self.conv2 = Conv2d(16, kernel_size=3, stride=1, pad=1, in_channels=8)
        self.fc = Linear(num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.flatten(x)
        x = self.fc(x)
        return x


class GroupConvNet(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(8, kernel_size=3, stride=1, pad=1, in_channels=1)
        self.gconv2 = Conv2d(16, kernel_size=3, stride=1, pad=1, in_channels=8, groups=2)
        self.fc = Linear(num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.relu(self.gconv2(x))
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.flatten(x)
        x = self.fc(x)
        return x


class DepthwiseConvNet(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dw1 = Conv2d(1, kernel_size=3, stride=1, pad=1, in_channels=1, depthwise=True)
        self.pw1 = Conv2d(8, kernel_size=1, stride=1, pad=0, in_channels=1)

        self.dw2 = Conv2d(8, kernel_size=3, stride=1, pad=1, in_channels=8, depthwise=True)
        self.pw2 = Conv2d(16, kernel_size=1, stride=1, pad=0, in_channels=8)

        self.fc = Linear(num_classes)

    def forward(self, x):
        x = F.relu(self.pw1(self.dw1(x)))
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.relu(self.pw2(self.dw2(x)))
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.flatten(x)
        x = self.fc(x)
        return x


def evaluate(model, x, y, batch_size=128):
    total = 0
    correct = 0
    loss_sum = 0.0
    with Config.using_config("train", False):
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            logits = model(as_Tensor(xb))
            loss = crossEntropyError(logits, yb)
            pred = logits.data.argmax(axis=1)
            correct += int((pred == yb).sum())
            total += len(yb)
            loss_sum += float(loss.data) * len(yb)
    return loss_sum / total, correct / total


def count_params(model):
    total = 0
    for p in model.params():
        if getattr(p, "data", None) is not None:
            total += int(np.prod(p.data.shape))
    return total


def grad_coverage(model, x_batch, y_batch):
    with Config.using_config("train", True):
        logits = model(as_Tensor(x_batch))
        loss = crossEntropyError(logits, y_batch)
        model.cleargrads()
        loss.backward()

    with_grad = 0
    total = 0
    for p in model.params():
        if getattr(p, "data", None) is None:
            continue
        total += 1
        if p.grad is not None:
            with_grad += 1

    if total == 0:
        return 0.0
    return with_grad / total


def train_model(model, train_loader, x_test, y_test, epochs=3, lr=1e-3):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    optimizer = SGD(model.params(), lr=lr)

    init_loss, init_acc = evaluate(model, x_test, y_test)
    history["val_loss"].append(init_loss)
    history["val_acc"].append(init_acc)

    for _ in range(epochs):
        t0 = time.time()
        batch_losses = []

        with Config.using_config("train", True):
            for xb, yb in train_loader:
                logits = model(as_Tensor(xb))
                loss = crossEntropyError(logits, yb)
                model.cleargrads()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.data))

        ep_time = time.time() - t0
        val_loss, val_acc = evaluate(model, x_test, y_test)

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(ep_time)

    return history


def save_plot(results, out_png):
    fig = plt.figure(figsize=(13, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    for name, item in results.items():
        ax1.plot(range(len(item["history"]["val_acc"])), item["history"]["val_acc"], marker="o", label=name)
    ax1.set_title("Validation Accuracy")
    ax1.set_xlabel("Epoch (0 is init)")
    ax1.set_ylabel("Acc")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    for name, item in results.items():
        ax2.plot(range(len(item["history"]["val_loss"])), item["history"]["val_loss"], marker="o", label=name)
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch (0 is init)")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    names = list(results.keys())
    params = [results[n]["params"] for n in names]
    ax3.bar(names, params)
    ax3.set_title("Parameter Count")
    ax3.set_ylabel("#Params")

    ax4 = fig.add_subplot(2, 2, 4)
    times = [float(np.mean(results[n]["history"]["epoch_time"])) for n in names]
    ax4.bar(names, times)
    ax4.set_title("Average Epoch Time (s)")
    ax4.set_ylabel("Seconds")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def save_csv(results, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "val_loss", "val_acc", "train_loss", "epoch_time_s"])
        for name, item in results.items():
            h = item["history"]
            max_epoch = len(h["val_loss"])  # epoch=0 is init
            for ep in range(max_epoch):
                train_loss = ""
                epoch_time = ""
                if ep > 0:
                    train_loss = h["train_loss"][ep - 1]
                    epoch_time = h["epoch_time"][ep - 1]
                writer.writerow([name, ep, h["val_loss"][ep], h["val_acc"][ep], train_loss, epoch_time])


def analyze_results(results):
    base = results["standard"]
    base_gain = base["history"]["val_acc"][-1] - base["history"]["val_acc"][0]

    print("\n========== Special Convolution Integration Summary ==========")
    for name, item in results.items():
        acc0 = item["history"]["val_acc"][0]
        accf = item["history"]["val_acc"][-1]
        gain = accf - acc0
        print(
            f"[{name}] init_acc={acc0:.4f}, final_acc={accf:.4f}, "
            f"acc_gain={gain:+.4f}, params={item['params']}, grad_coverage={item['grad_coverage']:.2f}"
        )

    print("\n========== Why Optimization Gains May Be Limited ==========")
    for name, item in results.items():
        if name == "standard":
            continue
        gain = item["history"]["val_acc"][-1] - item["history"]["val_acc"][0]
        reasons = []

        if gain < max(0.03, base_gain * 0.6):
            reasons.append("fewer parameters than baseline, but no accuracy gain in this setup")
        if item["grad_coverage"] < 0.9:
            reasons.append("gradient coverage is low, so some paths may not be trained effectively")
        if item["params"] < base["params"] and item["history"]["val_acc"][-1] < base["history"]["val_acc"][-1]:
            reasons.append("fewer params but lower accuracy; the task may be too small to show the advantage")

        if not reasons:
            reasons.append("no clear improvement under the current experimental setup")

        print(f"- {name}: " + "; ".join(reasons))


def main():
    np.random.seed(2026)

    pkl_path = find_mnist_pkl()
    x_train, y_train, x_test, y_test = load_mnist_from_pkl(pkl_path)

    # ����ʱ�����Ӽ�ʵ��
    x_train, y_train = x_train[:1500], y_train[:1500]
    x_test, y_test = x_test[:400], y_test[:400]

    train_loader = SimpleDataLoader(NumpyDataset(x_train, y_train), batch_size=64, shuffle=True)

    model_factories = {
        "standard": StandardConvNet,
        "group": GroupConvNet,
        "depthwise": DepthwiseConvNet,
    }

    results = {}

    for name, factory in model_factories.items():
        print(f"\n[RUN] training {name} model...")
        model = factory(num_classes=10)

        # gradient coverage check before training
        gc = grad_coverage(model, x_train[:64], y_train[:64])

        history = train_model(
            model,
            train_loader,
            x_test,
            y_test,
            epochs=3,
            lr=1e-3,
        )

        results[name] = {
            "history": history,
            "params": count_params(model),
            "grad_coverage": gc,
        }

    out_png = ARTIFACT_DIR / "special_convs_compare.png"
    out_csv = ARTIFACT_DIR / "special_convs_compare.csv"

    save_plot(results, out_png)
    save_csv(results, out_csv)
    analyze_results(results)

    print(f"\n[ARTIFACT] plot saved: {out_png}")
    print(f"[ARTIFACT] csv saved: {out_csv}")


if __name__ == "__main__":
    main()
