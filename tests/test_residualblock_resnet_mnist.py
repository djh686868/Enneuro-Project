# -*- coding: gbk -*-
import sys
import gzip
import pickle
from pathlib import Path

import numpy as np

# 让 tests 目录下脚本可直接导入 code/eneuro
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from eneuro.base import as_Tensor, Config  # noqa: E402
from eneuro.base import functions as F      # noqa: E402
from eneuro.nn.loss import crossEntropyError  # noqa: E402
from eneuro.nn.module import Module, Conv2d, BatchNorm, Linear, ResidualBlock  # noqa: E402
from eneuro.nn.optim import Adam  # noqa: E402


def find_mnist_pkl() -> Path:
    candidates = [
        PROJECT_ROOT / "tests" / "testdata" / "MNIST_data" / "mnist.pkl",
        PROJECT_ROOT / "code" / "testdata" / "MNIST_data" / "mnist.pkl",
        PROJECT_ROOT / "testdata" / "MNIST_data" / "mnist.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 mnist.pkl，已尝试路径:\n" + "\n".join(str(p) for p in candidates)
    )


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
        raise ValueError(f"未知的MNIST数据格式: {type(data)}")

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    # 归一化 + 统一为 NCHW
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
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)


class SimpleDataLoader:
    def __init__(self, dataset: NumpyDataset, batch_size=64, shuffle=True):
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


class TinyResNetForMNIST(Module):
    """用于验证 ResidualBlock 的简化 ResNet。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = Conv2d(16, kernel_size=3, stride=1, pad=1)
        self.stem_bn = BatchNorm(16)

        self.block1 = ResidualBlock(16, 16, stride=1, downsample=False)
        self.block2 = ResidualBlock(16, 16, stride=1, downsample=False)

        self.fc = Linear(num_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = F.pooling(x, kernel_size=2, stride=2)
        x = self.block2(x)
        x = F.pooling(x, kernel_size=2, stride=2)
        x = F.global_average_pooling(x)
        x = F.flatten(x)
        x = self.fc(x)
        return x


def evaluate(model: TinyResNetForMNIST, x: np.ndarray, y: np.ndarray, batch_size=128):
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


def test_residualblock_downsample_special():
    """ResidualBlock 下采样专项测试。"""
    block_ds = ResidualBlock(in_channels=16, out_channels=32, stride=2, downsample=True)
    x_ds = as_Tensor(np.random.randn(4, 16, 28, 28).astype(np.float32))
    y_ds = block_ds(x_ds)

    # 1) 下采样后形状应变化为 (N, 32, 14, 14)
    assert y_ds.shape == (4, 32, 14, 14), (
        f"Downsample形状错误: got={y_ds.shape}, expect=(4, 32, 14, 14)"
    )

    # 2) 下采样分支应存在
    assert block_ds.conv3 is not None, "downsample=True 时应创建 shortcut 的 1x1 卷积"

    # 3) 反向传播应覆盖主分支与shortcut分支
    ds_loss = y_ds * y_ds
    block_ds.cleargrads()
    ds_loss.backward()
    assert block_ds.conv1.W.grad is not None, "下采样块主分支 conv1 未获得梯度"
    assert block_ds.conv2.W.grad is not None, "下采样块主分支 conv2 未获得梯度"
    assert block_ds.conv3.W.grad is not None, "下采样块 shortcut 分支 conv3 未获得梯度"

    print("[PASS] ResidualBlock 下采样专项测试通过")


def main():
    np.random.seed(42)

    # 1) 加载数据（使用子集，保证测试时间可控）
    pkl_path = find_mnist_pkl()
    print(f"[INFO] 使用数据文件: {pkl_path}")

    x_train, y_train, x_test, y_test = load_mnist_from_pkl(pkl_path)
    x_train, y_train = x_train[:2000], y_train[:2000]
    x_test, y_test = x_test[:500], y_test[:500]

    # 2) 基础形状与反向传播测试（直接针对ResidualBlock）
    block = ResidualBlock(in_channels=16, out_channels=16, stride=1, downsample=False)
    x_dummy = as_Tensor(np.random.randn(8, 16, 28, 28).astype(np.float32))
    y_dummy = block(x_dummy)
    assert y_dummy.shape == x_dummy.shape, (
        f"ResidualBlock 形状不匹配: input={x_dummy.shape}, output={y_dummy.shape}"
    )

    dummy_loss = y_dummy * y_dummy
    block.cleargrads()
    dummy_loss.backward()
    has_grad = any((p.grad is not None) for p in block.params())
    assert has_grad, "ResidualBlock 反向传播后参数梯度为空"
    print("[PASS] ResidualBlock 形状与反向传播检查通过")

    # 2.1) 下采样专项测试
    test_residualblock_downsample_special()

    # 3) 用简单ResNet在MNIST上进行训练可学习性测试
    model = TinyResNetForMNIST(num_classes=10)
    optimizer = Adam(model.params(), lr=0.001)
    train_loader = SimpleDataLoader(NumpyDataset(x_train, y_train), batch_size=64, shuffle=True)

    init_loss, init_acc = evaluate(model, x_test, y_test)
    print(f"[INIT] loss={init_loss:.4f}, acc={init_acc:.4f}")

    epochs = 3
    for epoch in range(epochs):
        with Config.using_config("train", True):
            for xb, yb in train_loader:
                logits = model(as_Tensor(xb))
                loss = crossEntropyError(logits, yb)
                model.cleargrads()
                loss.backward()
                optimizer.step()

        val_loss, val_acc = evaluate(model, x_test, y_test)
        print(f"[EPOCH {epoch + 1}/{epochs}] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    final_loss, final_acc = evaluate(model, x_test, y_test)
    print(f"[FINAL] loss={final_loss:.4f}, acc={final_acc:.4f}")

    # 4) 验收标准：准确率明显高于随机猜测(10%)且较初始有提升
    assert final_acc > 0.20, f"最终准确率过低: {final_acc:.4f}"
    assert final_acc >= init_acc + 0.05, (
        f"训练后准确率未明显提升: init={init_acc:.4f}, final={final_acc:.4f}"
    )
    print("[PASS] TinyResNet + ResidualBlock 在MNIST子集上通过可学习性测试")


if __name__ == "__main__":
    main()
