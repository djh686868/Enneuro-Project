# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np

# 确保能导入 code 下的包
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eneuro.base import as_Tensor, Config  # noqa: E402
from eneuro.base import functions as F      # noqa: E402
from eneuro.nn.module import (
    Module,
    Conv2d,
    BatchNorm,
    Linear,
    ResidualBlock,
    Sequential,
)  # noqa: E402


class ResNet18(Module):
    """基于项目中 `ResidualBlock` 的 ResNet-18 风格微型实现（用于测试）。"""

    def __init__(self, num_classes=1000):
        super().__init__()
        # 初始通道数基准
        self.in_channels = 64

        # stem: 7x7 conv, stride=2, pad=3
        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=3)
        self.bn1 = BatchNorm(64)

        # 4 个 stage，每个 stage 包含若干个 ResidualBlock
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        # 分类头
        self.fc = Linear(num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        # 首个 block 可能需要下采样/升维
        downsample = (stride != 1) or (self.in_channels != out_channels)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels

        for _ in range(blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1, downsample=False))

        return Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.pooling(x, kernel_size=3, stride=2, pad=1)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global pool -> fc
        x = F.global_average_pooling(x)
        x = F.flatten(x)
        x = self.fc(x)
        return x


def test_resnet18_forward_backward():
    """简单的前向与反向检查，确保形状与梯度通路正常。"""
    np.random.seed(0)
    model = ResNet18(num_classes=10)

    # 随机输入 (batch=2, 3, 224, 224)
    x = np.random.randn(2, 3, 224, 224).astype(np.float32)
    xt = as_Tensor(x)

    # 前向
    y = model(xt)
    assert y.shape == (2, 10), f"ResNet18 输出形状错误: got={y.shape}"

    # 反向（简单平方和）并检查部分参数是否有梯度
    loss = y * y
    model.cleargrads()
    loss.backward()

    # 检查少量关键层是否有梯度
    assert model.conv1.W.grad is not None, "conv1.W 未获得梯度"
    # 检查某个 stage 的第一个 block 的 conv1 是否有梯度
    first_block = model.layer2.layers[0]
    assert first_block.conv1.W.grad is not None, "layer2 first block conv1 未获得梯度"

    print("[PASS] test_resnet18_forward_backward")


if __name__ == '__main__':
    test_resnet18_forward_backward()
