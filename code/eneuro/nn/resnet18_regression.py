"""用于 RGB 图像回归的 ResNet18 模型。"""

from __future__ import annotations

from ..base import functions as F
from .module import BatchNorm, Conv2d, Linear, Module, ResidualBlock, Sequential


class ResNet18Regressor(Module):
    """沿用 DonkeyCar 的 ResNet18 骨干，支持一个或多个连续输出。

    输入形状为 (N, 3, H, W)。首层 7×7 卷积和最大池化各做一次下采样，
    之后经过四个残差阶段。全局平均池化后接线性回归头，输出形状为
    (N, out_features)。预测值的单位和范围由调用方的目标变换决定。
    """

    def __init__(self, out_features: int = 1):
        super().__init__()
        if out_features < 1:
            raise ValueError("out_features must be positive")

        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=3)
        self.bn1 = BatchNorm(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.fc = Linear(out_features)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, *, stride: int) -> Sequential:
        downsample = stride != 1 or in_channels != out_channels
        return Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample),
            ResidualBlock(out_channels, out_channels, stride=1, downsample=False),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pooling(x, kernel_size=3, stride=2, pad=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(F.flatten(F.global_average_pooling(x)))
