# 基于EnNeuro框架的ResNet18自动驾驶模型
from eneuro.nn.module import Module, Conv2d, Linear, BatchNorm, ResidualBlock, Sequential
from eneuro.base import functions as F


class ResNet18AutoDrive(Module):
    """基于ResNet18架构的端到端自动驾驶模型"""
    def __init__(self, in_channels=3, num_classes=1):
        """初始化"""
        super().__init__()
        
        # 初始卷积层
        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=in_channels)
        self.bn1 = BatchNorm(64)
        
        # 残差块组
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全连接层
        self.fc = Linear(num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """创建残差块层"""
        layers = []
        
        # 第一个残差块可能需要下采样
        downsample = stride != 1 or in_channels != out_channels
        layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, 
                                    stride=stride, downsample=downsample))
        
        # 后续的残差块
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels,
                                        stride=1, downsample=False))
        
        return Sequential(*layers)
    
    def forward(self, x):
        """前向推理"""
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 残差块组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = F.global_average_pooling(x)
        
        # 展平
        x = F.flatten(x)
        
        # 全连接层
        x = self.fc(x)
        
        return x
