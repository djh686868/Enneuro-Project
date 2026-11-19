# 基于EnNeuro框架的自动驾驶模型
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
        # 调整输入形状 (batch, 3, 120, 160)
        if x.shape[1:] != (3, 120, 160):
            x = F.reshape(x, (x.shape[0], 3, 120, 160))
        
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


class AutoDriveNet(Module):
    """端到端自动驾驶模型 - 轻量级版本"""
    def __init__(self, in_channels=3, num_classes=1):
        """初始化"""
        super().__init__()
        
        # 卷积层
        self.conv1 = Conv2d(24, kernel_size=5, stride=2, pad=0, in_channels=in_channels)
        self.bn1 = BatchNorm(24)
        
        self.conv2 = Conv2d(36, kernel_size=5, stride=2, pad=0, in_channels=24)
        self.bn2 = BatchNorm(36)
        
        self.conv3 = Conv2d(48, kernel_size=5, stride=2, pad=0, in_channels=36)
        self.bn3 = BatchNorm(48)
        
        self.conv4 = Conv2d(64, kernel_size=3, stride=1, pad=0, in_channels=48)
        self.bn4 = BatchNorm(64)
        
        self.conv5 = Conv2d(64, kernel_size=3, stride=1, pad=0, in_channels=64)
        self.bn5 = BatchNorm(64)
        
        # 全连接层
        self.fc1 = Linear(100)
        self.fc2 = Linear(50)
        self.fc3 = Linear(10)
        self.fc4 = Linear(num_classes)
    
    def forward(self, x):
        """前向推理"""
        # 调整输入形状 (batch, 3, 120, 160)
        if x.shape[1:] != (3, 120, 160):
            x = F.reshape(x, (x.shape[0], 3, 120, 160))
        
        # 卷积模块: Conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # 展平
        x = F.flatten(x)
        
        # 线性变换模块: Linear -> ReLU (最后一层除外)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        
        return x
