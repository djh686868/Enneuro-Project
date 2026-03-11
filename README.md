# EnNeuro: A Lightweight CNN Framework

## 项目介绍

EnNeuro是一个轻量级的卷积神经网络(CNN)框架，专为深度学习研究和教育目的设计。它提供了简洁易用的API，支持构建、训练和评估各种神经网络模型，包括全连接网络和卷积神经网络。

## 主要特性

- **简洁易用的API**：类似于PyTorch的设计风格，易于学习和使用
- **支持多种神经网络层**：线性层、卷积层、池化层、 dropout层等
- **丰富的激活函数**：ReLU、Sigmoid、Softmax等
- **多种损失函数**：均方误差、交叉熵、sigmoid交叉熵等
- **多种优化器**：SGD、MomentumSGD、Adam等
- **支持L1和L2正则化**：防止过拟合
- **模型保存和加载**：支持模型参数的序列化和反序列化
- **数据加载和处理**：内置Dataset和DataLoader类
- **训练可视化**：支持训练过程的可视化

## 安装

### 环境要求

- Python 3.7+
- NumPy
- OpenCV (可选，用于可视化)

### 安装步骤

1. 克隆项目到本地：

```bash
git clone https://github.com/yourusername/enneuro.git
cd enneuro
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用示例

```python
import numpy as np
from eneuro.nn import Module, Linear, Sequential
from eneuro.nn.optim import SGD
from eneuro.nn.loss import meanSquaredError
from eneuro.base import Tensor

# 创建一个简单的线性模型
model = Sequential(
    Linear(10),
    Linear(1)
)

# 创建优化器
optimizer = SGD(model.get_params_list(), lr=0.01)

# 准备训练数据
x = Tensor(np.random.randn(100, 10))
y = Tensor(np.random.randn(100, 1))

# 训练模型
for epoch in range(100):
    # 前向传播
    y_pred = model(x)
    # 计算损失
    loss = meanSquaredError(y_pred, y)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')
```

### 卷积神经网络示例

```python
from eneuro.nn import Sequential, Conv2d, Linear
from eneuro.base import relu, pooling, softmax

# 创建一个简单的CNN模型
model = Sequential(
    Conv2d(32, 3, pad=1),  # 卷积层
    relu,                   # 激活函数
    pooling(2, 2),          # 池化层
    Conv2d(64, 3, pad=1),
    relu,
    pooling(2, 2),
    Linear(128),
    relu,
    Linear(10),
    softmax                 # Softmax输出
)
```

## 核心模块

### nn 模块

- **Layer**：所有网络层的基类
- **Module**：可序列化的网络模块
- **Sequential**：顺序容器，按顺序执行网络层
- **Linear**：线性全连接层
- **Conv2d**：二维卷积层
- **Dropout**： dropout层，用于防止过拟合
- **MLP**：多层感知机
- **CNNWithPooling**：带有池化层的卷积神经网络

### loss 模块

- **MSELoss**：均方误差损失
- **SoftmaxWithLoss**：softmax+交叉熵损失
- **SigmoidWithLoss**：sigmoid+交叉熵损失
- **CrossEntropyLoss**：交叉熵损失

### optim 模块

- **Optimizer**：优化器基类
- **SGD**：随机梯度下降优化器
- **MomentumSGD**：带动量的随机梯度下降优化器
- **Adam**：Adam优化器

### data 模块

- **Dataset**：数据集基类
- **DataLoader**：数据加载器，支持批量加载和多线程

### train 模块

- **Trainer**：训练器，用于模型训练
- **Meters**：用于跟踪训练指标

### utils 模块

- **StateDict**：状态字典，用于模型参数的序列化和反序列化
- **visualization**：可视化工具

## API文档

### 模型定义

```python
class Module(Layer, StateDict):
    """模型基类，继承自Layer和StateDict"""
    def forward(self, inputs):
        """前向传播"""
        raise NotImplementedError
    
    def to_dict(self):
        """将模型参数转换为字典"""
        pass
    
    def from_dict(self, d):
        """从字典加载模型参数"""
        pass
```

### 优化器使用

```python
optimizer = SGD(model.get_params_list(), lr=0.01, l2_lambda=0.001)

# 训练步骤
optimizer.zero_grad()  # 清空梯度
loss.backward()        # 反向传播
optimizer.step()       # 更新参数
```

### 损失函数使用

```python
loss = meanSquaredError(y_pred, y_true)
loss = softmaxCrossEntropy(y_pred, y_true)
```

## 示例

### 训练MNIST分类模型

```python
from eneuro.data import Dataset, DataLoader
from eneuro.nn import Sequential, Conv2d, Linear
from eneuro.nn.optim import Adam
from eneuro.nn.loss import softmaxCrossEntropy
from eneuro.base import relu, pooling, softmax
from eneuro.train import Trainer

# 准备数据
train_dataset = Dataset(...)  # 加载MNIST数据集
train_loader = DataLoader(train_dataset, batch_size=64)

# 创建模型
model = Sequential(
    Conv2d(32, 3, pad=1),
    relu,
    pooling(2, 2),
    Conv2d(64, 3, pad=1),
    relu,
    pooling(2, 2),
    Linear(128),
    relu,
    Linear(10),
    softmax
)

# 创建优化器
optimizer = Adam(model.get_params_list(), lr=0.001)

# 创建训练器
trainer = Trainer(model, optimizer, softmaxCrossEntropy)

# 训练模型
trainer.train(train_loader, epochs=10, validate=True)
```

## 贡献指南

我们欢迎社区贡献！如果您想为EnNeuro做出贡献，请按照以下步骤：

1. Fork 项目仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件

## 联系方式

- 项目链接：https://github.com/djh686868/Enneuro-Project
- 问题反馈：https://github.com/djh686868/Enneuro-Project/issues

---

EnNeuro框架旨在提供一个轻量级、易于理解的深度学习框架，适合学习和研究深度学习的基本原理。我们希望它能够帮助开发者快速构建和训练神经网络模型，同时也能够作为深度学习教育的工具。