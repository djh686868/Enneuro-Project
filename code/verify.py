import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eneuro.base import Tensor
from eneuro.nn.module import MLP
from eneuro.nn.loss import CrossEntropyLoss
from eneuro.nn.optim import SGD
from eneuro.base.functions import relu

# 创建简单的测试数据
X = np.random.randn(2, 4).astype(np.float32)
y = np.array([0, 1], dtype=np.int32)

# 创建模型
model = MLP([4, 12, 3], activation=relu)

# 创建损失函数和优化器
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.params(), lr=0.1)

# 测试前向传播
print("测试前向传播...")
y_hat = model(Tensor(X))
print(f"前向传播结果: {y_hat}")

# 测试损失计算
print("\n测试损失计算...")
loss = loss_fn(y_hat, Tensor(y))
print(f"损失值: {loss}")

# 测试反向传播
print("\n测试反向传播...")
loss.backward()
print("反向传播完成")

# 测试参数更新
print("\n测试参数更新...")
optimizer.step()
print("参数更新完成")

print("\n所有测试完成！")
